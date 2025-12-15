#!/usr/bin/env python3
"""
Create 2-column theta sketches from CSV data.

This script creates a 2-column CSV file with:
1. First row: feature_name="" (empty string), sketch=sketch_of_all_records
2. Subsequent rows: feature_name=column=value, sketch=sketch_of_records_with_that_feature

Usage:
    python tools/create_2col_sketches.py \
        --input data.csv \
        --output output_dir \
        --lg_k 12 \
        --target_column class \
        --skip_columns col1,col2 \
        --id_columns id,uuid

Example:
    python tools/create_2col_sketches.py \
        --input tests/resources/agaricus-lepiota.csv \
        --output sketches/ \
        --lg_k 12 \
        --target_column class \
        --skip_columns id \
        --id_columns sample_id
"""

import argparse
import csv
import sys
import os
import base64
import hashlib
from pathlib import Path
from typing import Dict, Set, List
import pandas as pd

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from datasketches import update_theta_sketch
except ImportError:
    raise ImportError("Apache DataSketches library required. Install with: pip install datasketches")


def hash_identifier(identifier: str) -> int:
    """Create a consistent hash of identifier for sketch updates."""
    return int(hashlib.md5(str(identifier).encode()).hexdigest()[:8], 16)


def load_csv_data(csv_file: str) -> pd.DataFrame:
    """Load CSV data with increased field size limit."""
    # Increase CSV field size limit to handle large data
    csv.field_size_limit(10000000)  # 10MB limit

    try:
        df = pd.read_csv(csv_file)
        print(f"📁 Loaded CSV: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {e}")


def get_feature_columns(df: pd.DataFrame, skip_columns: Set[str], id_columns: Set[str]) -> List[str]:
    """Get columns that should be processed as features."""
    all_columns = set(df.columns)
    excluded_columns = skip_columns | id_columns
    feature_columns = [col for col in df.columns if col not in excluded_columns]

    print(f"📊 Total columns: {len(all_columns)}")
    print(f"🚫 Excluded columns: {excluded_columns}")
    print(f"✅ Feature columns: {len(feature_columns)} ({feature_columns})")

    return feature_columns


def create_sketches(df: pd.DataFrame, feature_columns: List[str], lg_k: int) -> Dict[str, update_theta_sketch]:
    """Create theta sketches for all records and each feature."""
    sketches = {}

    # Create sketch for ALL records (empty string key)
    print("🔧 Creating sketch for all records...")
    all_records_sketch = update_theta_sketch(lg_k)
    for idx in df.index:
        all_records_sketch.update(hash_identifier(str(idx)))
    sketches[""] = all_records_sketch
    print(f"   Total records sketch: {all_records_sketch.get_estimate():,.0f} estimated items")

    # Create sketches for each feature=value combination
    print("🔧 Creating feature sketches...")
    feature_count = 0

    for col in feature_columns:
        unique_values = df[col].unique()
        print(f"   Processing column '{col}': {len(unique_values)} unique values")

        for value in unique_values:
            # Convert value to string (treat everything as categorical)
            value_str = str(value)
            feature_key = f"{col}={value_str}"

            # Create sketch for records with this feature=value
            feature_sketch = update_theta_sketch(lg_k)
            matching_rows = df[df[col] == value]

            for idx in matching_rows.index:
                feature_sketch.update(hash_identifier(str(idx)))

            sketches[feature_key] = feature_sketch
            feature_count += 1

            if feature_count % 50 == 0:  # Progress indicator
                print(f"   Created {feature_count} feature sketches...")

    print(f"✅ Created {len(sketches)} total sketches (1 all-records + {feature_count} features)")
    return sketches


def save_2col_csv(sketches: Dict[str, update_theta_sketch], output_dir: str, input_filename: str, lg_k: int):
    """Save sketches to 2-column CSV format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    dataset_name = Path(input_filename).stem
    output_filename = f"{dataset_name}_2col_sketches_lg_k_{lg_k}.csv"
    output_filepath = output_path / output_filename

    print(f"💾 Saving to: {output_filepath}")

    # Prepare CSV data
    csv_data = []

    # First add the all-records sketch (empty string feature name)
    if "" in sketches:
        all_sketch = sketches[""]
        compact_sketch = all_sketch.compact()
        sketch_b64 = base64.b64encode(compact_sketch.serialize()).decode('utf-8')
        csv_data.append({"feature": "", "sketch": sketch_b64})

    # Then add all feature sketches in sorted order
    feature_keys = [key for key in sketches.keys() if key != ""]
    for feature_key in sorted(feature_keys):
        feature_sketch = sketches[feature_key]
        compact_sketch = feature_sketch.compact()
        sketch_b64 = base64.b64encode(compact_sketch.serialize()).decode('utf-8')
        csv_data.append({"feature": feature_key, "sketch": sketch_b64})

    # Write CSV file
    with open(output_filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['feature', 'sketch'])
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"✅ Saved {len(csv_data)} sketches to {output_filename}")
    return output_filepath


def main():
    parser = argparse.ArgumentParser(
        description='Create 2-column theta sketches from CSV data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python tools/create_2col_sketches.py \\
      --input data.csv \\
      --output sketches/ \\
      --lg_k 12

  # With exclusions
  python tools/create_2col_sketches.py \\
      --input tests/resources/agaricus-lepiota.csv \\
      --output mushroom_sketches/ \\
      --lg_k 12 \\
      --skip_columns id,timestamp \\
      --id_columns sample_id,uuid
        """)

    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output directory path')
    parser.add_argument('--lg_k', type=int, required=True, help='Theta sketch lg_k parameter (sketch size)')
    parser.add_argument('--skip_columns', type=str, help='Comma-separated list of columns to skip')
    parser.add_argument('--id_columns', type=str, help='Comma-separated list of ID columns to exclude')

    args = parser.parse_args()

    print("🎯 Theta Sketch 2-Column Generator")
    print("=" * 50)

    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return 1

    # Parse column exclusions
    skip_columns = set()
    if args.skip_columns:
        skip_columns = set(col.strip() for col in args.skip_columns.split(','))

    id_columns = set()
    if args.id_columns:
        id_columns = set(col.strip() for col in args.id_columns.split(','))

    print(f"📋 Configuration:")
    print(f"   Input: {args.input}")
    print(f"   Output: {args.output}")
    print(f"   lg_k: {args.lg_k}")
    print(f"   Skip columns: {skip_columns or 'None'}")
    print(f"   ID columns: {id_columns or 'None'}")

    try:
        # Load data
        df = load_csv_data(args.input)

        # Get feature columns
        feature_columns = get_feature_columns(df, skip_columns, id_columns)

        if not feature_columns:
            print("❌ No feature columns found after exclusions!")
            return 1

        # Create sketches
        sketches = create_sketches(df, feature_columns, args.lg_k)

        # Save output
        output_file = save_2col_csv(sketches, args.output, args.input, args.lg_k)

        print(f"\n🎉 Success!")
        print(f"📄 Generated: {output_file}")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())