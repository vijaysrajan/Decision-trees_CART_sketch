#!/usr/bin/env python3
"""
Simple converter from 2-column class-based sketches to 3-column training format.

This script takes the output from create_2col_sketches.py and converts it directly
to the 3-column format needed for training.

Usage:
    python tools/simple_convert_to_3col.py \
        positive_class_2col.csv negative_class_2col.csv \
        feature_mapping.json output_dir

Example:
    python tools/simple_convert_to_3col.py \
        agaricus_lepiota_sketches/agaricus_lepiota_positive_2col_sketches_lg_k_12.csv \
        agaricus_lepiota_sketches/agaricus_lepiota_negative_2col_sketches_lg_k_12.csv \
        agaricus_lepiota_sketches/agaricus_lepiota_feature_mapping.json \
        agaricus_lepiota_sketches/
"""

import argparse
import csv
import sys
import os
import base64
import json
from pathlib import Path
from typing import Dict, List

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from datasketches import compact_theta_sketch, theta_a_not_b
    from tests.test_binary_classification_sketches import ThetaSketchWrapper
except ImportError:
    raise ImportError("Apache DataSketches library required. Install with: pip install datasketches")


def load_sketches_from_csv(csv_file: str) -> Dict[str, ThetaSketchWrapper]:
    """Load 2-column sketch CSV into dictionary."""
    sketches = {}

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            feature_name, sketch_data = row
            sketch_bytes = base64.b64decode(sketch_data)
            compact_sketch = compact_theta_sketch.deserialize(sketch_bytes)
            sketches[feature_name] = ThetaSketchWrapper(compact_sketch)

    return sketches


def save_3col_sketches(positive_sketches: Dict, negative_sketches: Dict,
                      feature_mapping: Dict, output_dir: str, dataset_name: str) -> None:
    """Save sketches in 3-column training format."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create 3-column data
    csv_data = []

    # Get all unique features from both classes
    all_features = set(positive_sketches.keys()) | set(negative_sketches.keys())

    for feature_name in sorted(all_features):
        # Get present/absent sketches from both classes
        pos_present = positive_sketches.get(feature_name)
        neg_present = negative_sketches.get(feature_name)

        # For absent sketches, we need to create them by taking the complement
        # For now, we'll use None and handle this in the tree building logic
        pos_absent = None  # Will be computed by tree builder
        neg_absent = None  # Will be computed by tree builder

        if pos_present:
            pos_present_b64 = base64.b64encode(pos_present._sketch.serialize()).decode('utf-8')
        else:
            pos_present_b64 = ""

        if neg_present:
            neg_present_b64 = base64.b64encode(neg_present._sketch.serialize()).decode('utf-8')
        else:
            neg_present_b64 = ""

        csv_data.append({
            'identifier': feature_name,
            'sketch_feature_present_positive': pos_present_b64,
            'sketch_feature_present_negative': neg_present_b64
        })

    # Save 3-column CSV
    csv_filename = f"{dataset_name}_3col_sketches.csv"
    csv_filepath = output_path / csv_filename

    with open(csv_filepath, 'w', newline='') as f:
        if csv_data:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)

    print(f"📄 Saved: {csv_filename} ({len(csv_data)} features)")

    # Also save the feature mapping
    mapping_filename = f"{dataset_name}_feature_mapping.json"
    mapping_filepath = output_path / mapping_filename

    with open(mapping_filepath, 'w') as f:
        json.dump(feature_mapping, f, indent=2)

    print(f"🗺️ Saved: {mapping_filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert class-based 2-column sketches to 3-column training format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert mushroom sketches
  python tools/simple_convert_to_3col.py \\
      agaricus_lepiota_sketches/agaricus_lepiota_positive_2col_sketches_lg_k_12.csv \\
      agaricus_lepiota_sketches/agaricus_lepiota_negative_2col_sketches_lg_k_12.csv \\
      agaricus_lepiota_sketches/agaricus_lepiota_feature_mapping.json \\
      agaricus_lepiota_sketches/
        """)

    parser.add_argument('positive_csv', type=str, help='Positive class 2-column CSV file')
    parser.add_argument('negative_csv', type=str, help='Negative class 2-column CSV file')
    parser.add_argument('feature_mapping', type=str, help='Feature mapping JSON file')
    parser.add_argument('output_dir', type=str, help='Output directory')

    args = parser.parse_args()

    print("📊 Simple 2-Column to 3-Column Converter")
    print("=" * 50)

    # Load feature mapping
    try:
        with open(args.feature_mapping, 'r') as f:
            feature_mapping = json.load(f)
        print(f"🗺️ Loaded feature mapping: {len(feature_mapping)} features")
    except Exception as e:
        print(f"❌ Error loading feature mapping: {e}")
        return 1

    # Load sketches
    try:
        print(f"📁 Loading positive sketches: {args.positive_csv}")
        positive_sketches = load_sketches_from_csv(args.positive_csv)

        print(f"📁 Loading negative sketches: {args.negative_csv}")
        negative_sketches = load_sketches_from_csv(args.negative_csv)

        print(f"✅ Loaded {len(positive_sketches)} positive and {len(negative_sketches)} negative sketches")

    except Exception as e:
        print(f"❌ Error loading sketches: {e}")
        return 1

    # Generate dataset name from input file
    dataset_name = Path(args.positive_csv).stem.replace('_positive_2col_sketches_lg_k_12', '')

    # Convert and save
    try:
        save_3col_sketches(positive_sketches, negative_sketches, feature_mapping,
                          args.output_dir, dataset_name)
        print(f"\n🎉 Conversion complete!")
        print(f"📂 Output directory: {args.output_dir}")

    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())