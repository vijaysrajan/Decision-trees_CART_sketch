#!/usr/bin/env python3
"""
Convert 2-column theta sketches to 3-column format (positive and negative).

This script takes a 2-column CSV file with sketches and converts it to:
1. Two 3-column CSV files (positive and negative)
2. A feature mapping JSON file

Usage:
    python tools/simple_convert_to_3col.py \
        --input 2col_sketches.csv \
        --suffix _3col_sketches \
        --mapping feature_mapping.json \
        --output output_dir/ \
        --target tripOutcome

Example:
    python tools/simple_convert_to_3col.py \
        --input DU_output/DU_raw_2col_sketches_lg_k_16.csv \
        --suffix _3col_sketches \
        --mapping DU_feature_mapping.json \
        --output DU_output/ \
        --target tripOutcome
"""

import argparse
import csv
import sys
import os
import json
import base64
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from datasketches import compact_theta_sketch, theta_intersection, theta_a_not_b
except ImportError:
    raise ImportError("Apache DataSketches library required. Install with: pip install datasketches")


def load_2col_sketches(csv_file: str) -> Tuple[str, Dict[str, str]]:
    """
    Load 2-column CSV and return total sketch and feature sketches.

    Returns:
        total_sketch_b64: Base64 encoded total sketch (from empty feature name row)
        feature_sketches: Dict mapping feature_name -> base64_sketch
    """
    # Increase CSV field size limit to handle large sketches
    csv.field_size_limit(5000000)  # 5MB limit

    total_sketch_b64 = None
    feature_sketches = {}

    print(f"📁 Loading 2-column sketches from: {csv_file}")

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        row_count = 0

        for row in reader:
            feature_name = row['feature']
            sketch_b64 = row['sketch']

            if feature_name == "":
                # This is the total sketch (all records)
                total_sketch_b64 = sketch_b64
                print(f"   📊 Found total sketch (all records)")
            else:
                # This is a feature sketch
                feature_sketches[feature_name] = sketch_b64
                row_count += 1

                if row_count % 100 == 0:
                    print(f"   Loaded {row_count} feature sketches...")

    print(f"✅ Loaded total sketch + {len(feature_sketches)} feature sketches")

    if total_sketch_b64 is None:
        raise ValueError("No total sketch found (missing empty feature name row)")

    return total_sketch_b64, feature_sketches


def create_feature_mapping(feature_sketches: Dict[str, str]) -> Dict[str, int]:
    """
    Create feature mapping from feature names to indices.

    Returns:
        feature_mapping: Dict mapping feature_name -> column_index
    """
    print("🗺️ Creating feature mapping...")

    # Sort feature names for consistent ordering
    sorted_features = sorted(feature_sketches.keys())

    feature_mapping = {}
    for idx, feature_name in enumerate(sorted_features):
        feature_mapping[feature_name] = idx

    print(f"✅ Created mapping for {len(feature_mapping)} features")
    return feature_mapping


def save_feature_mapping(feature_mapping: Dict[str, int], output_dir: str, mapping_filename: str):
    """Save feature mapping to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mapping_filepath = output_path / mapping_filename

    print(f"💾 Saving feature mapping to: {mapping_filepath}")

    with open(mapping_filepath, 'w') as f:
        json.dump(feature_mapping, f, indent=2, sort_keys=True)

    print(f"✅ Saved feature mapping with {len(feature_mapping)} features")
    return mapping_filepath


def parse_target_classes(target_param: str, feature_sketches: Dict[str, str]) -> Tuple[str, str]:
    """
    Parse target parameter and return positive and negative class feature names.

    Args:
        target_param: Either 'tripOutcome' or 'tripOutcome=good,tripOutcome=bad'
        feature_sketches: Dict of all feature sketches

    Returns:
        (positive_class_feature, negative_class_feature)
    """
    if ',' in target_param:
        # Option B: Explicit positive,negative specification
        parts = [part.strip() for part in target_param.split(',')]
        if len(parts) != 2:
            raise ValueError(f"Target parameter must have exactly 2 values when comma-separated: {target_param}")
        positive_class, negative_class = parts

        # Validate both exist in feature sketches
        if positive_class not in feature_sketches:
            raise ValueError(f"Positive class feature not found: {positive_class}")
        if negative_class not in feature_sketches:
            raise ValueError(f"Negative class feature not found: {negative_class}")

        print(f"   📊 Explicit classes: positive={positive_class}, negative={negative_class}")
        return positive_class, negative_class

    else:
        # Option A: Auto-discover from target name
        target_name = target_param.strip()
        matching_features = []

        for feature_name in feature_sketches.keys():
            if feature_name.startswith(f"{target_name}="):
                matching_features.append(feature_name)

        if len(matching_features) != 2:
            raise ValueError(f"Expected exactly 2 target features starting with '{target_name}=', found {len(matching_features)}: {matching_features}")

        # Sort to get consistent order (first in file order)
        sorted_features = sorted(matching_features, key=lambda x: list(feature_sketches.keys()).index(x))
        positive_class, negative_class = sorted_features[0], sorted_features[1]

        print(f"   📊 Auto-discovered classes: positive={positive_class}, negative={negative_class}")
        return positive_class, negative_class


def save_3col_csv(output_dir: str, suffix: str, total_sketch_b64: str,
                 feature_sketches: Dict[str, str], class_feature: str, class_type: str,
                 exclude_features: set):
    """
    Save 3-column CSV file with proper intersection and theta_a_not_b logic.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    output_filename = f"{class_type}{suffix}.csv"
    output_filepath = output_path / output_filename

    print(f"💾 Creating {class_type} file: {output_filepath}")

    # Deserialize sketches for theta operations
    total_sketch = compact_theta_sketch.deserialize(base64.b64decode(total_sketch_b64))
    class_sketch = compact_theta_sketch.deserialize(base64.b64decode(feature_sketches[class_feature]))

    csv_data = []

    # First row: total row
    a_not_b_op = theta_a_not_b()
    class_absent = a_not_b_op.compute(total_sketch, class_sketch)
    csv_data.append({
        "identifier": "total",
        "sketch_feature_present": base64.b64encode(class_sketch.serialize()).decode('utf-8'),
        "sketch_feature_absent": base64.b64encode(class_absent.serialize()).decode('utf-8')
    })

    # Add feature rows (excluding target features)
    feature_count = 0
    for feature_name, feature_sketch_b64 in sorted(feature_sketches.items()):
        if feature_name in exclude_features:
            continue  # Skip target features

        # Deserialize feature sketch
        feature_sketch = compact_theta_sketch.deserialize(base64.b64decode(feature_sketch_b64))

        # Compute intersection of class and feature
        intersection_op = theta_intersection()
        intersection_op.update(class_sketch)
        intersection_op.update(feature_sketch)
        feature_present = intersection_op.get_result()

        # Compute absent = total - feature_present
        a_not_b_op2 = theta_a_not_b()
        feature_absent = a_not_b_op2.compute(total_sketch, feature_present)

        csv_data.append({
            "identifier": feature_name,
            "sketch_feature_present": base64.b64encode(feature_present.serialize()).decode('utf-8'),
            "sketch_feature_absent": base64.b64encode(feature_absent.serialize()).decode('utf-8')
        })
        feature_count += 1

        if feature_count % 50 == 0:
            print(f"   Processed {feature_count} features...")

    # Write CSV file
    with open(output_filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['identifier', 'sketch_feature_present', 'sketch_feature_absent'])
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"✅ Created {class_type} file with {len(csv_data)} rows ({feature_count} features + 1 total)")
    return output_filepath


def main():
    parser = argparse.ArgumentParser(
        description='Convert 2-column theta sketches to 3-column format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python tools/simple_convert_to_3col.py \\
      --input sketches_2col.csv \\
      --suffix _3col_sketches \\
      --mapping feature_mapping.json \\
      --output output_dir/

  # With DU dataset (auto-discover target values)
  python tools/simple_convert_to_3col.py \\
      --input DU_output/DU_raw_2col_sketches_lg_k_16.csv \\
      --suffix _3col_sketches \\
      --mapping DU_feature_mapping.json \\
      --output DU_output/ \\
      --target tripOutcome

  # Explicit positive/negative classes
  python tools/simple_convert_to_3col.py \\
      --input DU_output/DU_raw_2col_sketches_lg_k_16.csv \\
      --suffix _3col_sketches \\
      --mapping DU_feature_mapping.json \\
      --output DU_output/ \\
      --target tripOutcome=good,tripOutcome=bad
        """)

    parser.add_argument('--input', type=str, required=True,
                       help='Input 2-column CSV file path')
    parser.add_argument('--suffix', type=str, required=True,
                       help='Suffix for output 3-column CSV files (e.g., _3col_sketches)')
    parser.add_argument('--mapping', type=str, required=True,
                       help='Output feature mapping JSON filename')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory path')
    parser.add_argument('--target', type=str, required=True,
                       help='Target variable: single name (auto-discover values) or comma-separated values (first=positive, second=negative)')

    args = parser.parse_args()

    print("🔄 Theta Sketch 2-Column to 3-Column Converter")
    print("=" * 50)

    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return 1

    print(f"📋 Configuration:")
    print(f"   Input: {args.input}")
    print(f"   Output directory: {args.output}")
    print(f"   Suffix: {args.suffix}")
    print(f"   Mapping file: {args.mapping}")
    print(f"   Target: {args.target}")

    try:
        # Load 2-column sketches
        total_sketch_b64, feature_sketches = load_2col_sketches(args.input)

        # Parse target classes
        positive_class, negative_class = parse_target_classes(args.target, feature_sketches)

        # Create feature mapping (excluding target features from mapping)
        exclude_features = {positive_class, negative_class}
        non_target_features = {k: v for k, v in feature_sketches.items() if k not in exclude_features}
        feature_mapping = create_feature_mapping(non_target_features)

        # Save feature mapping
        mapping_file = save_feature_mapping(feature_mapping, args.output, args.mapping)

        # Create 3-column files with proper theta operations
        positive_file = save_3col_csv(args.output, args.suffix, total_sketch_b64,
                                     feature_sketches, positive_class, "positive", exclude_features)
        negative_file = save_3col_csv(args.output, args.suffix, total_sketch_b64,
                                     feature_sketches, negative_class, "negative", exclude_features)

        print(f"\n🎉 Success!")
        print(f"📄 Feature mapping: {mapping_file}")
        print(f"📄 Positive sketches: {positive_file}")
        print(f"📄 Negative sketches: {negative_file}")
        print(f"\n✅ 3-column files contain proper intersection and theta_a_not_b computations.")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())