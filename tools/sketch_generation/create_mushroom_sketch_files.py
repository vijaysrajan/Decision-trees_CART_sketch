#!/usr/bin/env python3
"""
Generate real theta sketch files from mushroom dataset.

This script creates CSV files with feature sketches that can be used
to replace mocks in tests with real data.

Usage:
    python create_mushroom_sketch_files.py [--lg_k LG_K_VALUE]

Examples:
    python create_mushroom_sketch_files.py --lg_k 11
    python create_mushroom_sketch_files.py --lg_k 16
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datasketches import update_theta_sketch, compact_theta_sketch, theta_intersection, theta_union
import csv
import io
import base64
import argparse

DEFAULT_LG_K = 11


class ThetaSketchWrapper:
    """
    Wrapper for Apache DataSketches that adds intersection method for tree builder compatibility.
    """

    def __init__(self, sketch):
        self._sketch = sketch

    def get_estimate(self):
        """Return estimated cardinality."""
        return self._sketch.get_estimate()

    def update(self, item):
        """Add item to sketch."""
        return self._sketch.update(item)

    def intersection(self, other):
        """Compute intersection with another sketch using Apache DataSketches API."""
        if isinstance(other, ThetaSketchWrapper):
            other_sketch = other._sketch
        else:
            other_sketch = other

        intersector = theta_intersection()
        intersector.update(self._sketch)
        intersector.update(other_sketch)
        result = intersector.get_result()

        return ThetaSketchWrapper(result)

    def __getattr__(self, name):
        """Delegate other methods to the underlying sketch."""
        return getattr(self._sketch, name)


def serialize_sketch_to_string(sketch):
    """Serialize a theta sketch to a base64 string."""
    try:
        # Convert update sketch to compact representation if needed
        if hasattr(sketch, 'compact'):
            compact_sketch = sketch.compact()
        else:
            compact_sketch = sketch

        # Get the serialized bytes
        compact_bytes = compact_sketch.serialize()
        # Encode to base64 for storage in CSV
        return base64.b64encode(compact_bytes).decode('ascii')
    except Exception as e:
        print(f"Error serializing sketch: {e}")
        return ""


def deserialize_sketch_from_string(sketch_str, lg_k=DEFAULT_LG_K):
    """Deserialize a theta sketch from a base64 string."""
    try:
        # Decode from base64
        compact_bytes = base64.b64decode(sketch_str.encode('ascii'))
        # Create compact sketch from bytes
        compact_sketch = compact_theta_sketch.deserialize(compact_bytes)
        return compact_sketch
    except Exception as e:
        print(f"Error deserializing sketch: {e}")
        return update_theta_sketch(lg_k=lg_k)


def load_mushroom_dataset():
    """Load and prepare the mushroom dataset."""
    # Path relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    data_path = project_root / "tests/resources/agaricus-lepiota.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Mushroom dataset not found at {data_path}")

    print(f"Loading mushroom dataset from: {data_path}")
    df = pd.read_csv(data_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Map class values
    print(f"Class values found: {df['class'].unique()}")
    df['class'] = df['class'].map({'p': 'poisonous', 'e': 'edible'})
    print(f"Mapped p->poisonous, e->edible")

    class_counts = df['class'].value_counts()
    print(f"Final class distribution: {class_counts.to_dict()}")

    return df


def create_binary_features(df):
    """Create binary features from categorical columns."""
    print("Creating binary features from categorical data...")

    # Separate features from target
    feature_cols = [col for col in df.columns if col != 'class']
    target_col = 'class'

    binary_features = {}
    feature_mapping = {}
    feature_idx = 0

    for col in feature_cols:
        unique_values = df[col].unique()
        print(f"  Column '{col}' has values: {unique_values}")

        for value in unique_values:
            if pd.isna(value):
                continue

            feature_name = f"{col}={value}"
            binary_features[feature_name] = (df[col] == value).astype(int)
            feature_mapping[feature_name] = feature_idx
            feature_idx += 1

    print(f"Created {len(binary_features)} binary features")
    return binary_features, feature_mapping


def create_theta_sketches_with_ids(df, binary_features, lg_k=DEFAULT_LG_K):
    """Create theta sketches for each class and feature combination."""
    print(f"Creating theta sketches with lg_k={lg_k}")

    # Add row IDs starting from 1
    df = df.copy()
    df['id'] = range(1, len(df) + 1)

    # Separate by class
    positive_df = df[df['class'] == 'poisonous'].copy()
    negative_df = df[df['class'] == 'edible'].copy()

    print(f"Positive class (poisonous): {len(positive_df)} samples")
    print(f"Negative class (edible): {len(negative_df)} samples")

    sketch_data = {
        'positive': {},
        'negative': {}
    }

    # Create total sketches
    print("Creating total class sketches...")

    # Positive total sketch
    pos_total_sketch = update_theta_sketch(lg_k=lg_k)
    for row_id in positive_df['id']:
        pos_total_sketch.update(str(row_id))
    sketch_data['positive']['total'] = pos_total_sketch

    # Negative total sketch
    neg_total_sketch = update_theta_sketch(lg_k=lg_k)
    for row_id in negative_df['id']:
        neg_total_sketch.update(str(row_id))
    sketch_data['negative']['total'] = neg_total_sketch

    print(f"Positive total sketch estimate: {pos_total_sketch.get_estimate()}")
    print(f"Negative total sketch estimate: {neg_total_sketch.get_estimate()}")

    # Create feature sketches
    print("Creating feature sketches...")

    for feature_name, feature_values in binary_features.items():
        print(f"  Processing feature: {feature_name}")

        # Positive class feature sketches
        pos_present_sketch = update_theta_sketch(lg_k=lg_k)
        pos_absent_sketch = update_theta_sketch(lg_k=lg_k)

        for idx, row in positive_df.iterrows():
            row_id = row['id']
            if feature_values.iloc[idx] == 1:  # Feature present
                pos_present_sketch.update(str(row_id))
            else:  # Feature absent
                pos_absent_sketch.update(str(row_id))

        sketch_data['positive'][feature_name] = (pos_present_sketch, pos_absent_sketch)

        # Negative class feature sketches
        neg_present_sketch = update_theta_sketch(lg_k=lg_k)
        neg_absent_sketch = update_theta_sketch(lg_k=lg_k)

        for idx, row in negative_df.iterrows():
            row_id = row['id']
            if feature_values.iloc[idx] == 1:  # Feature present
                neg_present_sketch.update(str(row_id))
            else:  # Feature absent
                neg_absent_sketch.update(str(row_id))

        sketch_data['negative'][feature_name] = (neg_present_sketch, neg_absent_sketch)

    return sketch_data


def save_sketches_to_csv(sketch_data, positive_file="positive_sketches.csv", negative_file="negative_sketches.csv"):
    """Save sketch data to CSV files."""
    print(f"Saving sketches to CSV files...")

    # Save positive class sketches
    with open(positive_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['feature', 'sketch_type', 'sketch_data'])

        # Total sketch
        total_sketch_str = serialize_sketch_to_string(sketch_data['positive']['total'])
        writer.writerow(['total', 'total', total_sketch_str])

        # Feature sketches
        for feature_name, feature_data in sketch_data['positive'].items():
            if feature_name != 'total':
                present_sketch, absent_sketch = feature_data
                present_str = serialize_sketch_to_string(present_sketch)
                absent_str = serialize_sketch_to_string(absent_sketch)
                writer.writerow([feature_name, 'present', present_str])
                writer.writerow([feature_name, 'absent', absent_str])

    # Save negative class sketches
    with open(negative_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['feature', 'sketch_type', 'sketch_data'])

        # Total sketch
        total_sketch_str = serialize_sketch_to_string(sketch_data['negative']['total'])
        writer.writerow(['total', 'total', total_sketch_str])

        # Feature sketches
        for feature_name, feature_data in sketch_data['negative'].items():
            if feature_name != 'total':
                present_sketch, absent_sketch = feature_data
                present_str = serialize_sketch_to_string(present_sketch)
                absent_str = serialize_sketch_to_string(absent_sketch)
                writer.writerow([feature_name, 'present', present_str])
                writer.writerow([feature_name, 'absent', absent_str])

    print(f"  Positive sketches saved to: {positive_file}")
    print(f"  Negative sketches saved to: {negative_file}")


def load_sketches_from_csv(positive_file="positive_sketches.csv", negative_file="negative_sketches.csv", lg_k=DEFAULT_LG_K):
    """Load sketch data from CSV files."""
    print(f"Loading sketches from CSV files...")

    sketch_data = {
        'positive': {},
        'negative': {}
    }

    # Load positive sketches
    with open(positive_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            feature = row['feature']
            sketch_type = row['sketch_type']
            sketch_str = row['sketch_data']

            sketch = deserialize_sketch_from_string(sketch_str, lg_k)
            wrapped_sketch = ThetaSketchWrapper(sketch)

            if sketch_type == 'total':
                sketch_data['positive']['total'] = wrapped_sketch
            elif sketch_type == 'present':
                if feature not in sketch_data['positive']:
                    sketch_data['positive'][feature] = [None, None]
                sketch_data['positive'][feature][0] = wrapped_sketch
            elif sketch_type == 'absent':
                if feature not in sketch_data['positive']:
                    sketch_data['positive'][feature] = [None, None]
                sketch_data['positive'][feature][1] = wrapped_sketch

    # Convert lists to tuples
    for feature in list(sketch_data['positive'].keys()):
        if feature != 'total' and isinstance(sketch_data['positive'][feature], list):
            sketch_data['positive'][feature] = tuple(sketch_data['positive'][feature])

    # Load negative sketches
    with open(negative_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            feature = row['feature']
            sketch_type = row['sketch_type']
            sketch_str = row['sketch_data']

            sketch = deserialize_sketch_from_string(sketch_str, lg_k)
            wrapped_sketch = ThetaSketchWrapper(sketch)

            if sketch_type == 'total':
                sketch_data['negative']['total'] = wrapped_sketch
            elif sketch_type == 'present':
                if feature not in sketch_data['negative']:
                    sketch_data['negative'][feature] = [None, None]
                sketch_data['negative'][feature][0] = wrapped_sketch
            elif sketch_type == 'absent':
                if feature not in sketch_data['negative']:
                    sketch_data['negative'][feature] = [None, None]
                sketch_data['negative'][feature][1] = wrapped_sketch

    # Convert lists to tuples
    for feature in list(sketch_data['negative'].keys()):
        if feature != 'total' and isinstance(sketch_data['negative'][feature], list):
            sketch_data['negative'][feature] = tuple(sketch_data['negative'][feature])

    print(f"  Loaded positive sketches from: {positive_file}")
    print(f"  Loaded negative sketches from: {negative_file}")

    return sketch_data


def create_feature_mapping_from_sketches(sketch_data):
    """Create feature mapping from sketch data."""
    feature_names = []

    # Get feature names from positive class (excluding 'total')
    for feature_name in sketch_data['positive'].keys():
        if feature_name != 'total':
            feature_names.append(feature_name)

    # Create mapping
    feature_mapping = {name: idx for idx, name in enumerate(sorted(feature_names))}

    print(f"Created feature mapping for {len(feature_mapping)} features")
    return feature_mapping


def validate_sketch_data(sketch_data):
    """Validate that sketch data is properly formed."""
    print("Validating sketch data...")

    # Check basic structure
    assert 'positive' in sketch_data
    assert 'negative' in sketch_data
    assert 'total' in sketch_data['positive']
    assert 'total' in sketch_data['negative']

    # Check that features match between positive and negative
    pos_features = set(sketch_data['positive'].keys())
    neg_features = set(sketch_data['negative'].keys())

    assert pos_features == neg_features, f"Feature mismatch: {pos_features - neg_features}"

    # Check that each feature has proper tuple structure
    for class_name in ['positive', 'negative']:
        for feature_name, feature_data in sketch_data[class_name].items():
            if feature_name != 'total':
                assert isinstance(feature_data, tuple), f"{class_name}.{feature_name} is not a tuple"
                assert len(feature_data) == 2, f"{class_name}.{feature_name} tuple length != 2"
                assert feature_data[0] is not None, f"{class_name}.{feature_name} present sketch is None"
                assert feature_data[1] is not None, f"{class_name}.{feature_name} absent sketch is None"

    print("✅ Sketch data validation passed!")


def main(lg_k=DEFAULT_LG_K):
    """Main function to generate mushroom sketch files."""
    print("🍄 Generating Mushroom Theta Sketch Files")
    print("=" * 50)

    # Get project root for path resolution
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    try:
        # Load and prepare data
        df = load_mushroom_dataset()
        binary_features, feature_mapping = create_binary_features(df)

        # Create theta sketches with specified lg_k
        print(f"Using lg_k = {lg_k}")
        sketch_data = create_theta_sketches_with_ids(df, binary_features, lg_k)

        # Validate sketch data
        validate_sketch_data(sketch_data)

        # Save to CSV files with lg_k suffix (relative to project root)
        fixtures_dir = project_root / "tests/fixtures"
        positive_file = fixtures_dir / f"mushroom_positive_sketches_lg_k_{lg_k}.csv"
        negative_file = fixtures_dir / f"mushroom_negative_sketches_lg_k_{lg_k}.csv"

        # Create fixtures directory if it doesn't exist
        fixtures_dir.mkdir(exist_ok=True)

        save_sketches_to_csv(sketch_data, positive_file, negative_file)

        # Test loading the files
        print("\n🔄 Testing sketch file loading...")
        loaded_sketch_data = load_sketches_from_csv(positive_file, negative_file, lg_k)
        validate_sketch_data(loaded_sketch_data)

        # Create and save feature mapping (same for all lg_k values)
        feature_mapping = create_feature_mapping_from_sketches(loaded_sketch_data)

        # Save feature mapping to JSON
        import json
        mapping_file = fixtures_dir / "mushroom_feature_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(feature_mapping, f, indent=2)

        print(f"  Feature mapping saved to: {mapping_file}")

        print("\n✅ SUCCESS: Mushroom sketch files created and validated!")
        print(f"   - Positive sketches: {positive_file}")
        print(f"   - Negative sketches: {negative_file}")
        print(f"   - Feature mapping: {mapping_file}")
        print(f"   - Total features: {len(feature_mapping)}")
        print(f"   - lg_k parameter: {lg_k}")

        return sketch_data, feature_mapping

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate mushroom theta sketch files with configurable lg_k parameter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_mushroom_sketch_files.py --lg_k 11
  python create_mushroom_sketch_files.py --lg_k 16
  python create_mushroom_sketch_files.py --lg_k 8
        """
    )

    parser.add_argument(
        '--lg_k',
        type=int,
        default=DEFAULT_LG_K,
        help=f'lg_k parameter for theta sketches (default: {DEFAULT_LG_K})'
    )

    args = parser.parse_args()

    sketch_data, feature_mapping = main(lg_k=args.lg_k)

    # Print some sample statistics
    print(f"\n📊 Sample Statistics:")
    print(f"   Positive total estimate: {sketch_data['positive']['total'].get_estimate():.0f}")
    print(f"   Negative total estimate: {sketch_data['negative']['total'].get_estimate():.0f}")

    # Show first few features
    feature_names = list(feature_mapping.keys())[:5]
    print(f"   First 5 features: {feature_names}")
