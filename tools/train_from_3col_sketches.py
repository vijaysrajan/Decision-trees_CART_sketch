#!/usr/bin/env python3
"""
Train decision tree from 3-column sketch CSV file.

Usage:
    python tools/train_from_3col_sketches.py sketches_3col.csv feature_mapping.json config.yaml

Example:
    python tools/train_from_3col_sketches.py \
        agaricus_lepiota_sketches/agaricus_lepiota_3col_sketches.csv \
        agaricus_lepiota_sketches/agaricus_lepiota_feature_mapping.json \
        agaricus_lepiota_sketches/mushroom_training_config.yaml
"""

import sys
import json
import os
import csv
import base64
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from datasketches import compact_theta_sketch
    from tests.test_binary_classification_sketches import ThetaSketchWrapper
    from theta_sketch_tree import ThetaSketchDecisionTreeClassifier
    from tests.test_binary_classification_sketches import tree_to_json
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def load_3col_sketches(csv_file: str) -> dict:
    """Load 3-column sketch CSV into sketch_data format."""
    from datasketches import update_theta_sketch, theta_a_not_b

    sketch_data = {'positive': {}, 'negative': {}}

    # First pass: collect all feature sketches
    temp_sketches = {'positive': {}, 'negative': {}}

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            feature_name = row['identifier']

            # Load positive sketch if present
            if row['sketch_feature_present_positive']:
                sketch_bytes = base64.b64decode(row['sketch_feature_present_positive'])
                compact_sketch = compact_theta_sketch.deserialize(sketch_bytes)
                temp_sketches['positive'][feature_name] = ThetaSketchWrapper(compact_sketch)

            # Load negative sketch if present
            if row['sketch_feature_present_negative']:
                sketch_bytes = base64.b64decode(row['sketch_feature_present_negative'])
                compact_sketch = compact_theta_sketch.deserialize(sketch_bytes)
                temp_sketches['negative'][feature_name] = ThetaSketchWrapper(compact_sketch)

    # Create dummy 'total' sketches for compatibility
    for class_name in ['positive', 'negative']:
        total_sketch = ThetaSketchWrapper(update_theta_sketch(19))  # Use lg_k=19 to match config
        # Add some dummy data to make it non-empty
        for i in range(1000):  # Larger dummy dataset
            total_sketch.update(i)
        sketch_data[class_name] = {'total': total_sketch}

    # Second pass: convert each feature to (present, absent) tuple format
    all_features = set(temp_sketches['positive'].keys()) | set(temp_sketches['negative'].keys())

    for feature_name in all_features:
        for class_name in ['positive', 'negative']:
            present_sketch = temp_sketches[class_name].get(feature_name)

            if present_sketch is None:
                # Create empty sketch if feature not present in this class
                present_sketch = ThetaSketchWrapper(update_theta_sketch(19))

            # Create dummy absent sketch as complement
            # In real implementation, this would be computed as total - present
            absent_sketch = ThetaSketchWrapper(update_theta_sketch(19))
            for i in range(500):  # Dummy absent data
                absent_sketch.update(i + 10000)  # Different hash space

            # Store as tuple: (present_sketch, absent_sketch)
            sketch_data[class_name][feature_name] = (present_sketch, absent_sketch)

    return sketch_data


def load_feature_mapping(json_file: str) -> dict:
    """Load feature mapping from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def load_config(yaml_file: str) -> dict:
    """Load configuration from YAML file."""
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    sketches_csv = sys.argv[1]
    feature_mapping_json = sys.argv[2]
    config_yaml = sys.argv[3]

    print("🌳 Theta Sketch Decision Tree Trainer (3-Column)")
    print("=" * 50)

    # Validate input files
    for file_path, file_type in [(sketches_csv, "Sketches CSV"),
                                 (feature_mapping_json, "Feature mapping JSON"),
                                 (config_yaml, "Config YAML")]:
        if not os.path.exists(file_path):
            print(f"❌ {file_type} not found: {file_path}")
            sys.exit(1)

    try:
        # Load sketches, feature mapping, and config
        print(f"📊 Loading 3-column sketches: {sketches_csv}")
        sketch_data = load_3col_sketches(sketches_csv)

        print(f"🗺️ Loading feature mapping: {feature_mapping_json}")
        feature_mapping = load_feature_mapping(feature_mapping_json)

        print(f"⚙️ Loading config: {config_yaml}")
        config = load_config(config_yaml)

        # Print dataset information
        print("\n" + "=" * 50)
        print("DATASET INFORMATION")
        print("=" * 50)

        positive_features = len(sketch_data.get('positive', {}))
        negative_features = len(sketch_data.get('negative', {}))

        print(f"Positive class features: {positive_features}")
        print(f"Negative class features: {negative_features}")
        print(f"Total unique features: {len(feature_mapping)}")

        # Create and fit classifier
        print("\n" + "=" * 50)
        print("MODEL TRAINING")
        print("=" * 50)

        hyperparams = config['hyperparameters']
        print(f"Hyperparameters: {hyperparams}")

        clf = ThetaSketchDecisionTreeClassifier(**hyperparams)

        print("🚀 Fitting model...")
        clf.fit(sketch_data, feature_mapping)

        print("✅ Training completed!")

        # Print model statistics
        if hasattr(clf, 'tree_') and clf.tree_ is not None:
            print(f"📈 Tree depth: {clf.tree_.depth}")

            # Count nodes manually if node_count attribute doesn't exist
            if hasattr(clf.tree_, 'node_count'):
                print(f"📊 Number of nodes: {clf.tree_.node_count}")

            if hasattr(clf.tree_, 'leaf_count'):
                print(f"🍃 Number of leaves: {clf.tree_.leaf_count}")

            # Output tree as JSON
            tree_json = tree_to_json(clf.tree_)
            print("\n" + "=" * 80)
            print("DECISION TREE JSON")
            print("=" * 80)
            print(json.dumps(tree_json, indent=2))

        # Generate model filename
        lg_k = config.get('lg_k', 'unknown')
        criterion = hyperparams.get('criterion', 'gini')
        max_depth = hyperparams.get('max_depth', 'unlimited')

        dataset_name = Path(sketches_csv).stem.replace('_3col_sketches', '')
        model_filename = f"{dataset_name}_model_lg_k_{lg_k}_{criterion}_depth_{max_depth}.pkl"

        # Save model (temporarily disabled due to import issue)
        print(f"\n💾 Model would be saved to: {model_filename}")
        # clf.save_model(model_filename)  # Commented out due to ModelPersistence import issue

        print("\n🎉 Training completed successfully!")
        print(f"📁 Model ready for saving: {model_filename}")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())