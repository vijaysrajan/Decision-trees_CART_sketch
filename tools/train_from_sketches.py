#!/usr/bin/env python3
"""
Train decision tree directly from sketch CSV files.

Usage:
    python train_from_sketches.py positive.csv negative.csv config.yaml

Examples:
    # Train with mushroom sketches
    python train_from_sketches.py \
        tests/fixtures/mushroom_positive_sketches.csv \
        tests/fixtures/mushroom_negative_sketches_lg_k_12.csv \
        config.yaml

    # Train with custom DU sketches (once created)
    python train_from_sketches.py \
        du_sketches/positive.csv \
        du_sketches/negative.csv \
        du_config.yaml

Output:
    - Tree JSON printed to console
    - Model saved as .pkl file
    - Training details with verbose logging
"""

import sys
import json
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Go up one directory from tools/

try:
    from theta_sketch_tree import load_sketches, load_config, ThetaSketchDecisionTreeClassifier
    from tests.test_binary_classification_sketches import tree_to_json
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def validate_files(positive_csv: str, negative_csv: str, config_path: str) -> None:
    """Validate that all input files exist."""
    for file_path, file_type in [(positive_csv, "Positive CSV"),
                                 (negative_csv, "Negative CSV"),
                                 (config_path, "Config file")]:
        if not os.path.exists(file_path):
            print(f"Error: {file_type} not found: {file_path}")
            sys.exit(1)


def print_dataset_info(sketch_data: dict) -> None:
    """Print information about loaded sketch data."""
    print("\n" + "="*50)
    print("DATASET INFORMATION")
    print("="*50)

    positive_features = len(sketch_data.get('positive', {})) - 1  # Exclude 'total'
    negative_features = len(sketch_data.get('negative', {})) - 1  # Exclude 'total'

    print(f"Positive class features: {positive_features}")
    print(f"Negative class features: {negative_features}")

    if 'positive' in sketch_data and 'total' in sketch_data['positive']:
        pos_total_estimate = sketch_data['positive']['total'].get_estimate()
        print(f"Positive class total estimate: {pos_total_estimate:.0f}")

    if 'negative' in sketch_data and 'total' in sketch_data['negative']:
        neg_total_estimate = sketch_data['negative']['total'].get_estimate()
        print(f"Negative class total estimate: {neg_total_estimate:.0f}")


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    positive_csv = sys.argv[1]
    negative_csv = sys.argv[2]
    config_path = sys.argv[3]

    print("🌳 Theta Sketch Decision Tree Trainer")
    print("="*50)

    # Validate input files
    validate_files(positive_csv, negative_csv, config_path)

    try:
        # Load sketches and config
        print(f"📊 Loading sketches...")
        print(f"  Positive: {positive_csv}")
        print(f"  Negative: {negative_csv}")
        sketch_data = load_sketches(positive_csv=positive_csv, negative_csv=negative_csv)

        print(f"⚙️  Loading config from: {config_path}")
        config = load_config(config_path)

        # Print dataset information
        print_dataset_info(sketch_data)

        # Create and fit classifier
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        hyperparams = config['hyperparameters']
        print(f"Hyperparameters: {hyperparams}")

        clf = ThetaSketchDecisionTreeClassifier(**hyperparams)

        print("🚀 Fitting model...")
        clf.fit(sketch_data, config['feature_mapping'])

        print("✅ Training completed!")

        # Print model statistics
        print(f"📈 Tree depth: {clf.tree_.depth}")
        print(f"📊 Number of nodes: {clf.tree_.node_count}")
        print(f"🍃 Number of leaves: {clf.tree_.leaf_count}")

        # Output tree as JSON
        tree_json = tree_to_json(clf.tree_)
        print("\n" + "="*80)
        print("DECISION TREE JSON")
        print("="*80)
        print(json.dumps(tree_json, indent=2))

        # Generate model filename
        lg_k = config.get('lg_k', hyperparams.get('lg_k', 'unknown'))
        criterion = hyperparams.get('criterion', 'gini')
        max_depth = hyperparams.get('max_depth', 'unlimited')

        model_filename = f"tree_model_lg_k_{lg_k}_{criterion}_depth_{max_depth}.pkl"

        # Save model
        print(f"\n💾 Saving model to: {model_filename}")
        clf.save_model(model_filename)

        print("\n🎉 Training completed successfully!")
        print(f"📁 Model file: {model_filename}")
        print(f"📊 Tree JSON: displayed above")

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"❌ Missing required config key: {e}")
        print("Make sure your config.yaml has 'hyperparameters' and 'feature_mapping' sections")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()