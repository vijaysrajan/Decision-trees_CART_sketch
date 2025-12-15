#!/usr/bin/env python3
"""
Train decision tree from 3-column sketch CSV files.

This script trains a decision tree classifier using positive and negative
3-column CSV files and saves the model in both PKL and JSON formats.

Usage:
    python tools/train_from_3col_sketches.py \
        --lg_k 16 \
        --positive positive_3col_sketches.csv \
        --negative negative_3col_sketches.csv \
        --config config.yaml \
        --output output_dir/

Example:
    python tools/train_from_3col_sketches.py \
        --lg_k 16 \
        --positive DU_output/positive_3col_sketches.csv \
        --negative DU_output/negative_3col_sketches.csv \
        --config configs/du_config.yaml \
        --output DU_output/
"""

import argparse
import sys
import os
import json
import pickle
import csv
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from theta_sketch_tree.classifier_utils import ClassifierUtils
    from tests.test_binary_classification_sketches import tree_to_json
except ImportError as e:
    raise ImportError(f"Required modules not found: {e}")


def validate_input_files(positive_csv: str, negative_csv: str, config_yaml: str):
    """Validate that all input files exist."""
    files_to_check = [
        (positive_csv, "Positive CSV"),
        (negative_csv, "Negative CSV"),
        (config_yaml, "Config YAML")
    ]

    for file_path, file_type in files_to_check:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_type} not found: {file_path}")

    print(f"✅ All input files validated")


def save_model_outputs(classifier, output_dir: str, lg_k: int, positive_file: str):
    """Save the trained model in PKL and JSON formats."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate base filename from positive file
    base_name = Path(positive_file).stem.replace('positive_', '').replace('_3col_sketches', '')
    model_base = f"{base_name}_model_lg_k_{lg_k}"

    # Save PKL format
    pkl_path = output_path / f"{model_base}.pkl"
    print(f"💾 Saving model to: {pkl_path}")

    try:
        with open(pkl_path, 'wb') as f:
            pickle.dump(classifier, f)
        print(f"✅ Saved PKL model: {pkl_path}")
    except Exception as e:
        print(f"❌ Error saving PKL model: {e}")

    # Save JSON format
    json_path = output_path / f"{model_base}.json"
    print(f"💾 Saving tree JSON to: {json_path}")

    try:
        if hasattr(classifier, 'tree_') and classifier.tree_ is not None:
            tree_json = tree_to_json(classifier.tree_)

            # Add model metadata
            model_data = {
                "model_type": "ThetaSketchDecisionTree",
                "lg_k": lg_k,
                "hyperparameters": {
                    "criterion": getattr(classifier, 'criterion', 'unknown'),
                    "max_depth": getattr(classifier, 'max_depth', 'unknown'),
                    "min_samples_split": getattr(classifier, 'min_samples_split', 'unknown'),
                    "min_samples_leaf": getattr(classifier, 'min_samples_leaf', 'unknown')
                },
                "tree_structure": tree_json
            }

            with open(json_path, 'w') as f:
                json.dump(model_data, f, indent=2, default=str)
            print(f"✅ Saved JSON tree: {json_path}")
        else:
            print(f"⚠️  No tree found in classifier, skipping JSON save")
    except Exception as e:
        print(f"❌ Error saving JSON tree: {e}")

    return pkl_path, json_path


def print_model_stats(classifier):
    """Print model statistics and tree structure."""
    print("\n" + "=" * 50)
    print("MODEL STATISTICS")
    print("=" * 50)

    if hasattr(classifier, 'tree_') and classifier.tree_ is not None:
        tree = classifier.tree_
        print(f"🌳 Tree depth: {tree.depth}")

        if hasattr(tree, 'node_count'):
            print(f"📊 Number of nodes: {tree.node_count}")
        if hasattr(tree, 'leaf_count'):
            print(f"🍃 Number of leaves: {tree.leaf_count}")

        # Print feature importances if available
        if hasattr(classifier, 'feature_importances_') and classifier.feature_importances_ is not None:
            print(f"\n📈 Feature importances computed: {len(classifier.feature_importances_)} features")

        # Print tree JSON
        try:
            tree_json = tree_to_json(tree)
            print("\n" + "=" * 80)
            print("DECISION TREE JSON")
            print("=" * 80)
            print(json.dumps(tree_json, indent=2, default=str))
        except Exception as e:
            print(f"⚠️  Could not generate tree JSON: {e}")
    else:
        print("❌ No tree found in classifier")


def main():
    parser = argparse.ArgumentParser(
        description='Train decision tree from 3-column sketch CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python tools/train_from_3col_sketches.py \\
      --lg_k 16 \\
      --positive positive_3col_sketches.csv \\
      --negative negative_3col_sketches.csv \\
      --config config.yaml \\
      --output output_dir/

  # With DU dataset
  python tools/train_from_3col_sketches.py \\
      --lg_k 16 \\
      --positive DU_output/positive_3col_sketches.csv \\
      --negative DU_output/negative_3col_sketches.csv \\
      --config configs/du_config.yaml \\
      --output DU_output/
        """)

    parser.add_argument('--lg_k', type=int, required=True,
                       help='Theta sketch lg_k parameter used for generation')
    parser.add_argument('--positive', type=str, required=True,
                       help='Positive class 3-column CSV file')
    parser.add_argument('--negative', type=str, required=True,
                       help='Negative class 3-column CSV file')
    parser.add_argument('--config', type=str, required=True,
                       help='Configuration YAML file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for model files')

    args = parser.parse_args()

    print("🌳 Theta Sketch Decision Tree Trainer")
    print("=" * 50)

    print(f"📋 Configuration:")
    print(f"   lg_k: {args.lg_k}")
    print(f"   Positive CSV: {args.positive}")
    print(f"   Negative CSV: {args.negative}")
    print(f"   Config YAML: {args.config}")
    print(f"   Output directory: {args.output}")

    try:
        # Validate input files
        validate_input_files(args.positive, args.negative, args.config)

        # Increase CSV field size limit for large theta sketches
        csv.field_size_limit(5000000)  # 5MB limit

        # Train model using ClassifierUtils
        print("\n🚀 Training decision tree using ClassifierUtils.fit_from_csv...")
        classifier = ClassifierUtils.fit_from_csv(args.positive, args.negative, args.config)

        print("✅ Training completed successfully!")

        # Print model statistics
        print_model_stats(classifier)

        # Save model outputs
        pkl_path, json_path = save_model_outputs(classifier, args.output, args.lg_k, args.positive)

        print(f"\n🎉 Training complete!")
        print(f"📄 PKL model: {pkl_path}")
        print(f"📄 JSON tree: {json_path}")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())