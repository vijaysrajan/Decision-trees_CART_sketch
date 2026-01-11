#!/usr/bin/env python3
"""
Command-line wrapper for running binary classification with any CSV dataset.

Usage:
    python run_binary_classification.py data.csv class_column [--lg_k 12] [--max_depth 5]

Example:
    python run_binary_classification.py spam_data.csv spam_label --lg_k 14 --max_depth 10
"""

import sys
import pandas as pd
import argparse
import numpy as np
from tests.test_binary_classification_sketches import (
    create_binary_classification_sketches,
    create_binary_classification_feature_mapping,
    tree_to_json,
    print_tree_json,
    DEFAULT_LG_K,
    DEFAULT_MAX_DEPTH,
    DEFAULT_CRITERION,
    DEFAULT_TREE_BUILDER,
    DEFAULT_VERBOSE
)
from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier


def main():
    parser = argparse.ArgumentParser(description='Train theta sketch decision tree on any binary classification CSV')
    parser.add_argument('csv_file', help='Path to CSV file')
    parser.add_argument('class_column', help='Name of the class/target column')
    parser.add_argument('--lg_k', type=int, default=DEFAULT_LG_K, help=f'Log-base-2 of sketch size (default: {DEFAULT_LG_K})')
    parser.add_argument('--max_depth', type=int, default=DEFAULT_MAX_DEPTH, help=f'Maximum tree depth (default: {DEFAULT_MAX_DEPTH})')
    parser.add_argument('--criterion', default=DEFAULT_CRITERION, choices=['gini', 'entropy', 'gain_ratio'], help=f'Split criterion (default: {DEFAULT_CRITERION})')
    parser.add_argument('--tree_builder', default=DEFAULT_TREE_BUILDER, choices=['intersection', 'ratio_based'], help=f'Tree builder mode (default: {DEFAULT_TREE_BUILDER})')
    parser.add_argument('--verbose', type=int, default=DEFAULT_VERBOSE, choices=[0, 1, 2], help=f'Verbosity level: 0=silent, 1=basic, 2=detailed (default: {DEFAULT_VERBOSE})')
    parser.add_argument('--sample_size', type=int, help='Subsample size for large datasets')
    parser.add_argument('--save_model', type=str, help='Path to save the trained model (.pkl extension will be added)')
    parser.add_argument('--load_model', type=str, help='Path to load a pre-trained model for prediction only')
    parser.add_argument('--model_info', type=str, help='Path to model file to display information about')

    # Pruning parameters
    parser.add_argument('--pruning', default='none', choices=['none', 'validation', 'cost_complexity', 'reduced_error', 'min_impurity'], help='Pruning method (default: none)')
    parser.add_argument('--min_impurity_decrease', type=float, default=0.0, help='Minimum impurity decrease for pruning (default: 0.0)')
    parser.add_argument('--validation_fraction', type=float, default=0.2, help='Fraction of data for validation-based pruning (default: 0.2)')

    args = parser.parse_args()

    # Handle model info request
    if args.model_info:
        try:
            clf = ThetaSketchDecisionTreeClassifier.load_model(args.model_info)
            info = clf.get_model_info()

            print(f"📋 Model Information for: {args.model_info}")
            print("=" * 60)
            print(f"Hyperparameters:")
            for key, value in info['hyperparameters'].items():
                print(f"  {key}: {value}")
            print(f"\nTree Statistics:")
            print(f"  Features: {info['n_features']}")
            print(f"  Classes: {info['n_classes']}")
            print(f"  Tree depth: {info['tree_depth']}")
            print(f"  Tree nodes: {info['tree_nodes']}")
            print(f"  Tree leaves: {info['tree_leaves']}")
            print(f"  Has sketch data: {info['has_sketch_data']}")
            print(f"\nFeatures: {info['feature_names'][:10]}{'...' if len(info['feature_names']) > 10 else ''}")

            return 0
        except Exception as e:
            print(f"❌ Error loading model info: {e}")
            return 1

    # Handle model loading for prediction only
    if args.load_model:
        try:
            clf = ThetaSketchDecisionTreeClassifier.load_model(args.load_model)
            print("🔮 Model loaded successfully! You can now use it for predictions.")

            # Generate sample predictions if dataset provided
            if args.csv_file and args.class_column:
                print(f"\n📊 Generating sample predictions on: {args.csv_file}")
                df = pd.read_csv(args.csv_file)
                if args.class_column in df.columns:
                    df = df.drop(columns=[args.class_column])  # Remove target for prediction

                # Take sample for prediction demo
                sample_size = min(10, len(df))
                X_sample = df.sample(n=sample_size, random_state=42).values

                if X_sample.shape[1] == clf.n_features_in_:
                    predictions = clf.predict(X_sample)
                    probabilities = clf.predict_proba(X_sample)

                    print(f"Sample predictions:")
                    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                        print(f"  Sample {i+1}: Class {pred} (confidence: {max(prob):.3f})")
                else:
                    print(f"⚠️  Feature mismatch: dataset has {X_sample.shape[1]} features, model expects {clf.n_features_in_}")

            return 0
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return 1

    print(f"Loading dataset: {args.csv_file}")
    try:
        df = pd.read_csv(args.csv_file)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return 1

    # Check class column exists
    if args.class_column not in df.columns:
        print(f"Error: Column '{args.class_column}' not found in dataset")
        print(f"Available columns: {list(df.columns)}")
        return 1

    # Rename class column to 'class' for compatibility
    if args.class_column != 'class':
        df = df.rename(columns={args.class_column: 'class'})
        print(f"Renamed '{args.class_column}' to 'class'")

    # Check for binary classification
    class_values = df['class'].unique()
    if len(class_values) != 2:
        print(f"Error: Expected exactly 2 class values, found {len(class_values)}: {class_values}")
        return 1

    print(f"Class distribution: {df['class'].value_counts().to_dict()}")

    # Subsample if requested
    if args.sample_size and len(df) > args.sample_size:
        df = df.sample(n=args.sample_size, random_state=42)
        print(f"Subsampled to {args.sample_size} rows")

    try:
        print(f"\nCreating theta sketches with lg_k={args.lg_k}...")
        sketches = create_binary_classification_sketches(df, lg_k=args.lg_k)

        print("Creating feature mapping...")
        mapping = create_binary_classification_feature_mapping(sketches)

        print(f"Training decision tree (criterion={args.criterion}, max_depth={args.max_depth}, tree_builder={args.tree_builder}, pruning={args.pruning})...")
        clf = ThetaSketchDecisionTreeClassifier(
            criterion=args.criterion,
            max_depth=args.max_depth,
            tree_builder=args.tree_builder,
            pruning=args.pruning,
            min_impurity_decrease=args.min_impurity_decrease,
            validation_fraction=args.validation_fraction,
            verbose=args.verbose
        )
        clf.fit(sketches, mapping)

        print("\n🎉 Training completed successfully!")

        # Print tree structure in JSON format
        if hasattr(clf.tree_, 'is_leaf'):
            tree_json = print_tree_json(clf.tree_, max_depth=args.max_depth)
            print(f"\nTree depth: {tree_json.get('depth', 'N/A')}")
        else:
            print(f"Tree depth: {clf.tree_.depth if hasattr(clf.tree_, 'depth') else 'N/A'}")

        print(f"Number of features: {len(mapping)}")

        print("\nTop 10 important features:")
        for feature, importance in clf.get_top_features(10):
            print(f"  {feature}: {importance:.4f}")

        # Save model if requested
        if args.save_model:
            print(f"\n💾 Saving model to: {args.save_model}")
            try:
                # Ask user if they want to include sketch data (can be large)
                include_sketches = False
                if args.verbose > 0:
                    print("Note: Sketch data can be large. Including it allows model retraining.")
                    # For automated scripts, don't include sketches by default

                clf.save_model(args.save_model, include_sketches=include_sketches)

                # Show model info
                info = clf.get_model_info()
                print(f"✅ Model saved with {info['tree_nodes']} nodes and depth {info['tree_depth']}")

            except Exception as e:
                print(f"❌ Error saving model: {e}")
                return 1

        return 0

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())