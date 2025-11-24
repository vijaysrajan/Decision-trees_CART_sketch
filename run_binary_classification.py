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

    args = parser.parse_args()

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

        print(f"Training decision tree (criterion={args.criterion}, max_depth={args.max_depth}, tree_builder={args.tree_builder})...")
        clf = ThetaSketchDecisionTreeClassifier(
            criterion=args.criterion,
            max_depth=args.max_depth,
            tree_builder=args.tree_builder,
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

        return 0

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())