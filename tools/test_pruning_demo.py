#!/usr/bin/env python3
"""
Demonstration of Advanced Pruning Methods for Theta Sketch Decision Trees.

This script demonstrates all available pruning methods and their effects
on tree complexity and model generalization.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tests.test_binary_classification_sketches import (
    create_binary_classification_sketches,
    create_binary_classification_feature_mapping
)
from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier
from theta_sketch_tree.tree_builder import TreeBuilder


def demonstrate_pruning_methods():
    """Demonstrate all pruning methods on a real dataset."""
    print("🌳 Advanced Pruning Methods Demonstration")
    print("=" * 60)

    # Load and prepare dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_path = os.path.join(project_root, 'tests', 'resources', 'binary_classification_data.csv')
    df = pd.read_csv(dataset_path)
    df = df.rename(columns={'target': 'class'})

    # Split for validation data
    X = df.drop(columns=['class'])
    y = df['class']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    train_df = pd.concat([X_train, y_train], axis=1)

    print(f"📊 Dataset: {df.shape}")
    print(f"   Training: {train_df.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples")
    print(f"   Classes: {y.value_counts().to_dict()}")

    # Create sketches
    sketches = create_binary_classification_sketches(train_df, lg_k=12)
    mapping = create_binary_classification_feature_mapping(sketches)

    # Prepare validation data for pruning methods that need it
    X_val_binary = np.zeros((X_val.shape[0], len(mapping)))
    for i, row in X_val.iterrows():
        for feature_name, col_idx in mapping.items():
            if '>=' in feature_name:
                base_feature, threshold = feature_name.split('>=')
                X_val_binary[X_val.index.get_loc(i), col_idx] = 1 if row[base_feature] >= float(threshold) else 0
            elif '>' in feature_name:
                base_feature, threshold = feature_name.split('>')
                X_val_binary[X_val.index.get_loc(i), col_idx] = 1 if row[base_feature] > float(threshold) else 0

    # Test different pruning methods
    methods_config = {
        'none': {'description': 'No pruning (baseline)'},
        'validation': {'description': 'Validation-based pruning'},
        'cost_complexity': {'description': 'Cost-complexity pruning (minimal error)'},
        'reduced_error': {'description': 'Reduced error pruning'},
        'min_impurity': {'description': 'Minimum impurity decrease pruning'}
    }

    results = []

    for method, config in methods_config.items():
        print(f"\n🔍 Testing: {config['description']}")
        print("-" * 40)

        try:
            clf = ThetaSketchDecisionTreeClassifier(
                criterion='gini',
                max_depth=8,
                pruning=method,
                min_impurity_decrease=0.02 if method == 'min_impurity' else 0.0,
                validation_fraction=0.3,
                verbose=0
            )

            # Fit with validation data for methods that need it
            if method in ['validation', 'reduced_error']:
                clf.fit(sketches, mapping, X_val=X_val_binary, y_val=y_val.values)
            else:
                clf.fit(sketches, mapping)

            # Collect results
            nodes = TreeBuilder.count_tree_nodes(clf.tree_)
            leaves = TreeBuilder.count_tree_leaves(clf.tree_)
            depth = clf.tree_.depth

            # Convert training data to binary format for predictions
            X_train_binary = np.zeros((X_train.shape[0], len(mapping)))
            for i, row in X_train.iterrows():
                for feature_name, col_idx in mapping.items():
                    if '>=' in feature_name:
                        base_feature, threshold = feature_name.split('>=')
                        X_train_binary[X_train.index.get_loc(i), col_idx] = 1 if row[base_feature] >= float(threshold) else 0
                    elif '>' in feature_name:
                        base_feature, threshold = feature_name.split('>')
                        X_train_binary[X_train.index.get_loc(i), col_idx] = 1 if row[base_feature] > float(threshold) else 0

            # Make predictions to test accuracy
            train_pred = clf.predict(X_train_binary)
            train_acc = np.mean(train_pred == y_train.values)

            val_pred = clf.predict(X_val_binary)
            val_acc = np.mean(val_pred == y_val.values)

            result = {
                'method': method,
                'nodes': nodes,
                'leaves': leaves,
                'depth': depth,
                'train_acc': train_acc,
                'val_acc': val_acc
            }
            results.append(result)

            print(f"   Nodes: {nodes:2d} | Leaves: {leaves:2d} | Depth: {depth}")
            print(f"   Training accuracy: {train_acc:.3f}")
            print(f"   Validation accuracy: {val_acc:.3f}")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'method': method,
                'error': str(e)
            })

    # Summary comparison
    print(f"\n📈 Pruning Results Summary")
    print("=" * 80)
    print(f"{'Method':<18} | {'Nodes':<5} | {'Leaves':<6} | {'Depth':<5} | {'Train Acc':<9} | {'Val Acc':<7}")
    print("-" * 80)

    baseline_nodes = None
    for result in results:
        if 'error' in result:
            print(f"{result['method']:<18} | ERROR: {result['error']}")
            continue

        if result['method'] == 'none':
            baseline_nodes = result['nodes']

        reduction = ""
        if baseline_nodes and result['nodes'] < baseline_nodes:
            reduction = f"(-{baseline_nodes - result['nodes']:2d})"

        print(f"{result['method']:<18} | {result['nodes']:>5} | {result['leaves']:>6} | "
              f"{result['depth']:>5} | {result['train_acc']:>9.3f} | {result['val_acc']:>7.3f} {reduction}")

    print("\n✅ Pruning demonstration completed!")
    print("\nKey Insights:")
    print("- Validation and reduced error pruning prevent overfitting")
    print("- Cost-complexity finds optimal trade-off between complexity and accuracy")
    print("- Minimum impurity pruning removes low-benefit splits")
    print("- Pruned models often generalize better to new data")


if __name__ == "__main__":
    demonstrate_pruning_methods()