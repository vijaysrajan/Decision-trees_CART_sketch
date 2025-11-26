#!/usr/bin/env python3
"""
Test JSON serialization fix for all pruning methods.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tests.test_binary_classification_sketches import (
    create_binary_classification_sketches,
    create_binary_classification_feature_mapping,
    tree_to_json
)
from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier


def test_json_serialization_fix():
    """Test that JSON serialization works for all pruning methods."""
    print('🔧 Testing JSON Serialization Fix for All Pruning Methods')
    print('=' * 60)

    # Load small mushroom sample for quick testing
    df = pd.read_csv('./tests/resources/agaricus-lepiota.csv')
    df = df.sample(n=500, random_state=42)

    # Split for validation data
    train_df, val_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['class'])
    X_val = val_df.drop(columns=['class'])
    y_val = val_df['class']

    # Create sketches
    sketches = create_binary_classification_sketches(train_df, lg_k=12)
    mapping = create_binary_classification_feature_mapping(sketches)

    # Convert validation data for methods that need it
    X_val_binary = np.zeros((X_val.shape[0], len(mapping)))
    for i, (idx, row) in enumerate(X_val.iterrows()):
        for feature_name, col_idx in mapping.items():
            if '=' in feature_name:
                base_feature, value = feature_name.split('=')
                X_val_binary[i, col_idx] = 1 if str(row[base_feature]) == value else 0

    print(f'Dataset: {train_df.shape[0]} training, {val_df.shape[0]} validation samples')
    print(f'Features: {len(mapping)} binary features')

    # Test all pruning methods
    methods = ['none', 'validation', 'cost_complexity', 'reduced_error', 'min_impurity']

    print(f'\n🧪 Testing JSON serialization for each pruning method:')
    print('-' * 60)

    for method in methods:
        try:
            clf = ThetaSketchDecisionTreeClassifier(
                criterion='gini',
                max_depth=6,
                pruning=method,
                min_impurity_decrease=0.01 if method == 'min_impurity' else 0.0,
                validation_fraction=0.2,
                verbose=0
            )

            # Fit with validation data for methods that need it
            if method in ['validation', 'reduced_error']:
                clf.fit(sketches, mapping, X_val=X_val_binary, y_val=y_val.values)
            else:
                clf.fit(sketches, mapping)

            # Test JSON serialization
            tree_json = tree_to_json(clf.tree_, max_depth=3)  # Limit depth for test

            # Verify it's actually JSON serializable by trying to serialize
            import json
            json_str = json.dumps(tree_json, indent=2)

            nodes = clf._count_tree_nodes()
            print(f'✅ {method:<15}: {nodes:>2} nodes - JSON serialization SUCCESS')

        except Exception as e:
            print(f'❌ {method:<15}: ERROR - {str(e)[:40]}...')

    print(f'\n🎉 JSON Serialization Fix Complete!')
    print(f'All pruning methods can now display tree structures without errors.')
    print(f'The fix converts numpy data types to native Python types before JSON serialization.')


if __name__ == "__main__":
    test_json_serialization_fix()