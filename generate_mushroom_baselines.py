#!/usr/bin/env python3
"""
Generate baseline reference outputs for mushroom dataset with various hyperparameters.

This script creates serialized outputs that can be used to detect regression during
code cleanup and refactoring.
"""

import json
import pandas as pd
import numpy as np
from tests.test_binary_classification_sketches import (
    load_mushroom_dataset,
    create_mushroom_sketches,
    create_mushroom_feature_mapping,
    tree_to_json
)
from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier


def serialize_tree_output(clf, description):
    """Serialize classifier output for regression testing."""
    if not hasattr(clf, 'tree_') or clf.tree_ is None:
        return {"error": "No tree found"}

    try:
        # Basic tree info
        tree_json = tree_to_json(clf.tree_, max_depth=10)

        # Feature importances
        try:
            feature_importances = clf.feature_importances_.tolist() if hasattr(clf, 'feature_importances_') else []
        except:
            feature_importances = []

        # Top features
        try:
            top_features = [(name, imp) for name, imp in clf.get_top_features(10)]
        except:
            top_features = []

        return {
            "description": description,
            "tree_structure": tree_json,
            "tree_depth": tree_json.get('depth', 0) if tree_json else 0,
            "n_samples": tree_json.get('n_samples', 0) if tree_json else 0,
            "feature_importances": feature_importances,
            "top_features": top_features,
            "n_features": clf.n_features_in_ if hasattr(clf, 'n_features_in_') else 0,
            "classes": clf.classes_.tolist() if hasattr(clf, 'classes_') else []
        }
    except Exception as e:
        return {"error": f"Serialization failed: {str(e)}"}


def generate_mushroom_baselines():
    """Generate baseline outputs for various hyperparameter combinations."""
    print("🍄 Generating Mushroom Dataset Baselines")
    print("=" * 50)

    # Load data
    print("Loading mushroom dataset...")
    df = load_mushroom_dataset()
    sketches = create_mushroom_sketches(df)
    feature_mapping = create_mushroom_feature_mapping(sketches)

    print(f"Dataset: {df.shape[0]} samples, {len(feature_mapping)} binary features")

    # Define hyperparameter combinations to test
    test_configurations = [
        # Basic configurations
        {
            "name": "default_gini",
            "criterion": "gini",
            "max_depth": 5,
            "description": "Default Gini criterion, depth 5"
        },
        {
            "name": "entropy_shallow",
            "criterion": "entropy",
            "max_depth": 3,
            "description": "Entropy criterion, shallow tree (depth 3)"
        },
        {
            "name": "gain_ratio_medium",
            "criterion": "gain_ratio",
            "max_depth": 7,
            "description": "Gain ratio criterion, medium depth"
        },
        {
            "name": "binomial_deep",
            "criterion": "binomial",
            "max_depth": 10,
            "description": "Binomial criterion, deeper tree"
        },
        {
            "name": "chi_square_default",
            "criterion": "chi_square",
            "max_depth": 5,
            "description": "Chi-square criterion, default depth"
        },

        # Pruning combinations
        {
            "name": "gini_cost_complexity",
            "criterion": "gini",
            "max_depth": 8,
            "pruning": "cost_complexity",
            "description": "Gini with cost-complexity pruning"
        },
        {
            "name": "entropy_validation_pruning",
            "criterion": "entropy",
            "max_depth": 10,
            "pruning": "validation",
            "description": "Entropy with validation pruning"
        },

        # Edge cases
        {
            "name": "very_shallow",
            "criterion": "gini",
            "max_depth": 1,
            "description": "Very shallow tree (depth 1)"
        },
        {
            "name": "min_samples_high",
            "criterion": "gini",
            "max_depth": 5,
            "min_samples_split": 100,
            "min_samples_leaf": 50,
            "description": "High minimum samples requirements"
        }
    ]

    baselines = {}

    # Generate baselines for each configuration
    for i, config in enumerate(test_configurations):
        config_name = config.pop("name")
        description = config.pop("description")

        print(f"\n[{i+1}/{len(test_configurations)}] Testing: {config_name}")
        print(f"  Description: {description}")
        print(f"  Config: {config}")

        try:
            # Create classifier with configuration
            clf = ThetaSketchDecisionTreeClassifier(
                verbose=0,  # Suppress output
                **config
            )

            # Fit the model
            clf.fit(sketches, feature_mapping)

            # Generate baseline
            baseline = serialize_tree_output(clf, description)
            baselines[config_name] = baseline

            print(f"  ✅ Success: Tree depth {baseline.get('tree_depth', 'N/A')}, "
                  f"samples {baseline.get('n_samples', 'N/A')}")

        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
            baselines[config_name] = {
                "description": description,
                "error": str(e)
            }

    # Save baselines to file
    baseline_file = "mushroom_baseline_outputs.json"
    with open(baseline_file, 'w') as f:
        json.dump(baselines, f, indent=2, default=str)

    print(f"\n💾 Baselines saved to: {baseline_file}")

    # Print summary
    successful = sum(1 for b in baselines.values() if "error" not in b)
    total = len(baselines)

    print(f"\n📊 Summary:")
    print(f"  Total configurations: {total}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {total - successful}")

    # Show top features for a few key configurations
    print(f"\n🔝 Sample Top Features:")
    for config_name in ["default_gini", "entropy_shallow", "gain_ratio_medium"]:
        if config_name in baselines and "top_features" in baselines[config_name]:
            top_features = baselines[config_name]["top_features"][:3]
            print(f"  {config_name}: {top_features}")

    return baselines


if __name__ == "__main__":
    baselines = generate_mushroom_baselines()
    print("\n🎉 Baseline generation complete!")