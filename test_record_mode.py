#!/usr/bin/env python3
"""
Quick record mode script for generating baseline JSON outputs.
This script trains a classifier and outputs the complete JSON structure
for manual inspection and baseline creation.
"""

import json
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.test_binary_classification_sketches import (
    load_mushroom_dataset,
    create_mushroom_sketches,
    create_mushroom_feature_mapping,
    tree_to_json
)
from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier


def serialize_classifier_output(clf, description):
    """Serialize classifier output to JSON format."""
    try:
        tree_json = tree_to_json(clf.tree_, max_depth=10)

        # Feature importances
        feature_importances = clf.feature_importances_.tolist() if hasattr(clf, 'feature_importances_') else []

        # Top features
        try:
            top_features = [(name, float(imp)) for name, imp in clf.get_top_features(10)]
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


def main():
    """Generate baseline outputs for all configurations."""
    print("Loading mushroom dataset...")
    df = load_mushroom_dataset()
    sketches = create_mushroom_sketches(df)
    feature_mapping = create_mushroom_feature_mapping(sketches)

    # Configuration mapping
    configurations = {
        "default_gini": {"criterion": "gini", "max_depth": 5},
        "entropy_shallow": {"criterion": "entropy", "max_depth": 3},
        "gain_ratio_medium": {"criterion": "gain_ratio", "max_depth": 7},
        "binomial_deep": {"criterion": "binomial", "max_depth": 10},
        "chi_square_default": {"criterion": "chi_square", "max_depth": 5}
    }

    baseline_outputs = {}

    for config_name, config in configurations.items():
        print(f"\nTraining {config_name} configuration...")

        # Train classifier
        clf = ThetaSketchDecisionTreeClassifier(verbose=1, **config)
        clf.fit(sketches, feature_mapping)

        # Serialize output
        output = serialize_classifier_output(clf, f"{config_name} configuration")
        baseline_outputs[config_name] = output

        # Print the JSON for this configuration
        print(f"\n{'='*80}")
        print(f"JSON OUTPUT FOR {config_name.upper()}:")
        print(f"{'='*80}")
        print(json.dumps(output, indent=2, default=str))
        print(f"{'='*80}")

    # Save complete baseline file
    with open("mushroom_baseline_outputs.json", 'w') as f:
        json.dump(baseline_outputs, f, indent=2, default=str)

    print(f"\n✅ Baseline outputs saved to mushroom_baseline_outputs.json")
    print(f"✅ Generated {len(baseline_outputs)} configurations")


if __name__ == "__main__":
    main()