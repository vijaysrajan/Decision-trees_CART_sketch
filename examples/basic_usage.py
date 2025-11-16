#!/usr/bin/env python3
"""
Basic usage example for Theta Sketch Decision Tree Classifier.

This example demonstrates the complete workflow from sketch data
to trained model to predictions and feature importance analysis.
"""

import sys
import os

# Add parent directory to path to find theta_sketch_tree module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier
from tests.test_mock_sketches import create_mock_sketch_data, create_feature_mapping


def main():
    """Run basic usage example."""
    print("🌳 Theta Sketch Decision Tree - Basic Usage Example")
    print("=" * 50)

    # Step 1: Load sketch data (using mock data for this example)
    print("📊 Loading sketch data...")
    sketch_data = create_mock_sketch_data()
    feature_mapping = create_feature_mapping()

    print(f"   Features: {list(feature_mapping.keys())}")
    print(f"   Positive class total: {sketch_data['positive']['total'].get_estimate():.0f}")
    print(f"   Negative class total: {sketch_data['negative']['total'].get_estimate():.0f}")

    # Step 2: Create and train classifier
    print("\n🚀 Training classifier...")
    clf = ThetaSketchDecisionTreeClassifier(
        criterion='gini',
        max_depth=10,
        min_samples_split=2,
        verbose=1
    )

    clf.fit(sketch_data, feature_mapping)
    print("   ✅ Training complete!")

    # Step 3: Make predictions
    print("\n🔮 Making predictions...")

    # Create test data (binary features matching feature_mapping)
    X_test = np.array([
        [1, 0],  # age>30=True, income>50k=False
        [0, 1],  # age>30=False, income>50k=True
        [1, 1],  # age>30=True, income>50k=True
        [0, 0],  # age>30=False, income>50k=False
    ])

    # Get predictions
    predictions = clf.predict(X_test)
    probabilities = clf.predict_proba(X_test)

    print("   Test cases:")
    feature_names = list(feature_mapping.keys())
    for i, (features, pred, prob) in enumerate(zip(X_test, predictions, probabilities)):
        feature_desc = ", ".join([
            f"{name}={'Yes' if val else 'No'}"
            for name, val in zip(feature_names, features)
        ])
        confidence = prob[pred]
        result = "Positive" if pred == 1 else "Negative"
        print(f"   {i+1}. [{feature_desc}] → {result} (confidence: {confidence:.3f})")

    # Step 4: Analyze feature importance
    print("\n📈 Feature Importance Analysis:")

    # Get feature importances
    importances = clf.feature_importances_
    importance_dict = clf.get_feature_importance_dict()
    top_features = clf.get_top_features(top_k=5)

    print("   Feature importance scores:")
    for feature, importance in importance_dict.items():
        print(f"   • {feature}: {importance:.3f}")

    print(f"\n   Top features: {top_features}")

    # Step 5: Model performance summary
    print("\n📊 Model Summary:")
    print(f"   • Features trained: {clf.n_features_in_}")
    print(f"   • Classes: {clf.n_classes_}")
    print(f"   • Criterion: {clf.criterion}")
    print(f"   • Max depth: {clf.max_depth}")

    # Tree structure info (if accessible)
    if hasattr(clf.tree_, 'depth') and hasattr(clf.tree_, 'n_samples'):
        print(f"   • Tree depth: {clf.tree_.depth}")
        print(f"   • Root samples: {clf.tree_.n_samples}")

    print("\n✅ Example complete! See docs/user_guide.md for more advanced usage.")


if __name__ == "__main__":
    main()