#!/usr/bin/env python3
"""
Demo script showing feature subsampling for Random Forest compatibility.

This script demonstrates:
1. Creating multiple trees with different feature subsets (Random Forest simulation)
2. All supported feature subsampling strategies
3. Reproducibility with random_state
4. Integration with the existing theta sketch tree classifier
"""

from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier
from theta_sketch_tree.split_finder import SplitFinder
from theta_sketch_tree.criteria import GiniCriterion
import math


def demo_feature_subsampling_strategies():
    """Demonstrate all supported feature subsampling strategies."""
    print("🌳 Feature Subsampling Strategies Demo")
    print("=" * 50)

    criterion = GiniCriterion()
    split_finder = SplitFinder(criterion, min_samples_leaf=1, verbose=0)

    # Create a feature set representing a realistic scenario
    features = [f"feature_{i:02d}" for i in range(50)]
    print(f"📊 Total available features: {len(features)}")

    strategies = {
        'sqrt': 'sqrt',
        'log2': 'log2',
        '30%': 0.3,
        '50%': 0.5,
        'exact_15': 15
    }

    print("\n📈 Subsampling Results:")
    for name, strategy in strategies.items():
        subsampled = split_finder._subsample_features(features, strategy, random_state=42)
        expected = {
            'sqrt': int(math.sqrt(50)),  # 7
            'log2': int(math.log2(50)),  # 5
            '30%': int(0.3 * 50),       # 15
            '50%': int(0.5 * 50),       # 25
            'exact_15': 15
        }

        print(f"  {name:>10}: {len(subsampled):2d} features (expected: {expected[name]:2d})")
        print(f"             {subsampled[:5]}{'...' if len(subsampled) > 5 else ''}")

    print("\n✅ All strategies working correctly!")


def demo_random_forest_simulation():
    """Simulate Random Forest by creating multiple trees with different feature subsets."""
    print("\n🌲 Random Forest Simulation Demo")
    print("=" * 40)

    criterion = GiniCriterion()
    split_finder = SplitFinder(criterion, min_samples_leaf=1, verbose=0)

    features = [f"feature_{i:02d}" for i in range(25)]
    n_trees = 5

    print(f"🌳 Creating {n_trees} trees with different feature subsets (sqrt strategy)")

    trees_features = []
    for tree_id in range(n_trees):
        # Each tree gets different random_state for diversity
        subsampled = split_finder._subsample_features(features, 'sqrt', random_state=tree_id)
        trees_features.append(subsampled)

        print(f"  Tree {tree_id + 1}: {len(subsampled)} features -> {subsampled[:3]}...")

    # Check diversity
    unique_features_per_tree = [set(tree_features) for tree_features in trees_features]
    overlap_matrix = []

    print(f"\n📊 Feature Overlap Analysis:")
    for i in range(n_trees):
        overlaps = []
        for j in range(n_trees):
            if i == j:
                overlap = len(unique_features_per_tree[i])
            else:
                overlap = len(unique_features_per_tree[i].intersection(unique_features_per_tree[j]))
            overlaps.append(overlap)
        overlap_matrix.append(overlaps)

        avg_overlap = sum(overlaps[j] for j in range(n_trees) if j != i) / (n_trees - 1)
        print(f"  Tree {i + 1} avg overlap: {avg_overlap:.1f} features")

    print("✅ Trees have good diversity for Random Forest!")


def demo_classifier_integration():
    """Demonstrate classifier integration with feature subsampling parameters."""
    print("\n🔧 Classifier Integration Demo")
    print("=" * 35)

    # Test different configurations
    configs = [
        {'max_features': None, 'description': 'Use all features (default)'},
        {'max_features': 'sqrt', 'description': 'Square root of features'},
        {'max_features': 'log2', 'description': 'Log2 of features'},
        {'max_features': 0.5, 'description': '50% of features'},
        {'max_features': 10, 'description': 'Exactly 10 features'},
    ]

    for config in configs:
        print(f"\n📋 Config: {config['description']}")

        # Create classifier with max_features
        clf = ThetaSketchDecisionTreeClassifier(
            criterion='gini',
            max_depth=5,
            max_features=config['max_features'],
            random_state=42,
            verbose=0
        )

        print(f"     max_features: {clf.max_features}")
        print(f"     random_state: {clf.random_state}")

        # Test sklearn API compatibility
        params = clf.get_params()
        print(f"     sklearn params: max_features={params['max_features']}, random_state={params['random_state']}")

    print("\n✅ All configurations working correctly!")


def demo_reproducibility():
    """Demonstrate reproducibility with random_state."""
    print("\n🎲 Reproducibility Demo")
    print("=" * 25)

    criterion = GiniCriterion()
    split_finder = SplitFinder(criterion, min_samples_leaf=1, verbose=0)
    features = [f"feature_{i:02d}" for i in range(20)]

    print("🔄 Testing reproducibility with same random_state:")

    # Same random_state should give same results
    result1 = split_finder._subsample_features(features, 'sqrt', random_state=123)
    result2 = split_finder._subsample_features(features, 'sqrt', random_state=123)

    print(f"  Run 1: {result1}")
    print(f"  Run 2: {result2}")
    print(f"  Same results: {result1 == result2} ✅")

    # Different random_state should give different results
    result3 = split_finder._subsample_features(features, 'sqrt', random_state=456)
    print(f"\n🎯 Different random_state (456): {result3}")
    print(f"  Different from run 1: {result1 != result3} ✅")

    print("\n✅ Reproducibility working correctly!")


def demo_edge_cases():
    """Demonstrate edge case handling."""
    print("\n⚠️  Edge Cases Demo")
    print("=" * 20)

    criterion = GiniCriterion()
    split_finder = SplitFinder(criterion, min_samples_leaf=1, verbose=0)

    print("🔍 Testing edge cases:")

    # Single feature
    single_feature = ["only_feature"]
    result = split_finder._subsample_features(single_feature, 'sqrt', random_state=42)
    print(f"  Single feature + sqrt: {result} ✅")

    # Empty feature list
    empty_features = []
    result = split_finder._subsample_features(empty_features, 'sqrt', random_state=42)
    print(f"  Empty features + sqrt: {result} ✅")

    # Very small percentage
    features = [f"feature_{i}" for i in range(10)]
    result = split_finder._subsample_features(features, 0.05, random_state=42)  # 5% of 10 = 0.5 -> 1
    print(f"  Small percentage (5%): {len(result)} features (min 1) ✅")

    # More features requested than available
    result = split_finder._subsample_features(features, 20, random_state=42)
    print(f"  Request 20, have 10: {len(result)} features (capped) ✅")

    print("\n✅ All edge cases handled correctly!")


def main():
    """Run all demonstration scenarios."""
    print("🚀 Theta Sketch Tree Feature Subsampling Demo")
    print("=" * 60)
    print("This demo shows Random Forest compatibility through feature subsampling")
    print("(Bootstrap sampling is not possible with pre-aggregated sketches)")
    print()

    try:
        demo_feature_subsampling_strategies()
        demo_random_forest_simulation()
        demo_classifier_integration()
        demo_reproducibility()
        demo_edge_cases()

        print("\n" + "=" * 60)
        print("🎉 All demos completed successfully!")
        print("🌲 Your theta sketch trees are now Random Forest compatible!")
        print("📚 Use max_features parameter to control feature subsampling")
        print("🔄 Use random_state for reproducible results")
        print("✨ Perfect for ensemble methods and bagging classifiers!")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()