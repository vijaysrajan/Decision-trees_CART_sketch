#!/usr/bin/env python3
"""
Test both intersection and ratio-based modes to verify they work correctly.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_mushroom_sketches import (
    load_mushroom_dataset, create_mushroom_sketches, create_mushroom_feature_mapping,
    DEFAULT_LG_K, DEFAULT_MIN_SAMPLES_SPLIT, DEFAULT_MIN_SAMPLES_LEAF, DEFAULT_MAX_DEPTH,
    DEFAULT_CRITERION, DEFAULT_VERBOSE
)
from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier


def test_both_modes():
    """Test both tree builder modes work with real theta sketches."""
    print("🧪 Testing Both Tree Builder Modes with Real Theta Sketches")
    print("=" * 60)

    # Load small dataset for fast testing
    df = load_mushroom_dataset()
    sketches = create_mushroom_sketches(df.sample(200, random_state=42), lg_k=DEFAULT_LG_K)
    mapping = create_mushroom_feature_mapping(sketches)

    print(f"📊 Dataset: 200 samples, {len(mapping)} features")

    # Test parameters
    test_params = {
        'criterion': DEFAULT_CRITERION,
        'max_depth': DEFAULT_MAX_DEPTH,
        'min_samples_split': DEFAULT_MIN_SAMPLES_SPLIT,
        'min_samples_leaf': DEFAULT_MIN_SAMPLES_LEAF,
        'verbose': DEFAULT_VERBOSE
    }

    results = {}

    for mode in ["intersection", "ratio_based"]:
        print(f"\n🔄 Testing {mode.upper()} mode...")

        try:
            clf = ThetaSketchDecisionTreeClassifier(
                **test_params,
                tree_builder=mode
            )
            clf.fit(sketches, mapping)

            print(f"✅ {mode} mode: Success")
            print(f"   Root samples: {clf.tree_.n_samples:.0f}")
            print(f"   Tree depth: {clf.tree_.depth}")

            # Test prediction
            import numpy as np
            np.random.seed(42)
            X_test = np.random.randint(0, 2, size=(10, len(mapping)))
            predictions = clf.predict(X_test)

            results[mode] = {
                'success': True,
                'root_samples': clf.tree_.n_samples,
                'tree_depth': clf.tree_.depth,
                'predictions': predictions
            }

        except Exception as e:
            print(f"❌ {mode} mode: Failed - {e}")
            results[mode] = {'success': False, 'error': str(e)}

    # Compare results
    print("\n" + "=" * 60)
    print("🔍 COMPARISON RESULTS")
    print("=" * 60)

    for mode, result in results.items():
        if result['success']:
            print(f"✅ {mode.upper()}: Working correctly")
            print(f"   Root samples: {result['root_samples']:.0f}")
            print(f"   Tree depth: {result['tree_depth']}")
        else:
            print(f"❌ {mode.upper()}: Failed - {result['error']}")

    # Check if both succeeded
    if all(r['success'] for r in results.values()):
        print("\n🎉 Both tree builder modes are working correctly with real theta sketches!")
        return True
    else:
        print("\n⚠️  Some modes failed - check errors above")
        return False


if __name__ == "__main__":
    success = test_both_modes()
    exit(0 if success else 1)