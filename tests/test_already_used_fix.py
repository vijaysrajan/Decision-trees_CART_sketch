#!/usr/bin/env python3
"""
Test script to verify the already_used logic fix for categorical features.

This test verifies that:
1. Left child (absent) can still use other values of the same categorical variable
2. Right child (present) cannot use ANY other value of the same categorical variable
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from theta_sketch_tree.tree_orchestrator import TreeOrchestrator
from theta_sketch_tree.criteria import GiniCriterion


def test_already_used_logic():
    """Test the _create_child_already_used_sets method."""

    # Create orchestrator
    orchestrator = TreeOrchestrator(
        criterion=GiniCriterion(),
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5
    )

    # Test data: categorical features for cities and booking types
    all_features = [
        'city=Bangalore', 'city=Mumbai', 'city=Delhi', 'city=Chennai',
        'booking_type=one_way', 'booking_type=round_trip', 'booking_type=outstation',
        'standalone_feature_1', 'standalone_feature_2'
    ]

    # Test Case 1: Split on categorical feature city=Bangalore
    print("🔍 Test Case 1: Split on city=Bangalore")
    parent_already_used = {'booking_type=one_way'}  # Already used one booking type
    split_feature = 'city=Bangalore'

    left_used, right_used = orchestrator._create_child_already_used_sets(
        parent_already_used, split_feature, all_features
    )

    print(f"   Parent already_used: {parent_already_used}")
    print(f"   Split feature: {split_feature}")
    print(f"   Left child (absent) already_used: {left_used}")
    print(f"   Right child (present) already_used: {right_used}")

    # Verify expectations
    # Left child: should exclude only city=Bangalore, can use other cities
    expected_left = {'booking_type=one_way', 'city=Bangalore'}
    assert left_used == expected_left, f"Left child mismatch: {left_used} != {expected_left}"

    # Right child: should exclude ALL city features
    expected_right = {'booking_type=one_way', 'city=Bangalore', 'city=Mumbai', 'city=Delhi', 'city=Chennai'}
    assert right_used == expected_right, f"Right child mismatch: {right_used} != {expected_right}"

    print("   ✅ Test Case 1 PASSED")
    print()

    # Test Case 2: Split on standalone feature
    print("🔍 Test Case 2: Split on standalone_feature_1")
    parent_already_used = {'city=Mumbai'}
    split_feature = 'standalone_feature_1'

    left_used, right_used = orchestrator._create_child_already_used_sets(
        parent_already_used, split_feature, all_features
    )

    print(f"   Parent already_used: {parent_already_used}")
    print(f"   Split feature: {split_feature}")
    print(f"   Left child (absent) already_used: {left_used}")
    print(f"   Right child (present) already_used: {right_used}")

    # For standalone features, both children should have the same already_used
    expected_both = {'city=Mumbai', 'standalone_feature_1'}
    assert left_used == expected_both, f"Left child mismatch: {left_used} != {expected_both}"
    assert right_used == expected_both, f"Right child mismatch: {right_used} != {expected_both}"

    print("   ✅ Test Case 2 PASSED")
    print()

    # Test Case 3: Verify available features after split using Test Case 1 results
    print("🔍 Test Case 3: Available features after categorical split")

    # Use the already_used sets from Test Case 1 (city=Bangalore split)
    test1_parent_already_used = {'booking_type=one_way'}
    test1_split_feature = 'city=Bangalore'

    test1_left_used, test1_right_used = orchestrator._create_child_already_used_sets(
        test1_parent_already_used, test1_split_feature, all_features
    )

    # Simulate: After splitting on city=Bangalore, what features are available?
    from theta_sketch_tree.split_finder import SplitFinder

    # Mock sketch_dict - need to provide valid sketch data
    sketch_dict = {
        'positive': {f: [None, None] for f in all_features},  # [present_sketch, absent_sketch]
        'negative': {f: [None, None] for f in all_features}
    }

    finder = SplitFinder(GiniCriterion(), min_samples_leaf=1)

    # Left child (absent): should see other cities
    left_available = finder._get_available_features(all_features, test1_left_used, sketch_dict)
    print(f"   Left child available features: {left_available}")
    print(f"   Left child already_used: {test1_left_used}")

    # Should include other cities but not Bangalore
    assert 'city=Mumbai' in left_available, f"Left child should see city=Mumbai. Available: {left_available}, Already used: {test1_left_used}"
    assert 'city=Delhi' in left_available, "Left child should see city=Delhi"
    assert 'city=Bangalore' not in left_available, "Left child should NOT see city=Bangalore"

    # Right child (present): should NOT see any cities
    right_available = finder._get_available_features(all_features, test1_right_used, sketch_dict)
    print(f"   Right child available features: {right_available}")
    print(f"   Right child already_used: {test1_right_used}")

    # Should not include any cities
    assert 'city=Mumbai' not in right_available, "Right child should NOT see city=Mumbai"
    assert 'city=Delhi' not in right_available, "Right child should NOT see city=Delhi"
    assert 'city=Chennai' not in right_available, "Right child should NOT see city=Chennai"
    assert 'city=Bangalore' not in right_available, "Right child should NOT see city=Bangalore"

    # Should still see booking types (different categorical variable)
    assert 'booking_type=round_trip' in right_available, "Right child should see other booking types"

    print("   ✅ Test Case 3 PASSED")
    print()

    print("🎉 All tests PASSED! The already_used logic is now correct.")


if __name__ == '__main__':
    test_already_used_logic()