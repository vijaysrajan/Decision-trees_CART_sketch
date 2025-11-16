"""
Unit tests for feature importance calculation.

Tests FeatureImportanceCalculator and related functionality.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from theta_sketch_tree.feature_importance import (
    FeatureImportanceCalculator, compute_feature_importances
)
from theta_sketch_tree.tree_structure import TreeNode


class TestFeatureImportanceCalculator:
    """Test FeatureImportanceCalculator class."""

    @pytest.fixture
    def feature_names(self):
        """Sample feature names for testing."""
        return ['age>30', 'income>50k', 'has_degree']

    @pytest.fixture
    def calculator(self, feature_names):
        """Feature importance calculator instance."""
        return FeatureImportanceCalculator(feature_names)

    @pytest.fixture
    def simple_tree(self):
        """Create a simple tree for testing."""
        # Root node: 100 samples, balanced classes, impurity=0.5
        root = TreeNode(
            depth=0,
            n_samples=100,
            class_counts=np.array([50, 50]),
            impurity=0.5
        )

        # Left child: pure negative class
        left_child = TreeNode(
            depth=1,
            n_samples=60,
            class_counts=np.array([60, 0]),
            impurity=0.0
        )
        left_child.make_leaf()

        # Right child: pure positive class
        right_child = TreeNode(
            depth=1,
            n_samples=40,
            class_counts=np.array([0, 40]),
            impurity=0.0
        )
        right_child.make_leaf()

        # Set split on root (using first feature)
        root.set_split(
            feature_idx=0,
            feature_name='age>30',
            left_child=left_child,
            right_child=right_child
        )

        return root

    @pytest.fixture
    def multi_level_tree(self):
        """Create a multi-level tree for testing."""
        # Root node
        root = TreeNode(
            depth=0,
            n_samples=100,
            class_counts=np.array([60, 40]),
            impurity=0.48  # 1 - (0.6^2 + 0.4^2) = 0.48
        )

        # Level 1 - split on 'age>30'
        left_l1 = TreeNode(
            depth=1,
            n_samples=70,
            class_counts=np.array([50, 20]),
            impurity=0.408  # approximate
        )

        right_l1 = TreeNode(
            depth=1,
            n_samples=30,
            class_counts=np.array([10, 20]),
            impurity=0.444  # approximate
        )

        # Level 2 - split left child on 'income>50k'
        left_l2_left = TreeNode(
            depth=2,
            n_samples=40,
            class_counts=np.array([35, 5]),
            impurity=0.25
        )
        left_l2_left.make_leaf()

        left_l2_right = TreeNode(
            depth=2,
            n_samples=30,
            class_counts=np.array([15, 15]),
            impurity=0.5
        )
        left_l2_right.make_leaf()

        # Make right_l1 a leaf
        right_l1.make_leaf()

        # Set splits
        left_l1.set_split(
            feature_idx=1,
            feature_name='income>50k',
            left_child=left_l2_left,
            right_child=left_l2_right
        )

        root.set_split(
            feature_idx=0,
            feature_name='age>30',
            left_child=left_l1,
            right_child=right_l1
        )

        return root

    def test_initialization(self, feature_names):
        """Test calculator initialization."""
        calc = FeatureImportanceCalculator(feature_names)
        assert calc.feature_names == feature_names
        assert calc.n_features == len(feature_names)

    def test_single_split_importance(self, calculator, simple_tree):
        """Test importance calculation for tree with single split."""
        importances = calculator.compute_importances(simple_tree)

        # Should have same length as features
        assert len(importances) == 3

        # Should sum to 1.0
        assert_allclose(importances.sum(), 1.0)

        # All should be non-negative
        assert np.all(importances >= 0)

        # First feature (age>30) should have positive importance since it was used
        assert importances[0] > 0

        # Other features should have zero importance since they weren't used
        assert importances[1] == 0
        assert importances[2] == 0

    def test_multi_level_importance(self, calculator, multi_level_tree):
        """Test importance calculation for multi-level tree."""
        importances = calculator.compute_importances(multi_level_tree)

        # Should sum to 1.0
        assert_allclose(importances.sum(), 1.0)

        # Both age>30 and income>50k should have positive importance
        assert importances[0] > 0  # age>30
        assert importances[1] > 0  # income>50k
        assert importances[2] == 0  # has_degree (unused)

        # Root split should generally have higher importance than child splits
        # (This is not always true, but in our constructed example it should be)
        assert importances[0] >= importances[1]

    def test_single_node_tree(self, calculator):
        """Test importance calculation for single node (no splits)."""
        root = TreeNode(
            depth=0,
            n_samples=100,
            class_counts=np.array([60, 40]),
            impurity=0.48
        )
        root.make_leaf()

        importances = calculator.compute_importances(root)

        # Should return equal importance for all features
        expected = np.ones(3) / 3
        assert_allclose(importances, expected)

    def test_get_feature_importance_dict(self, calculator, simple_tree):
        """Test conversion to feature importance dictionary."""
        importances = calculator.compute_importances(simple_tree)
        importance_dict = calculator.get_feature_importance_dict(importances)

        # Should be a dict with all feature names
        assert isinstance(importance_dict, dict)
        assert len(importance_dict) == 3
        assert set(importance_dict.keys()) == set(calculator.feature_names)

        # Values should match the array
        for i, feature_name in enumerate(calculator.feature_names):
            assert importance_dict[feature_name] == importances[i]

    def test_get_top_features(self, calculator, multi_level_tree):
        """Test getting top k most important features."""
        importances = calculator.compute_importances(multi_level_tree)

        # Get top 2 features
        top_features = calculator.get_top_features(importances, top_k=2)

        assert len(top_features) == 2
        assert all(isinstance(item, tuple) for item in top_features)
        assert all(len(item) == 2 for item in top_features)

        # Should be sorted by importance (descending)
        assert top_features[0][1] >= top_features[1][1]

        # First feature should be one that was actually used
        assert top_features[0][0] in ['age>30', 'income>50k']

    def test_get_top_features_more_than_available(self, calculator, simple_tree):
        """Test getting more top features than available."""
        importances = calculator.compute_importances(simple_tree)

        # Ask for more features than we have
        top_features = calculator.get_top_features(importances, top_k=10)

        # Should return all features
        assert len(top_features) == 3

    def test_edge_cases(self, calculator):
        """Test edge cases and error conditions."""
        # Test with zero samples
        zero_node = TreeNode(
            depth=0,
            n_samples=0,
            class_counts=np.array([0, 0]),
            impurity=0.0
        )
        zero_node.make_leaf()

        importances = calculator.compute_importances(zero_node)
        assert np.allclose(importances.sum(), 1.0)

    def test_unknown_feature_in_tree(self, calculator):
        """Test handling of unknown feature names in tree."""
        # Create tree with unknown feature
        root = TreeNode(
            depth=0,
            n_samples=100,
            class_counts=np.array([50, 50]),
            impurity=0.5
        )

        left_child = TreeNode(
            depth=1,
            n_samples=60,
            class_counts=np.array([60, 0]),
            impurity=0.0
        )
        left_child.make_leaf()

        right_child = TreeNode(
            depth=1,
            n_samples=40,
            class_counts=np.array([0, 40]),
            impurity=0.0
        )
        right_child.make_leaf()

        # Use unknown feature name
        root.set_split(
            feature_idx=-1,
            feature_name='unknown_feature',
            left_child=left_child,
            right_child=right_child
        )

        # Should handle gracefully (skip unknown feature)
        importances = calculator.compute_importances(root)
        assert np.allclose(importances.sum(), 1.0)


class TestConvenienceFunction:
    """Test the compute_feature_importances convenience function."""

    def test_convenience_function(self):
        """Test the standalone convenience function."""
        # Create simple tree
        root = TreeNode(
            depth=0,
            n_samples=100,
            class_counts=np.array([50, 50]),
            impurity=0.5
        )

        left_child = TreeNode(
            depth=1,
            n_samples=60,
            class_counts=np.array([60, 0]),
            impurity=0.0
        )
        left_child.make_leaf()

        right_child = TreeNode(
            depth=1,
            n_samples=40,
            class_counts=np.array([0, 40]),
            impurity=0.0
        )
        right_child.make_leaf()

        root.set_split(
            feature_idx=0,
            feature_name='feature_A',
            left_child=left_child,
            right_child=right_child
        )

        # Test convenience function
        feature_names = ['feature_A', 'feature_B']
        importances = compute_feature_importances(root, feature_names)

        assert len(importances) == 2
        assert_allclose(importances.sum(), 1.0)
        assert importances[0] > 0  # Used feature
        assert importances[1] == 0  # Unused feature


class TestFeatureImportanceEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_feature_list(self):
        """Test with empty feature list."""
        calc = FeatureImportanceCalculator([])

        # Create single node tree
        root = TreeNode(
            depth=0,
            n_samples=100,
            class_counts=np.array([50, 50]),
            impurity=0.5
        )
        root.make_leaf()

        importances = calc.compute_importances(root)
        assert len(importances) == 0

    def test_negative_impurity_handling(self):
        """Test handling of negative impurity decreases (shouldn't happen in practice)."""
        calc = FeatureImportanceCalculator(['feature_A'])

        # Create tree with artificially bad split (child impurity > parent)
        root = TreeNode(
            depth=0,
            n_samples=100,
            class_counts=np.array([50, 50]),
            impurity=0.1  # Artificially low
        )

        left_child = TreeNode(
            depth=1,
            n_samples=50,
            class_counts=np.array([25, 25]),
            impurity=0.5  # Higher than parent (bad split)
        )
        left_child.make_leaf()

        right_child = TreeNode(
            depth=1,
            n_samples=50,
            class_counts=np.array([25, 25]),
            impurity=0.5  # Higher than parent (bad split)
        )
        right_child.make_leaf()

        root.set_split(
            feature_idx=0,
            feature_name='feature_A',
            left_child=left_child,
            right_child=right_child
        )

        # Should handle negative impurity decrease gracefully
        importances = calc.compute_importances(root)
        assert np.all(importances >= 0)  # Should be non-negative
        assert_allclose(importances.sum(), 1.0)  # Should still sum to 1

    def test_zero_child_samples_edge_case(self):
        """Test edge case where both children have zero samples (line 127 coverage)."""
        feature_names = ['feature_A']
        calc = FeatureImportanceCalculator(feature_names)

        # Create parent node with samples
        root = TreeNode(
            depth=0,
            n_samples=100,
            class_counts=np.array([50, 50]),
            impurity=0.5
        )

        # Create children with zero samples (malformed tree - shouldn't happen in practice)
        left_child = TreeNode(
            depth=1,
            n_samples=0,  # ← Zero samples
            class_counts=np.array([0, 0]),
            impurity=0.0
        )
        left_child.make_leaf()

        right_child = TreeNode(
            depth=1,
            n_samples=0,  # ← Zero samples
            class_counts=np.array([0, 0]),
            impurity=0.0
        )
        right_child.make_leaf()

        # Set split (this creates the malformed case)
        root.set_split(
            feature_idx=0,
            feature_name='feature_A',
            left_child=left_child,
            right_child=right_child
        )

        # This should trigger line 127: weighted_child_impurity = parent_impurity
        # because total_child_samples = 0 + 0 = 0
        importances = calc.compute_importances(root)

        # Should handle gracefully - when both children have zero samples,
        # impurity_decrease = parent_impurity - parent_impurity = 0
        # But normalized, this becomes 1.0 since it's the only feature
        assert len(importances) == 1
        assert importances[0] == 1.0  # Still gets normalized to 1.0 since it's the only feature
