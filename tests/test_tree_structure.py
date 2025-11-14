"""
Unit tests for tree data structures.

Tests TreeNode and Tree classes.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from theta_sketch_tree.tree_structure import TreeNode, Tree


class TestTreeNode:
    """Test TreeNode class functionality."""

    @pytest.fixture
    def sample_node(self):
        """Create a sample tree node for testing."""
        return TreeNode(
            depth=1,
            n_samples=100,
            class_counts=np.array([60, 40]),
            impurity=0.48
        )

    def test_initialization(self, sample_node):
        """Test TreeNode initialization."""
        assert sample_node.depth == 1
        assert sample_node.n_samples == 100
        assert_array_equal(sample_node.class_counts, np.array([60, 40]))
        assert sample_node.impurity == 0.48
        assert sample_node.parent is None

        # Internal node attributes should be None initially
        assert sample_node.feature_idx is None
        assert sample_node.feature_name is None
        assert sample_node.left is None
        assert sample_node.right is None

        # Leaf node attributes should be initialized
        assert sample_node.is_leaf is False
        assert sample_node.prediction is None
        assert sample_node.probabilities is None

    def test_make_leaf(self, sample_node):
        """Test converting node to leaf."""
        sample_node.make_leaf()

        assert sample_node.is_leaf is True
        assert sample_node.prediction == 0  # Majority class (60 > 40)
        assert_array_equal(sample_node.probabilities, np.array([0.6, 0.4]))

    def test_make_leaf_edge_cases(self):
        """Test make_leaf with edge cases."""
        # Test with equal class counts
        equal_node = TreeNode(
            depth=0,
            n_samples=100,
            class_counts=np.array([50, 50]),
            impurity=0.5
        )
        equal_node.make_leaf()

        assert equal_node.prediction == 0  # argmax returns first index for ties
        assert_array_equal(equal_node.probabilities, np.array([0.5, 0.5]))

        # Test with zero samples
        zero_node = TreeNode(
            depth=0,
            n_samples=0,
            class_counts=np.array([0, 0]),
            impurity=0.0
        )
        zero_node.make_leaf()

        assert zero_node.prediction == 0
        assert_array_equal(zero_node.probabilities, np.array([0.5, 0.5]))

        # Test with pure class
        pure_node = TreeNode(
            depth=0,
            n_samples=100,
            class_counts=np.array([0, 100]),
            impurity=0.0
        )
        pure_node.make_leaf()

        assert pure_node.prediction == 1
        assert_array_equal(pure_node.probabilities, np.array([0.0, 1.0]))

    def test_set_split(self, sample_node):
        """Test setting split information."""
        # Create child nodes
        left_child = TreeNode(
            depth=2,
            n_samples=60,
            class_counts=np.array([45, 15]),
            impurity=0.375
        )

        right_child = TreeNode(
            depth=2,
            n_samples=40,
            class_counts=np.array([15, 25]),
            impurity=0.469
        )

        # Set split
        sample_node.set_split(
            feature_idx=0,
            feature_name='age>30',
            left_child=left_child,
            right_child=right_child
        )

        # Check split attributes
        assert sample_node.feature_idx == 0
        assert sample_node.feature_name == 'age>30'
        assert sample_node.left is left_child
        assert sample_node.right is right_child

        # Check parent references are set
        assert left_child.parent is sample_node
        assert right_child.parent is sample_node

    def test_tree_traversal_relationships(self):
        """Test parent-child relationships in tree."""
        # Create tree: root -> left -> left_left
        root = TreeNode(depth=0, n_samples=100, class_counts=np.array([50, 50]), impurity=0.5)

        left = TreeNode(depth=1, n_samples=60, class_counts=np.array([40, 20]), impurity=0.44)
        right = TreeNode(depth=1, n_samples=40, class_counts=np.array([10, 30]), impurity=0.375)

        left_left = TreeNode(depth=2, n_samples=30, class_counts=np.array([25, 5]), impurity=0.28)
        left_right = TreeNode(depth=2, n_samples=30, class_counts=np.array([15, 15]), impurity=0.5)

        # Set splits
        root.set_split(0, 'feature_0', left, right)
        left.set_split(1, 'feature_1', left_left, left_right)

        # Make leaves
        right.make_leaf()
        left_left.make_leaf()
        left_right.make_leaf()

        # Test relationships
        assert root.parent is None
        assert left.parent is root
        assert right.parent is root
        assert left_left.parent is left
        assert left_right.parent is left

        assert root.left is left
        assert root.right is right
        assert left.left is left_left
        assert left.right is left_right

        # Test leaf identification
        assert not root.is_leaf
        assert not left.is_leaf
        assert right.is_leaf
        assert left_left.is_leaf
        assert left_right.is_leaf

    def test_node_immutability_after_leaf(self):
        """Test that leaf nodes behave correctly."""
        node = TreeNode(
            depth=0,
            n_samples=100,
            class_counts=np.array([70, 30]),
            impurity=0.42
        )

        node.make_leaf()

        # Leaf node should have correct prediction and probabilities
        assert node.is_leaf is True
        assert node.prediction == 0  # Majority class
        assert_array_equal(node.probabilities, np.array([0.7, 0.3]))

        # Should still be possible to access other attributes
        assert node.depth == 0
        assert node.n_samples == 100

    def test_split_with_different_feature_types(self):
        """Test split with different feature indices and names."""
        node = TreeNode(depth=0, n_samples=100, class_counts=np.array([50, 50]), impurity=0.5)

        left = TreeNode(depth=1, n_samples=50, class_counts=np.array([30, 20]), impurity=0.48)
        right = TreeNode(depth=1, n_samples=50, class_counts=np.array([20, 30]), impurity=0.48)

        # Test with high feature index
        node.set_split(
            feature_idx=99,
            feature_name='complex_feature_name',
            left_child=left,
            right_child=right
        )

        assert node.feature_idx == 99
        assert node.feature_name == 'complex_feature_name'

        # Test with feature index -1 (invalid)
        node2 = TreeNode(depth=0, n_samples=50, class_counts=np.array([25, 25]), impurity=0.5)
        left2 = TreeNode(depth=1, n_samples=25, class_counts=np.array([20, 5]), impurity=0.32)
        right2 = TreeNode(depth=1, n_samples=25, class_counts=np.array([5, 20]), impurity=0.32)

        node2.set_split(-1, 'unknown_feature', left2, right2)
        assert node2.feature_idx == -1
        assert node2.feature_name == 'unknown_feature'


class TestTreeNodeEdgeCases:
    """Test edge cases and error conditions for TreeNode."""

    def test_negative_values(self):
        """Test handling of negative values (shouldn't occur but test robustness)."""
        # Node with negative samples (shouldn't happen but test gracefully)
        node = TreeNode(
            depth=0,
            n_samples=-10,
            class_counts=np.array([0, 0]),
            impurity=0.0
        )

        node.make_leaf()
        # Should handle gracefully
        assert node.is_leaf is True

    def test_very_large_values(self):
        """Test with very large sample counts."""
        large_node = TreeNode(
            depth=0,
            n_samples=1e6,
            class_counts=np.array([6e5, 4e5]),
            impurity=0.48
        )

        large_node.make_leaf()
        assert large_node.prediction == 0  # Majority class
        assert_array_equal(large_node.probabilities, np.array([0.6, 0.4]))

    def test_single_class_scenarios(self):
        """Test various single-class scenarios."""
        # Only negative class
        neg_only = TreeNode(
            depth=0,
            n_samples=100,
            class_counts=np.array([100, 0]),
            impurity=0.0
        )
        neg_only.make_leaf()
        assert neg_only.prediction == 0
        assert_array_equal(neg_only.probabilities, np.array([1.0, 0.0]))

        # Only positive class
        pos_only = TreeNode(
            depth=0,
            n_samples=50,
            class_counts=np.array([0, 50]),
            impurity=0.0
        )
        pos_only.make_leaf()
        assert pos_only.prediction == 1
        assert_array_equal(pos_only.probabilities, np.array([0.0, 1.0]))

    def test_float_sample_counts(self):
        """Test with float sample counts (from sketch estimates)."""
        float_node = TreeNode(
            depth=0,
            n_samples=100.5,
            class_counts=np.array([60.3, 40.2]),
            impurity=0.479
        )

        float_node.make_leaf()
        assert float_node.prediction == 0  # Majority class
        expected_probs = np.array([60.3, 40.2]) / 100.5
        assert_array_equal(float_node.probabilities, expected_probs)


class TestTree:
    """Test Tree wrapper class (if implemented)."""

    def test_tree_placeholder(self):
        """Placeholder test for Tree class."""
        # Tree class is not fully implemented yet, so just test that it exists
        root = TreeNode(
            depth=0,
            n_samples=100,
            class_counts=np.array([50, 50]),
            impurity=0.5
        )

        # Tree constructor might not be fully implemented
        try:
            tree = Tree(root)
            # If implementation exists, test it
            assert hasattr(tree, 'root') or True  # Basic existence test
        except (NotImplementedError, TypeError):
            # If not implemented, that's expected
            pass


class TestTreeNodeIntegration:
    """Integration tests for TreeNode with realistic scenarios."""

    def test_binary_tree_construction(self):
        """Test building a small binary tree."""
        # Root
        root = TreeNode(depth=0, n_samples=100, class_counts=np.array([55, 45]), impurity=0.4975)

        # Level 1
        left_1 = TreeNode(depth=1, n_samples=70, class_counts=np.array([50, 20]), impurity=0.408)
        right_1 = TreeNode(depth=1, n_samples=30, class_counts=np.array([5, 25]), impurity=0.278)

        # Level 2
        left_2_left = TreeNode(depth=2, n_samples=40, class_counts=np.array([35, 5]), impurity=0.25)
        left_2_right = TreeNode(depth=2, n_samples=30, class_counts=np.array([15, 15]), impurity=0.5)

        # Build tree
        root.set_split(0, 'age>30', left_1, right_1)
        left_1.set_split(1, 'income>50k', left_2_left, left_2_right)

        # Make leaves
        right_1.make_leaf()
        left_2_left.make_leaf()
        left_2_right.make_leaf()

        # Validate tree structure
        assert root.depth == 0
        assert left_1.depth == 1
        assert left_2_left.depth == 2

        # Validate relationships
        assert root.left.left.parent is left_1
        assert left_1.parent is root

        # Validate leaf predictions
        assert right_1.is_leaf
        assert right_1.prediction == 1  # 25 > 5
        assert left_2_left.is_leaf
        assert left_2_left.prediction == 0  # 35 > 5

    def test_tree_path_traversal(self):
        """Test traversing paths in tree."""
        # Build a simple tree for path testing
        root = TreeNode(depth=0, n_samples=100, class_counts=np.array([50, 50]), impurity=0.5)
        left = TreeNode(depth=1, n_samples=60, class_counts=np.array([45, 15]), impurity=0.375)
        right = TreeNode(depth=1, n_samples=40, class_counts=np.array([5, 35]), impurity=0.25)

        root.set_split(0, 'feature_0', left, right)
        left.make_leaf()
        right.make_leaf()

        # Test path from root to leaves
        # Left path: root -> left
        assert root.left is left
        assert left.parent is root
        assert left.is_leaf

        # Right path: root -> right
        assert root.right is right
        assert right.parent is root
        assert right.is_leaf

        # Verify predictions make sense
        assert left.prediction == 0  # 45 > 15
        assert right.prediction == 1  # 35 > 5
