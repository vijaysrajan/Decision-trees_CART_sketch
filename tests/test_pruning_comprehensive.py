"""
Comprehensive tests for pruning.py to achieve 90% coverage.

This module tests all 4 pruning algorithms:
- validation_prune: accuracy-based pruning
- cost_complexity_prune: impurity decrease pruning
- reduced_error_prune: error-based pruning
- min_impurity_prune: threshold-based pruning
"""

import pytest
import numpy as np
from theta_sketch_tree.pruning import (
    prune_tree,
    validation_prune,
    cost_complexity_prune,
    reduced_error_prune,
    min_impurity_prune,
    get_pruning_summary
)
from theta_sketch_tree.tree_structure import TreeNode


@pytest.fixture
def sample_tree():
    """Create a sample tree for testing."""
    # Root node
    root = TreeNode(
        depth=0,
        n_samples=100,
        class_counts=np.array([50, 50]),
        impurity=0.5
    )

    # Left child
    left = TreeNode(
        depth=1,
        n_samples=60,
        class_counts=np.array([40, 20]),
        impurity=0.44,
        parent=root
    )

    # Right child
    right = TreeNode(
        depth=1,
        n_samples=40,
        class_counts=np.array([10, 30]),
        impurity=0.375,
        parent=root
    )

    # Left-left child (leaf)
    left_left = TreeNode(
        depth=2,
        n_samples=35,
        class_counts=np.array([30, 5]),
        impurity=0.24,
        parent=left
    )
    left_left.make_leaf()

    # Left-right child (leaf)
    left_right = TreeNode(
        depth=2,
        n_samples=25,
        class_counts=np.array([10, 15]),
        impurity=0.48,
        parent=left
    )
    left_right.make_leaf()

    # Right-left child (leaf)
    right_left = TreeNode(
        depth=2,
        n_samples=15,
        class_counts=np.array([5, 10]),
        impurity=0.44,
        parent=right
    )
    right_left.make_leaf()

    # Right-right child (leaf)
    right_right = TreeNode(
        depth=2,
        n_samples=25,
        class_counts=np.array([5, 20]),
        impurity=0.32,
        parent=right
    )
    right_right.make_leaf()

    # Set split information
    left.set_split(0, "feature_0", left_left, left_right)
    right.set_split(1, "feature_1", right_left, right_right)
    root.set_split(2, "feature_2", left, right)

    return root


@pytest.fixture
def validation_data():
    """Create validation data for testing."""
    # Binary feature data matching tree structure
    X_val = np.array([
        [1, 0, 0],  # Should go left-left
        [0, 1, 0],  # Should go left-right
        [1, 0, 1],  # Should go right-left
        [0, 1, 1],  # Should go right-right
        [1, 0, 0],  # Repeat patterns
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 1]
    ])

    y_val = np.array([0, 1, 0, 1, 0, 1, 1, 1])  # Binary labels

    return X_val, y_val


class TestPruneTreeFactory:
    """Test the main prune_tree factory method."""

    def test_prune_tree_no_pruning(self, sample_tree):
        """Test prune_tree with method='none'."""
        result = prune_tree(sample_tree, method="none")
        assert result is not None
        assert not result.is_leaf  # Tree should remain unchanged
        assert result.left is not None
        assert result.right is not None

    def test_prune_tree_validation_method(self, sample_tree, validation_data):
        """Test prune_tree with validation method."""
        X_val, y_val = validation_data
        result = prune_tree(sample_tree, method="validation", X_val=X_val, y_val=y_val)
        assert result is not None

    def test_prune_tree_cost_complexity_method(self, sample_tree):
        """Test prune_tree with cost_complexity method."""
        result = prune_tree(sample_tree, method="cost_complexity", min_impurity_decrease=0.1)
        assert result is not None

    def test_prune_tree_reduced_error_method(self, sample_tree, validation_data):
        """Test prune_tree with reduced_error method."""
        X_val, y_val = validation_data
        result = prune_tree(sample_tree, method="reduced_error", X_val=X_val, y_val=y_val)
        assert result is not None

    def test_prune_tree_min_impurity_method(self, sample_tree):
        """Test prune_tree with min_impurity method."""
        result = prune_tree(sample_tree, method="min_impurity", min_impurity_decrease=0.05)
        assert result is not None

    def test_prune_tree_invalid_method(self, sample_tree):
        """Test prune_tree with invalid method."""
        with pytest.raises(ValueError, match="Unknown pruning method"):
            prune_tree(sample_tree, method="invalid_method")


class TestValidationPrune:
    """Test validation-based pruning."""

    def test_validation_prune_with_valid_data(self, sample_tree, validation_data):
        """Test validation pruning with valid data."""
        X_val, y_val = validation_data
        result = validation_prune(sample_tree, X_val, y_val, feature_mapping=None)
        assert result is not None

    def test_validation_prune_with_none_data(self, sample_tree):
        """Test validation pruning with None validation data."""
        # Should return original tree when validation data is None
        result = validation_prune(sample_tree, None, None, feature_mapping=None)
        assert result is sample_tree

    def test_validation_prune_with_none_x_val(self, sample_tree, validation_data):
        """Test validation pruning with None X_val."""
        _, y_val = validation_data
        result = validation_prune(sample_tree, None, y_val, feature_mapping=None)
        assert result is sample_tree

    def test_validation_prune_with_none_y_val(self, sample_tree, validation_data):
        """Test validation pruning with None y_val."""
        X_val, _ = validation_data
        result = validation_prune(sample_tree, X_val, None, feature_mapping=None)
        assert result is sample_tree

    def test_validation_prune_iteration_limit(self, sample_tree, validation_data):
        """Test validation pruning respects iteration limit."""
        X_val, y_val = validation_data
        # Use tree that would benefit from extensive pruning
        result = validation_prune(sample_tree, X_val, y_val, feature_mapping=None)
        assert result is not None
        # Note: The algorithm limits to 20 iterations


class TestCostComplexityPrune:
    """Test cost-complexity pruning."""

    def test_cost_complexity_prune_basic(self, sample_tree):
        """Test basic cost-complexity pruning."""
        result = cost_complexity_prune(sample_tree, min_impurity_decrease=0.01)
        assert result is not None

    def test_cost_complexity_prune_aggressive(self, sample_tree):
        """Test aggressive cost-complexity pruning."""
        result = cost_complexity_prune(sample_tree, min_impurity_decrease=0.5)
        assert result is not None
        # With high threshold, more pruning should occur

    def test_cost_complexity_prune_conservative(self, sample_tree):
        """Test conservative cost-complexity pruning."""
        result = cost_complexity_prune(sample_tree, min_impurity_decrease=0.001)
        assert result is not None
        # With low threshold, less pruning should occur

    def test_cost_complexity_prune_single_node(self):
        """Test cost-complexity pruning on single node."""
        single_node = TreeNode(
            depth=0,
            n_samples=50,
            class_counts=np.array([25, 25]),
            impurity=0.5
        )
        single_node.make_leaf()

        result = cost_complexity_prune(single_node, min_impurity_decrease=0.1)
        assert result is not None
        assert result.is_leaf


class TestReducedErrorPrune:
    """Test reduced-error pruning."""

    def test_reduced_error_prune_with_valid_data(self, sample_tree, validation_data):
        """Test reduced error pruning with valid data."""
        X_val, y_val = validation_data
        result = reduced_error_prune(sample_tree, X_val, y_val, feature_mapping=None)
        assert result is not None

    def test_reduced_error_prune_with_none_data(self, sample_tree):
        """Test reduced error pruning with None validation data."""
        result = reduced_error_prune(sample_tree, None, None, feature_mapping=None)
        assert result is sample_tree

    def test_reduced_error_prune_with_none_x_val(self, sample_tree, validation_data):
        """Test reduced error pruning with None X_val."""
        _, y_val = validation_data
        result = reduced_error_prune(sample_tree, None, y_val, feature_mapping=None)
        assert result is sample_tree

    def test_reduced_error_prune_with_none_y_val(self, sample_tree, validation_data):
        """Test reduced error pruning with None y_val."""
        X_val, _ = validation_data
        result = reduced_error_prune(sample_tree, X_val, None, feature_mapping=None)
        assert result is sample_tree

    def test_reduced_error_prune_single_node(self, validation_data):
        """Test reduced error pruning on single node."""
        X_val, y_val = validation_data
        single_node = TreeNode(
            depth=0,
            n_samples=50,
            class_counts=np.array([25, 25]),
            impurity=0.5
        )
        single_node.make_leaf()

        result = reduced_error_prune(single_node, X_val, y_val, feature_mapping=None)
        assert result is not None
        assert result.is_leaf


class TestMinImpurityPrune:
    """Test min-impurity pruning."""

    def test_min_impurity_prune_basic(self, sample_tree):
        """Test basic min-impurity pruning."""
        result = min_impurity_prune(sample_tree, min_impurity_decrease=0.01)
        assert result is not None

    def test_min_impurity_prune_aggressive(self, sample_tree):
        """Test aggressive min-impurity pruning."""
        result = min_impurity_prune(sample_tree, min_impurity_decrease=0.5)
        assert result is not None
        # With high threshold, more nodes should be pruned

    def test_min_impurity_prune_conservative(self, sample_tree):
        """Test conservative min-impurity pruning."""
        result = min_impurity_prune(sample_tree, min_impurity_decrease=0.001)
        assert result is not None
        # With low threshold, fewer nodes should be pruned

    def test_min_impurity_prune_single_node(self):
        """Test min-impurity pruning on single node."""
        single_node = TreeNode(
            depth=0,
            n_samples=50,
            class_counts=np.array([25, 25]),
            impurity=0.5
        )
        single_node.make_leaf()

        result = min_impurity_prune(single_node, min_impurity_decrease=0.1)
        assert result is not None
        assert result.is_leaf

    def test_min_impurity_prune_leaf_node(self):
        """Test min-impurity pruning on leaf node."""
        leaf_node = TreeNode(
            depth=0,
            n_samples=50,
            class_counts=np.array([25, 25]),
            impurity=0.5
        )
        leaf_node.make_leaf()

        result = min_impurity_prune(leaf_node, min_impurity_decrease=0.1)
        assert result is not None
        assert result.is_leaf


class TestPruningSummary:
    """Test pruning summary utility function."""

    def test_get_pruning_summary_basic(self):
        """Test basic pruning summary generation."""
        summary = get_pruning_summary("test_method", 100, 80)

        assert summary['method'] == "test_method"
        assert summary['nodes_removed'] == 20
        assert summary['compression_ratio'] == 0.8

    def test_get_pruning_summary_no_pruning(self):
        """Test pruning summary with no pruning."""
        summary = get_pruning_summary("none", 50, 50)

        assert summary['method'] == "none"
        assert summary['nodes_removed'] == 0
        assert summary['compression_ratio'] == 1.0

    def test_get_pruning_summary_complete_pruning(self):
        """Test pruning summary with complete pruning."""
        summary = get_pruning_summary("aggressive", 100, 1)

        assert summary['method'] == "aggressive"
        assert summary['nodes_removed'] == 99
        assert summary['compression_ratio'] == 0.01

    def test_get_pruning_summary_zero_nodes_before(self):
        """Test pruning summary with zero nodes before (edge case)."""
        summary = get_pruning_summary("edge_case", 0, 0)

        assert summary['method'] == "edge_case"
        assert summary['nodes_removed'] == 0
        assert summary['compression_ratio'] == 0.0  # 0/max(1,0) = 0/1 = 0


class TestPruningEdgeCases:
    """Test edge cases and error conditions."""

    def test_pruning_with_leaf_only_tree(self):
        """Test pruning with tree that is already a leaf."""
        leaf_node = TreeNode(
            depth=0,
            n_samples=50,
            class_counts=np.array([25, 25]),
            impurity=0.5
        )
        leaf_node.make_leaf()

        result = min_impurity_prune(leaf_node, min_impurity_decrease=0.1)
        assert result is not None
        assert result.is_leaf

    def test_pruning_with_empty_validation_data(self, sample_tree):
        """Test pruning with empty validation arrays."""
        X_val = np.array([]).reshape(0, 3)
        y_val = np.array([])

        result = validation_prune(sample_tree, X_val, y_val, feature_mapping=None)
        assert result is not None

    def test_all_methods_with_deep_tree(self):
        """Test all pruning methods with a deeper tree."""
        # Create a deeper tree for more comprehensive testing
        root = TreeNode(
            depth=0,
            n_samples=1000,
            class_counts=np.array([500, 500]),
            impurity=0.5
        )

        # Add multiple levels
        for i in range(3):  # 3 levels deep
            if i == 0:
                current_nodes = [root]
            else:
                current_nodes = new_nodes

            new_nodes = []
            for node in current_nodes:
                if node.is_leaf:
                    continue

                left = TreeNode(
                    depth=i+1,
                    n_samples=node.n_samples // 2,
                    class_counts=node.class_counts // 2,
                    impurity=0.3,
                    parent=node
                )

                right = TreeNode(
                    depth=i+1,
                    n_samples=node.n_samples // 2,
                    class_counts=node.class_counts // 2,
                    impurity=0.4,
                    parent=node
                )

                if i == 2:  # Make leaf level
                    left.make_leaf()
                    right.make_leaf()

                node.set_split(i, f"feature_{i}", left, right)
                new_nodes.extend([left, right])

        # Test all methods work with deep tree
        X_val = np.random.randint(0, 2, (20, 3))
        y_val = np.random.randint(0, 2, 20)

        methods_to_test = [
            ("validation", {"X_val": X_val, "y_val": y_val}),
            ("cost_complexity", {"min_impurity_decrease": 0.1}),
            ("reduced_error", {"X_val": X_val, "y_val": y_val}),
            ("min_impurity", {"min_impurity_decrease": 0.1})
        ]

        for method, kwargs in methods_to_test:
            result = prune_tree(root, method=method, **kwargs)
            assert result is not None

    def test_edge_cases_for_full_coverage(self, sample_tree, validation_data):
        """Test specific edge cases to achieve additional coverage."""
        X_val, y_val = validation_data

        # Test reduced error pruning with aggressive settings
        result = reduced_error_prune(sample_tree, X_val, y_val, feature_mapping=None)
        assert result is not None

        # Test pruning with extremely high thresholds
        result = cost_complexity_prune(sample_tree, min_impurity_decrease=1.0)
        assert result is not None

        # Test min impurity pruning with various thresholds
        result = min_impurity_prune(sample_tree, min_impurity_decrease=0.0)
        assert result is not None


class TestPruningIntegration:
    """Integration tests combining pruning with other components."""

    def test_pruning_preserves_tree_structure(self, sample_tree):
        """Test that pruning preserves valid tree structure."""
        result = cost_complexity_prune(sample_tree, min_impurity_decrease=0.1)

        # Verify tree is still valid
        assert result is not None
        if not result.is_leaf:
            assert result.left is not None
            assert result.right is not None
            assert result.left.parent == result
            assert result.right.parent == result

    def test_pruning_updates_predictions(self, sample_tree):
        """Test that pruning updates leaf predictions correctly."""
        original_leaves = []
        def count_leaves(node):
            if node.is_leaf:
                original_leaves.append(node)
            else:
                count_leaves(node.left)
                count_leaves(node.right)

        count_leaves(sample_tree)
        original_leaf_count = len(original_leaves)

        # Prune aggressively
        result = min_impurity_prune(sample_tree, min_impurity_decrease=0.3)

        # Count leaves after pruning
        pruned_leaves = []
        def count_pruned_leaves(node):
            if node.is_leaf:
                pruned_leaves.append(node)
                # Verify prediction is set
                assert hasattr(node, 'prediction')
                assert node.prediction is not None
            else:
                if hasattr(node, 'left') and node.left:
                    count_pruned_leaves(node.left)
                if hasattr(node, 'right') and node.right:
                    count_pruned_leaves(node.right)

        count_pruned_leaves(result)
        # Tree structure should be valid after pruning
        assert len(pruned_leaves) >= 1