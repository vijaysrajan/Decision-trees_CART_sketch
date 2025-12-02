"""
Comprehensive tests for tree_traverser module to achieve 100% coverage.

Tests all TreeTraverser functionality including prediction, probability estimation,
tree traversal with missing values, and convenience functions.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from theta_sketch_tree.tree_traverser import (
    TreeTraverser, traverse_to_leaf, predict_sample
)
from theta_sketch_tree.tree_structure import TreeNode


@pytest.fixture
def simple_leaf_tree():
    """Create a simple tree with just a leaf node."""
    root = TreeNode(
        depth=0,
        n_samples=100.0,
        class_counts=np.array([40, 60]),
        impurity=0.48
    )
    root.make_leaf()  # prediction=1, probabilities=[0.4, 0.6]
    return root


@pytest.fixture
def binary_tree():
    """Create a simple binary tree for testing."""
    # Root node (internal)
    root = TreeNode(
        depth=0,
        n_samples=100.0,
        class_counts=np.array([50, 50]),
        impurity=0.5
    )
    root.feature_idx = 0  # Split on feature 0

    # Left child (leaf)
    left = TreeNode(
        depth=1,
        n_samples=30.0,
        class_counts=np.array([25, 5]),
        impurity=0.278
    )
    left.make_leaf()  # prediction=0, probabilities=[0.833, 0.167]

    # Right child (leaf)
    right = TreeNode(
        depth=1,
        n_samples=70.0,
        class_counts=np.array([25, 45]),
        impurity=0.367
    )
    right.make_leaf()  # prediction=1, probabilities=[0.357, 0.643]

    root.left = left
    root.right = right

    return root


@pytest.fixture
def deeper_tree():
    """Create a deeper tree for testing more complex traversal."""
    # Root (splits on feature 0)
    root = TreeNode(depth=0, n_samples=100.0, class_counts=np.array([40, 60]), impurity=0.48)
    root.feature_idx = 0

    # Level 1 - Left branch (splits on feature 1)
    left_l1 = TreeNode(depth=1, n_samples=40.0, class_counts=np.array([30, 10]), impurity=0.375)
    left_l1.feature_idx = 1

    # Level 1 - Right branch (splits on feature 1)
    right_l1 = TreeNode(depth=1, n_samples=60.0, class_counts=np.array([10, 50]), impurity=0.278)
    right_l1.feature_idx = 1

    # Level 2 - Left-Left (leaf)
    ll_leaf = TreeNode(depth=2, n_samples=20.0, class_counts=np.array([18, 2]), impurity=0.18)
    ll_leaf.make_leaf()  # prediction=0

    # Level 2 - Left-Right (leaf)
    lr_leaf = TreeNode(depth=2, n_samples=20.0, class_counts=np.array([12, 8]), impurity=0.48)
    lr_leaf.make_leaf()  # prediction=0

    # Level 2 - Right-Left (leaf)
    rl_leaf = TreeNode(depth=2, n_samples=30.0, class_counts=np.array([8, 22]), impurity=0.391)
    rl_leaf.make_leaf()  # prediction=1

    # Level 2 - Right-Right (leaf)
    rr_leaf = TreeNode(depth=2, n_samples=30.0, class_counts=np.array([2, 28]), impurity=0.133)
    rr_leaf.make_leaf()  # prediction=1

    # Connect the tree
    root.left = left_l1
    root.right = right_l1
    left_l1.left = ll_leaf
    left_l1.right = lr_leaf
    right_l1.left = rl_leaf
    right_l1.right = rr_leaf

    return root


class TestTreeTraverserInitialization:
    """Test TreeTraverser initialization."""

    def test_init_with_tree(self, simple_leaf_tree):
        """Test initialization with a tree."""
        traverser = TreeTraverser(simple_leaf_tree)
        assert traverser.tree_root == simple_leaf_tree

    def test_init_with_none(self):
        """Test that initialization with None works."""
        traverser = TreeTraverser(None)
        assert traverser.tree_root is None


class TestPredict:
    """Test predict method for class prediction."""

    def test_predict_empty_array(self, simple_leaf_tree):
        """Test predict with empty input array."""
        traverser = TreeTraverser(simple_leaf_tree)
        X = np.array([]).reshape(0, 2)
        result = traverser.predict(X)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64
        assert len(result) == 0

    def test_predict_single_sample_leaf_tree(self, simple_leaf_tree):
        """Test predict with single sample on leaf-only tree."""
        traverser = TreeTraverser(simple_leaf_tree)
        X = np.array([[1, 0]])
        result = traverser.predict(X)

        assert result.shape == (1,)
        assert result[0] == 1  # Leaf prediction

    def test_predict_multiple_samples(self, binary_tree):
        """Test predict with multiple samples."""
        traverser = TreeTraverser(binary_tree)
        X = np.array([
            [0, 1],  # Left branch -> prediction 0
            [1, 0],  # Right branch -> prediction 1
            [0, 0],  # Left branch -> prediction 0
            [1, 1]   # Right branch -> prediction 1
        ])
        result = traverser.predict(X)

        assert result.shape == (4,)
        assert result.dtype == np.int64
        expected = np.array([0, 1, 0, 1])
        np.testing.assert_array_equal(result, expected)

    def test_predict_deeper_tree(self, deeper_tree):
        """Test predict with deeper tree structure."""
        traverser = TreeTraverser(deeper_tree)
        X = np.array([
            [0, 0],  # Left-Left -> prediction 0
            [0, 1],  # Left-Right -> prediction 0
            [1, 0],  # Right-Left -> prediction 1
            [1, 1]   # Right-Right -> prediction 1
        ])
        result = traverser.predict(X)

        assert result.shape == (4,)
        expected = np.array([0, 0, 1, 1])
        np.testing.assert_array_equal(result, expected)


class TestPredictProba:
    """Test predict_proba method for probability estimation."""

    def test_predict_proba_empty_array(self, simple_leaf_tree):
        """Test predict_proba with empty input array."""
        traverser = TreeTraverser(simple_leaf_tree)
        X = np.array([]).reshape(0, 2)
        result = traverser.predict_proba(X)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result.shape == (0, 2)

    def test_predict_proba_single_sample(self, simple_leaf_tree):
        """Test predict_proba with single sample."""
        traverser = TreeTraverser(simple_leaf_tree)
        X = np.array([[1, 0]])
        result = traverser.predict_proba(X)

        assert result.shape == (1, 2)
        assert result.dtype == np.float64
        expected_proba = np.array([0.4, 0.6])  # From class_counts [40, 60]
        np.testing.assert_array_almost_equal(result[0], expected_proba, decimal=3)

    def test_predict_proba_multiple_samples(self, binary_tree):
        """Test predict_proba with multiple samples."""
        traverser = TreeTraverser(binary_tree)
        X = np.array([
            [0, 1],  # Left branch
            [1, 0]   # Right branch
        ])
        result = traverser.predict_proba(X)

        assert result.shape == (2, 2)
        assert result.dtype == np.float64

        # Left branch: [25, 5] -> [0.833, 0.167]
        # Right branch: [25, 45] -> [0.357, 0.643]
        expected_left = np.array([0.833, 0.167])
        expected_right = np.array([0.357, 0.643])

        np.testing.assert_array_almost_equal(result[0], expected_left, decimal=2)
        np.testing.assert_array_almost_equal(result[1], expected_right, decimal=2)


class TestPredictSingle:
    """Test predict_single method."""

    def test_predict_single_leaf_tree(self, simple_leaf_tree):
        """Test predict_single on leaf-only tree."""
        traverser = TreeTraverser(simple_leaf_tree)
        sample = np.array([1, 0])
        result = traverser.predict_single(sample)

        assert isinstance(result, (int, np.integer))
        assert result == 1

    def test_predict_single_binary_tree(self, binary_tree):
        """Test predict_single on binary tree."""
        traverser = TreeTraverser(binary_tree)

        # Test left branch
        sample_left = np.array([0, 1])
        result_left = traverser.predict_single(sample_left)
        assert result_left == 0

        # Test right branch
        sample_right = np.array([1, 0])
        result_right = traverser.predict_single(sample_right)
        assert result_right == 1


class TestPredictProbaSingle:
    """Test predict_proba_single method."""

    def test_predict_proba_single_leaf_tree(self, simple_leaf_tree):
        """Test predict_proba_single on leaf-only tree."""
        traverser = TreeTraverser(simple_leaf_tree)
        sample = np.array([1, 0])
        result = traverser.predict_proba_single(sample)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result.shape == (2,)
        expected = np.array([0.4, 0.6])
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_predict_proba_single_binary_tree(self, binary_tree):
        """Test predict_proba_single on binary tree."""
        traverser = TreeTraverser(binary_tree)

        # Test left branch
        sample_left = np.array([0, 1])
        result_left = traverser.predict_proba_single(sample_left)
        expected_left = np.array([0.833, 0.167])
        np.testing.assert_array_almost_equal(result_left, expected_left, decimal=2)

        # Test right branch
        sample_right = np.array([1, 0])
        result_right = traverser.predict_proba_single(sample_right)
        expected_right = np.array([0.357, 0.643])
        np.testing.assert_array_almost_equal(result_right, expected_right, decimal=2)


class TestTraverseToLeaf:
    """Test traverse_to_leaf method including missing value handling."""

    def test_traverse_to_leaf_simple(self, simple_leaf_tree):
        """Test traverse_to_leaf on simple leaf tree."""
        traverser = TreeTraverser(simple_leaf_tree)
        sample = np.array([1, 0])
        result = traverser.traverse_to_leaf(sample)

        assert result == simple_leaf_tree
        assert result.is_leaf

    def test_traverse_to_leaf_with_specific_node(self, binary_tree):
        """Test traverse_to_leaf with specific starting node."""
        traverser = TreeTraverser(binary_tree)
        sample = np.array([1, 0])

        # Start from left child directly
        result = traverser.traverse_to_leaf(sample, binary_tree.left)
        assert result == binary_tree.left
        assert result.is_leaf

    def test_traverse_to_leaf_binary_tree(self, binary_tree):
        """Test traverse_to_leaf on binary tree."""
        traverser = TreeTraverser(binary_tree)

        # Test left path (feature 0 = 0)
        sample_left = np.array([0, 1])
        result_left = traverser.traverse_to_leaf(sample_left)
        assert result_left == binary_tree.left

        # Test right path (feature 0 = 1)
        sample_right = np.array([1, 0])
        result_right = traverser.traverse_to_leaf(sample_right)
        assert result_right == binary_tree.right

    def test_traverse_missing_value_left_majority(self, binary_tree):
        """Test missing value handling when left branch has more samples."""
        # Modify tree so left has more samples
        binary_tree.left.n_samples = 80.0
        binary_tree.right.n_samples = 20.0

        traverser = TreeTraverser(binary_tree)

        # Test with NaN (missing value)
        sample_nan = np.array([np.nan, 0])
        result = traverser.traverse_to_leaf(sample_nan)
        assert result == binary_tree.left  # Should follow majority (left)

        # Test with pd.NA (pandas missing value)
        sample_pd_na = np.array([pd.NA, 0])
        result_pd = traverser.traverse_to_leaf(sample_pd_na)
        assert result_pd == binary_tree.left

    def test_traverse_missing_value_right_majority(self, binary_tree):
        """Test missing value handling when right branch has more samples."""
        # Modify tree so right has more samples
        binary_tree.left.n_samples = 20.0
        binary_tree.right.n_samples = 80.0

        traverser = TreeTraverser(binary_tree)

        # Test with NaN
        sample_nan = np.array([np.nan, 0])
        result = traverser.traverse_to_leaf(sample_nan)
        assert result == binary_tree.right  # Should follow majority (right)

    def test_traverse_missing_value_equal_samples(self, binary_tree):
        """Test missing value handling when left and right have equal samples."""
        # Equal samples (left >= right, so should choose left)
        binary_tree.left.n_samples = 50.0
        binary_tree.right.n_samples = 50.0

        traverser = TreeTraverser(binary_tree)

        sample_nan = np.array([np.nan, 0])
        result = traverser.traverse_to_leaf(sample_nan)
        assert result == binary_tree.left  # Should choose left when equal

    def test_traverse_deeper_tree_with_missing_values(self, deeper_tree):
        """Test missing value handling in deeper tree."""
        traverser = TreeTraverser(deeper_tree)

        # Missing value at root (feature 0 = NaN)
        # Right branch has more samples (60 vs 40)
        sample = np.array([np.nan, 1])
        result = traverser.traverse_to_leaf(sample)

        # Should go right at root, then follow feature 1 = 1 to right-right leaf
        assert result == deeper_tree.right.right
        assert result.is_leaf


class TestIsMissingValue:
    """Test _is_missing_value method."""

    def test_is_missing_value_pandas_na(self, simple_leaf_tree):
        """Test detection of pandas NA values."""
        traverser = TreeTraverser(simple_leaf_tree)

        assert traverser._is_missing_value(pd.NA) is True
        assert traverser._is_missing_value(pd.NaT) is True

    def test_is_missing_value_numpy_nan(self, simple_leaf_tree):
        """Test detection of numpy NaN values."""
        traverser = TreeTraverser(simple_leaf_tree)

        # Test np.nan (which is float)
        assert traverser._is_missing_value(np.nan) is True

        # Test explicit float NaN
        float_nan = float('nan')
        assert isinstance(float_nan, float)  # Ensure it's float type
        assert traverser._is_missing_value(float_nan) is True

        # Test numpy float64 NaN
        np_float_nan = np.float64('nan')
        assert traverser._is_missing_value(np_float_nan) is True

    def test_is_missing_value_numpy_nan_bypass_pandas(self, simple_leaf_tree):
        """Test numpy NaN detection bypassing pandas check to cover line 182."""
        from unittest.mock import patch

        traverser = TreeTraverser(simple_leaf_tree)
        float_nan = float('nan')

        # Patch pd.isna to return False so we reach the numpy check
        with patch('pandas.isna', return_value=False):
            result = traverser._is_missing_value(float_nan)
            assert result is True  # Should be caught by numpy check on line 182

    def test_is_missing_value_valid_values(self, simple_leaf_tree):
        """Test that valid values are not considered missing."""
        traverser = TreeTraverser(simple_leaf_tree)

        assert traverser._is_missing_value(0) is False
        assert traverser._is_missing_value(1) is False
        assert traverser._is_missing_value(0.0) is False
        assert traverser._is_missing_value(1.0) is False
        assert traverser._is_missing_value(True) is False
        assert traverser._is_missing_value(False) is False
        assert traverser._is_missing_value(-1) is False
        assert traverser._is_missing_value(42.5) is False

    def test_is_missing_value_edge_cases(self, simple_leaf_tree):
        """Test edge cases for missing value detection."""
        traverser = TreeTraverser(simple_leaf_tree)

        # Zero and negative zero
        assert traverser._is_missing_value(0.0) is False
        assert traverser._is_missing_value(-0.0) is False

        # Infinite values are not missing
        assert traverser._is_missing_value(float('inf')) is False
        assert traverser._is_missing_value(float('-inf')) is False

        # Large numbers are not missing
        assert traverser._is_missing_value(1e100) is False
        assert traverser._is_missing_value(-1e100) is False


class TestConvenienceFunctions:
    """Test convenience functions for backward compatibility."""

    def test_traverse_to_leaf_function(self, binary_tree):
        """Test traverse_to_leaf convenience function."""
        sample = np.array([0, 1])
        result = traverse_to_leaf(sample, binary_tree)

        assert result == binary_tree.left
        assert result.is_leaf

    def test_traverse_to_leaf_function_right_path(self, binary_tree):
        """Test traverse_to_leaf function with right path."""
        sample = np.array([1, 0])
        result = traverse_to_leaf(sample, binary_tree)

        assert result == binary_tree.right
        assert result.is_leaf

    def test_predict_sample_function(self, binary_tree):
        """Test predict_sample convenience function."""
        # Left path
        sample_left = np.array([0, 1])
        result_left = predict_sample(sample_left, binary_tree)
        assert result_left == 0

        # Right path
        sample_right = np.array([1, 0])
        result_right = predict_sample(sample_right, binary_tree)
        assert result_right == 1

    def test_convenience_functions_with_missing_values(self, binary_tree):
        """Test convenience functions with missing values."""
        # Modify for left majority
        binary_tree.left.n_samples = 80.0
        binary_tree.right.n_samples = 20.0

        sample_nan = np.array([np.nan, 0])

        # Test traverse_to_leaf function
        leaf = traverse_to_leaf(sample_nan, binary_tree)
        assert leaf == binary_tree.left

        # Test predict_sample function
        prediction = predict_sample(sample_nan, binary_tree)
        assert prediction == binary_tree.left.prediction


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_traverser_with_none_root(self):
        """Test traverser behavior with None root."""
        traverser = TreeTraverser(None)
        assert traverser.tree_root is None

        # The traverser with None root would fail if used, which is expected behavior
        # This test just verifies initialization works with None

    def test_predict_with_various_dtypes(self, binary_tree):
        """Test predict with various input data types."""
        traverser = TreeTraverser(binary_tree)

        # Integer input
        X_int = np.array([[0, 1], [1, 0]], dtype=np.int32)
        result_int = traverser.predict(X_int)
        expected = np.array([0, 1])
        np.testing.assert_array_equal(result_int, expected)

        # Float input
        X_float = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        result_float = traverser.predict(X_float)
        np.testing.assert_array_equal(result_float, expected)

        # Boolean input
        X_bool = np.array([[False, True], [True, False]], dtype=bool)
        result_bool = traverser.predict(X_bool)
        np.testing.assert_array_equal(result_bool, expected)

    def test_predict_proba_with_various_dtypes(self, binary_tree):
        """Test predict_proba with various input data types."""
        traverser = TreeTraverser(binary_tree)

        X_int = np.array([[0, 1]], dtype=np.int32)
        result_int = traverser.predict_proba(X_int)

        X_float = np.array([[0.0, 1.0]], dtype=np.float64)
        result_float = traverser.predict_proba(X_float)

        # Results should be the same regardless of input dtype
        np.testing.assert_array_equal(result_int, result_float)

        assert result_int.shape == (1, 2)
        assert result_int.dtype == np.float64


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_mixed_missing_and_valid_values(self, deeper_tree):
        """Test prediction with mix of missing and valid values in same batch."""
        traverser = TreeTraverser(deeper_tree)

        X = np.array([
            [0, 1],        # Valid path: Left-Right
            [np.nan, 1],   # Missing first feature, valid second
            [1, np.nan],   # Valid first, missing second
            [1, 0]         # Valid path: Right-Left
        ])

        result = traverser.predict(X)
        assert len(result) == 4
        assert result.dtype == np.int64

        # All should be valid predictions
        assert all(pred in [0, 1] for pred in result)

    def test_all_missing_values_single_sample(self, binary_tree):
        """Test sample with all missing values."""
        # Set left as majority
        binary_tree.left.n_samples = 80.0
        binary_tree.right.n_samples = 20.0

        traverser = TreeTraverser(binary_tree)
        sample = np.array([np.nan, np.nan])

        result = traverser.predict_single(sample)
        assert result == binary_tree.left.prediction  # Should follow majority

    def test_large_batch_prediction(self, binary_tree):
        """Test prediction on larger batch."""
        traverser = TreeTraverser(binary_tree)

        # Create large batch with mixed paths
        np.random.seed(42)
        X = np.random.randint(0, 2, size=(1000, 2))

        result = traverser.predict(X)
        assert result.shape == (1000,)
        assert result.dtype == np.int64
        assert all(pred in [0, 1] for pred in result)

        # Verify results match single predictions
        for i in range(10):  # Test first 10 to avoid long test
            single_result = traverser.predict_single(X[i])
            assert result[i] == single_result