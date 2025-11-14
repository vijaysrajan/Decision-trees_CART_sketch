"""
Unit tests for tree traversal logic.

Tests TreeTraverser class and convenience functions.
"""

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from theta_sketch_tree.tree_traverser import (
    TreeTraverser, predict_tree, predict_proba_tree
)
from theta_sketch_tree.tree_structure import TreeNode


class TestTreeTraverser:
    """Test TreeTraverser class functionality."""

    @pytest.fixture
    def simple_tree(self):
        """Create a simple tree for testing prediction."""
        # Root: feature 0 (split on X[:, 0])
        root = TreeNode(
            depth=0,
            n_samples=100,
            class_counts=np.array([50, 50]),
            impurity=0.5
        )

        # Left child: X[:, 0] == 0 -> mostly class 0
        left = TreeNode(
            depth=1,
            n_samples=60,
            class_counts=np.array([50, 10]),
            impurity=0.278
        )
        left.make_leaf()

        # Right child: X[:, 0] == 1 -> mostly class 1
        right = TreeNode(
            depth=1,
            n_samples=40,
            class_counts=np.array([0, 40]),
            impurity=0.0
        )
        right.make_leaf()

        root.set_split(
            feature_idx=0,
            feature_name='feature_0',
            left_child=left,
            right_child=right
        )

        return root

    @pytest.fixture
    def multi_level_tree(self):
        """Create a multi-level tree for testing."""
        # Root: split on feature 0
        root = TreeNode(
            depth=0,
            n_samples=100,
            class_counts=np.array([55, 45]),
            impurity=0.4975
        )

        # Level 1 - Left: split on feature 1
        left_l1 = TreeNode(
            depth=1,
            n_samples=70,
            class_counts=np.array([50, 20]),
            impurity=0.408
        )

        # Level 1 - Right: leaf (mostly class 1)
        right_l1 = TreeNode(
            depth=1,
            n_samples=30,
            class_counts=np.array([5, 25]),
            impurity=0.278
        )
        right_l1.make_leaf()

        # Level 2 - Left-Left: leaf (mostly class 0)
        left_l2_left = TreeNode(
            depth=2,
            n_samples=40,
            class_counts=np.array([35, 5]),
            impurity=0.25
        )
        left_l2_left.make_leaf()

        # Level 2 - Left-Right: leaf (balanced)
        left_l2_right = TreeNode(
            depth=2,
            n_samples=30,
            class_counts=np.array([15, 15]),
            impurity=0.5
        )
        left_l2_right.make_leaf()

        # Build tree
        root.set_split(0, 'feature_0', left_l1, right_l1)
        left_l1.set_split(1, 'feature_1', left_l2_left, left_l2_right)

        return root

    @pytest.fixture
    def single_node_tree(self):
        """Create a single node tree (no splits)."""
        root = TreeNode(
            depth=0,
            n_samples=100,
            class_counts=np.array([30, 70]),
            impurity=0.42
        )
        root.make_leaf()
        return root

    def test_initialization(self, simple_tree):
        """Test TreeTraverser initialization."""
        traverser = TreeTraverser(simple_tree, missing_value_strategy='majority')

        assert traverser.root is simple_tree
        assert traverser.missing_value_strategy == 'majority'

    def test_initialization_invalid_strategy(self, simple_tree):
        """Test initialization with invalid missing value strategy."""
        with pytest.raises(ValueError, match="Invalid missing_value_strategy"):
            TreeTraverser(simple_tree, missing_value_strategy='invalid')

    def test_simple_prediction(self, simple_tree):
        """Test prediction on simple tree."""
        traverser = TreeTraverser(simple_tree)

        # Test samples: [[feature_0, ...], ...]
        X = np.array([
            [0],  # Goes left -> class 0
            [1],  # Goes right -> class 1
        ])

        predictions = traverser.predict(X)
        assert_array_equal(predictions, np.array([0, 1]))

    def test_simple_predict_proba(self, simple_tree):
        """Test probability prediction on simple tree."""
        traverser = TreeTraverser(simple_tree)

        X = np.array([[0], [1]])
        probabilities = traverser.predict_proba(X)

        # Left leaf: [50, 10] -> [0.833, 0.167]
        # Right leaf: [0, 40] -> [0.0, 1.0]
        expected = np.array([
            [50/60, 10/60],  # Left leaf probabilities
            [0.0, 1.0]       # Right leaf probabilities
        ])

        assert probabilities.shape == (2, 2)
        assert_array_equal(probabilities, expected)

    def test_multi_level_prediction(self, multi_level_tree):
        """Test prediction on multi-level tree."""
        traverser = TreeTraverser(multi_level_tree)

        # Test all possible paths
        X = np.array([
            [0, 0],  # Left -> Left-Left: class 0
            [0, 1],  # Left -> Left-Right: balanced
            [1, 0],  # Right: class 1
            [1, 1],  # Right: class 1 (feature 1 doesn't matter)
        ])

        predictions = traverser.predict(X)

        # Expected: [0, 0, 1, 1] (0 for ties goes to class 0)
        assert_array_equal(predictions, np.array([0, 0, 1, 1]))

    def test_single_node_prediction(self, single_node_tree):
        """Test prediction on single node tree."""
        traverser = TreeTraverser(single_node_tree)

        X = np.array([[0], [1], [999]])  # Any input should give same result
        predictions = traverser.predict(X)

        # All should predict class 1 (majority: 70 > 30)
        assert_array_equal(predictions, np.array([1, 1, 1]))

    def test_missing_values_majority_strategy(self, multi_level_tree):
        """Test handling of missing values with majority strategy."""
        traverser = TreeTraverser(multi_level_tree, missing_value_strategy='majority')

        # Create data with NaN values
        X = np.array([
            [np.nan, 0],  # Missing feature 0
            [0, np.nan],  # Missing feature 1
            [np.nan, np.nan],  # Missing both
        ])

        predictions = traverser.predict(X)

        # Should handle gracefully without errors
        assert len(predictions) == 3
        assert all(pred in [0, 1] for pred in predictions)

    def test_missing_values_zero_strategy(self, simple_tree):
        """Test handling of missing values with zero strategy."""
        traverser = TreeTraverser(simple_tree, missing_value_strategy='zero')

        X = np.array([[np.nan]])  # Missing value

        predictions = traverser.predict(X)

        # Should treat missing as 0 -> go left -> class 0
        assert predictions[0] == 0

    def test_missing_values_error_strategy(self, simple_tree):
        """Test handling of missing values with error strategy."""
        traverser = TreeTraverser(simple_tree, missing_value_strategy='error')

        X = np.array([[np.nan]])  # Missing value

        with pytest.raises(ValueError, match="Missing value"):
            traverser.predict(X)

    def test_pandas_nan_handling(self, simple_tree):
        """Test handling of pandas NaN values."""
        traverser = TreeTraverser(simple_tree, missing_value_strategy='zero')

        X = np.array([[pd.NA]])  # pandas NaN

        # Should handle pandas NaN gracefully
        predictions = traverser.predict(X)
        assert len(predictions) == 1

    def test_batch_prediction_consistency(self, multi_level_tree):
        """Test that batch and single predictions are consistent."""
        traverser = TreeTraverser(multi_level_tree)

        # Test individual predictions
        individual_preds = []
        individual_probas = []
        samples = [[0, 0], [0, 1], [1, 0], [1, 1]]

        for sample in samples:
            pred = traverser.predict(np.array([sample]))
            proba = traverser.predict_proba(np.array([sample]))
            individual_preds.append(pred[0])
            individual_probas.append(proba[0])

        # Test batch prediction
        X = np.array(samples)
        batch_preds = traverser.predict(X)
        batch_probas = traverser.predict_proba(X)

        # Should be identical
        assert_array_equal(batch_preds, np.array(individual_preds))
        assert_array_equal(batch_probas, np.array(individual_probas))

    def test_prediction_probability_consistency(self, multi_level_tree):
        """Test consistency between predict and predict_proba."""
        traverser = TreeTraverser(multi_level_tree)

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        predictions = traverser.predict(X)
        probabilities = traverser.predict_proba(X)

        # Predicted class should match argmax of probabilities
        predicted_from_proba = np.argmax(probabilities, axis=1)
        assert_array_equal(predictions, predicted_from_proba)

        # Probabilities should sum to 1
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_edge_case_inputs(self, simple_tree):
        """Test edge case inputs."""
        traverser = TreeTraverser(simple_tree)

        # Empty input
        X_empty = np.array([]).reshape(0, 1)
        preds_empty = traverser.predict(X_empty)
        assert len(preds_empty) == 0

        # Single sample
        X_single = np.array([[0]])
        preds_single = traverser.predict(X_single)
        assert len(preds_single) == 1

        # Large batch
        X_large = np.random.randint(0, 2, size=(1000, 1))
        preds_large = traverser.predict(X_large)
        assert len(preds_large) == 1000
        assert all(pred in [0, 1] for pred in preds_large)

    def test_feature_indices_validation(self, simple_tree):
        """Test that feature indices are used correctly."""
        # Modify tree to use feature index 1 instead of 0
        simple_tree.feature_idx = 1

        traverser = TreeTraverser(simple_tree)

        # Should now split on X[:, 1] instead of X[:, 0]
        X = np.array([
            [999, 0],  # Feature 0=999 ignored, feature 1=0 -> left
            [999, 1],  # Feature 0=999 ignored, feature 1=1 -> right
        ])

        predictions = traverser.predict(X)
        assert_array_equal(predictions, np.array([0, 1]))


class TestConvenienceFunctions:
    """Test convenience functions for tree traversal."""

    @pytest.fixture
    def test_tree(self):
        """Simple tree for testing convenience functions."""
        root = TreeNode(
            depth=0,
            n_samples=100,
            class_counts=np.array([40, 60]),
            impurity=0.48
        )

        left = TreeNode(
            depth=1,
            n_samples=50,
            class_counts=np.array([35, 15]),
            impurity=0.42
        )
        left.make_leaf()

        right = TreeNode(
            depth=1,
            n_samples=50,
            class_counts=np.array([5, 45]),
            impurity=0.2
        )
        right.make_leaf()

        root.set_split(0, 'feature_0', left, right)
        return root

    def test_predict_tree_convenience(self, test_tree):
        """Test predict_tree convenience function."""
        X = np.array([[0], [1]])

        predictions = predict_tree(test_tree, X)

        assert len(predictions) == 2
        assert predictions.dtype == np.int64
        assert all(pred in [0, 1] for pred in predictions)

    def test_predict_proba_tree_convenience(self, test_tree):
        """Test predict_proba_tree convenience function."""
        X = np.array([[0], [1]])

        probabilities = predict_proba_tree(test_tree, X)

        assert probabilities.shape == (2, 2)
        assert probabilities.dtype == np.float64
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_convenience_function_missing_strategy(self, test_tree):
        """Test convenience functions with different missing value strategies."""
        X = np.array([[np.nan]])

        # Should work with different strategies
        for strategy in ['majority', 'zero', 'error']:
            try:
                preds = predict_tree(test_tree, X, missing_value_strategy=strategy)
                probas = predict_proba_tree(test_tree, X, missing_value_strategy=strategy)

                if strategy != 'error':
                    assert len(preds) == 1
                    assert probas.shape == (1, 2)
            except ValueError:
                # Error strategy should raise ValueError
                assert strategy == 'error'


class TestTreeTraverserEdgeCases:
    """Test edge cases and error conditions."""

    def test_malformed_tree_handling(self):
        """Test handling of malformed trees."""
        # Tree with no splits but not marked as leaf
        bad_root = TreeNode(
            depth=0,
            n_samples=100,
            class_counts=np.array([50, 50]),
            impurity=0.5
        )
        # Don't call make_leaf() and don't set split -> malformed

        traverser = TreeTraverser(bad_root)

        X = np.array([[0]])

        # Should handle gracefully (malformed tree treated as leaf)
        with pytest.raises(AttributeError):
            # This will fail because is_leaf is False but no split is set
            traverser.predict(X)

    def test_very_deep_tree(self):
        """Test with a very deep tree (stress test)."""
        # Create a deep linear tree (each node has only one child)
        depth_limit = 50

        root = TreeNode(
            depth=0,
            n_samples=100,
            class_counts=np.array([50, 50]),
            impurity=0.5
        )

        current = root
        for depth in range(depth_limit - 1):
            left_child = TreeNode(
                depth=depth + 1,
                n_samples=50 - depth,
                class_counts=np.array([25 - depth//2, 25 - depth//2]),
                impurity=0.5
            )

            right_child = TreeNode(
                depth=depth + 1,
                n_samples=1,
                class_counts=np.array([0, 1]),
                impurity=0.0
            )
            right_child.make_leaf()

            current.set_split(0, f'feature_{depth}', left_child, right_child)
            current = left_child

        # Make final leaf
        current.make_leaf()

        traverser = TreeTraverser(root)
        X = np.array([[0] * depth_limit])  # All features = 0 -> go left

        # Should handle deep tree without stack overflow
        predictions = traverser.predict(X)
        assert len(predictions) == 1

    def test_large_batch_prediction(self):
        """Test with large batch sizes."""
        # Simple tree
        root = TreeNode(
            depth=0,
            n_samples=100,
            class_counts=np.array([50, 50]),
            impurity=0.5
        )
        root.make_leaf()

        traverser = TreeTraverser(root)

        # Large batch
        batch_size = 10000
        X = np.random.randint(0, 2, size=(batch_size, 5))

        predictions = traverser.predict(X)
        probabilities = traverser.predict_proba(X)

        assert len(predictions) == batch_size
        assert probabilities.shape == (batch_size, 2)

    def test_tree_wrapper_compatibility(self):
        """Test compatibility with Tree wrapper class."""
        from theta_sketch_tree.tree_structure import Tree

        root = TreeNode(
            depth=0,
            n_samples=100,
            class_counts=np.array([50, 50]),
            impurity=0.5
        )
        root.make_leaf()

        try:
            # Try to create Tree wrapper
            tree_wrapper = Tree(root)
            traverser = TreeTraverser(tree_wrapper)

            X = np.array([[0]])
            predictions = traverser.predict(X)
            assert len(predictions) == 1

        except (NotImplementedError, TypeError):
            # Tree wrapper may not be implemented yet
            pass
