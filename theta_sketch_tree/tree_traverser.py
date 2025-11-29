"""
Tree traversal logic for theta sketch decision trees.

This module provides clean separation of prediction traversal logic
from the main classifier class.
"""

from typing import Union
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .tree_structure import TreeNode


class TreeTraverser:
    """
    Handles tree traversal for predictions with proper missing value handling.

    This class implements the standard decision tree traversal algorithm
    with a majority-vote strategy for handling missing values.
    """

    def __init__(self, tree_root: TreeNode):
        """
        Initialize traverser with a fitted tree.

        Parameters
        ----------
        tree_root : TreeNode
            Root node of the fitted decision tree
        """
        self.tree_root = tree_root

    def predict(self, X: NDArray) -> NDArray:
        """
        Predict class labels for binary feature data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Binary feature matrix (0/1 values).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        if len(X) == 0:
            return np.array([], dtype=np.int64)

        predictions = np.array([
            self.predict_single(sample)
            for sample in X
        ], dtype=np.int64)

        return predictions

    def predict_proba(self, X: NDArray) -> NDArray:
        """
        Predict class probabilities for binary feature data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Binary feature matrix (0/1 values).

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Class probabilities. For binary classification:
            - Column 0: P(class=0)
            - Column 1: P(class=1)
        """
        if len(X) == 0:
            return np.array([], dtype=np.float64).reshape(0, 2)

        probabilities = np.array([
            self.predict_proba_single(sample)
            for sample in X
        ], dtype=np.float64)

        return probabilities

    def predict_single(self, sample: NDArray) -> int:
        """
        Predict class label for a single sample.

        Parameters
        ----------
        sample : array-like of shape (n_features,)
            Single sample to predict

        Returns
        -------
        prediction : int
            Predicted class label (0 or 1)
        """
        leaf_node = self.traverse_to_leaf(sample)
        return leaf_node.prediction

    def predict_proba_single(self, sample: NDArray) -> NDArray[np.float64]:
        """
        Predict class probabilities for a single sample.

        Parameters
        ----------
        sample : array-like of shape (n_features,)
            Single sample to predict

        Returns
        -------
        probabilities : ndarray of shape (n_classes,)
            Class probabilities [P(class=0), P(class=1)]
        """
        leaf_node = self.traverse_to_leaf(sample)
        return leaf_node.probabilities

    def traverse_to_leaf(self, sample: NDArray, node: TreeNode = None) -> TreeNode:
        """
        Traverse tree to a leaf node using majority strategy for missing values.

        Parameters
        ----------
        sample : array-like of shape (n_features,)
            Sample to traverse with
        node : TreeNode, optional
            Current node (defaults to root)

        Returns
        -------
        leaf_node : TreeNode
            The leaf node reached by traversal
        """
        if node is None:
            node = self.tree_root

        # Base case: reached a leaf
        if node.is_leaf:
            return node

        # Get feature value for this node's split
        feature_value = sample[node.feature_idx]

        # Handle missing value with majority strategy
        if self._is_missing_value(feature_value):
            # Follow direction with more training samples
            left_samples = node.left.n_samples
            right_samples = node.right.n_samples

            if left_samples >= right_samples:
                return self.traverse_to_leaf(sample, node.left)
            else:
                return self.traverse_to_leaf(sample, node.right)

        # Standard traversal: False (0) -> left, True (1) -> right
        if feature_value:  # True or 1
            return self.traverse_to_leaf(sample, node.right)
        else:  # False or 0
            return self.traverse_to_leaf(sample, node.left)

    def _is_missing_value(self, value: Union[float, int, bool]) -> bool:
        """
        Check if a value is considered missing.

        Parameters
        ----------
        value : float, int, or bool
            Value to check

        Returns
        -------
        is_missing : bool
            True if value is considered missing
        """
        # Check for pandas NA
        if pd.isna(value):
            return True

        # Check for numpy NaN
        if isinstance(value, float) and np.isnan(value):
            return True

        return False


# Convenience functions for backward compatibility
def traverse_to_leaf(sample: NDArray, tree_root: TreeNode) -> TreeNode:
    """
    Convenience function to traverse to a leaf node.

    Parameters
    ----------
    sample : array-like of shape (n_features,)
        Sample to traverse with
    tree_root : TreeNode
        Root of the tree

    Returns
    -------
    leaf_node : TreeNode
        The leaf node reached by traversal
    """
    traverser = TreeTraverser(tree_root)
    return traverser.traverse_to_leaf(sample)


def predict_sample(sample: NDArray, tree_root: TreeNode) -> int:
    """
    Convenience function to predict a single sample.

    Parameters
    ----------
    sample : array-like of shape (n_features,)
        Sample to predict
    tree_root : TreeNode
        Root of the tree

    Returns
    -------
    prediction : int
        Predicted class label (0 or 1)
    """
    traverser = TreeTraverser(tree_root)
    return traverser.predict_single(sample)