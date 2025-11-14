"""
Tree traverser module.

This module implements tree navigation for making predictions on binary feature data.
"""

from typing import Union
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from .tree_structure import TreeNode, Tree


class TreeTraverser:
    """
    Navigate decision tree for making predictions.

    This class handles tree traversal during inference, including
    missing value handling strategies.

    Parameters
    ----------
    root : TreeNode or Tree
        The decision tree root node or Tree object
    missing_value_strategy : str, default='majority'
        How to handle missing values:
        - 'majority': Follow direction with most training samples
        - 'zero': Treat missing as False
        - 'error': Raise error on missing values

    Attributes
    ----------
    root : TreeNode
        Root node of the decision tree
    missing_value_strategy : str
        Strategy for handling missing values
    """

    def __init__(
        self,
        root: Union[TreeNode, Tree],
        missing_value_strategy: str = 'majority'
    ) -> None:
        """
        Initialize tree traverser.

        Parameters
        ----------
        root : TreeNode or Tree
            Root node or Tree wrapper
        missing_value_strategy : str
            Strategy for missing values ('majority', 'zero', 'error')
        """
        # Handle both TreeNode and Tree objects
        if isinstance(root, TreeNode):
            self.root = root
        elif hasattr(root, 'root'):
            self.root = root.root
        else:
            raise TypeError("root must be TreeNode or Tree object")

        if missing_value_strategy not in ['majority', 'zero', 'error']:
            raise ValueError(
                f"Invalid missing_value_strategy: '{missing_value_strategy}'. "
                f"Valid options: 'majority', 'zero', 'error'"
            )
        self.missing_value_strategy = missing_value_strategy

    def predict(
        self,
        X: NDArray
    ) -> NDArray[np.int64]:
        """
        Predict class labels for binary feature matrix.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Binary features (0/1, True/False, possibly with NaN for missing)

        Returns
        -------
        predictions : array of shape (n_samples,)
            Predicted class labels (0 or 1)

        Examples
        --------
        >>> traverser = TreeTraverser(root_node)
        >>> X_test = np.array([[1, 0, 1], [0, 1, 0]])
        >>> predictions = traverser.predict(X_test)
        >>> print(predictions)  # [1, 0]
        """
        if len(X) == 0:
            return np.array([], dtype=np.int64)

        predictions = np.array([
            self._predict_single(sample)
            for sample in X
        ], dtype=np.int64)
        return predictions

    def predict_proba(
        self,
        X: NDArray
    ) -> NDArray[np.float64]:
        """
        Predict class probabilities for binary feature matrix.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Binary features (0/1, True/False, possibly with NaN for missing)

        Returns
        -------
        probabilities : array of shape (n_samples, n_classes)
            Class probabilities [[P(class=0), P(class=1)], ...]

        Examples
        --------
        >>> traverser = TreeTraverser(root_node)
        >>> X_test = np.array([[1, 0, 1], [0, 1, 0]])
        >>> proba = traverser.predict_proba(X_test)
        >>> print(proba)  # [[0.2, 0.8], [0.9, 0.1]]
        """
        if len(X) == 0:
            return np.array([], dtype=np.float64).reshape(0, 2)

        probabilities = np.array([
            self._predict_proba_single(sample)
            for sample in X
        ], dtype=np.float64)
        return probabilities

    def _predict_single(self, sample: NDArray) -> int:
        """
        Predict class label for a single sample.

        Parameters
        ----------
        sample : array of shape (n_features,)
            Binary feature values for one sample

        Returns
        -------
        prediction : int
            Predicted class label (0 or 1)
        """
        node = self._traverse_to_leaf(sample, self.root)
        return node.prediction

    def _predict_proba_single(self, sample: NDArray) -> NDArray[np.float64]:
        """
        Predict class probabilities for a single sample.

        Parameters
        ----------
        sample : array of shape (n_features,)
            Binary feature values for one sample

        Returns
        -------
        probabilities : array of shape (n_classes,)
            Class probabilities [P(class=0), P(class=1)]
        """
        node = self._traverse_to_leaf(sample, self.root)
        return node.probabilities

    def _traverse_to_leaf(
        self,
        sample: NDArray,
        node: TreeNode
    ) -> TreeNode:
        """
        Recursively traverse tree to a leaf node.

        Handles missing values according to the configured strategy.

        Parameters
        ----------
        sample : array of shape (n_features,)
            Binary feature values
        node : TreeNode
            Current node in traversal

        Returns
        -------
        leaf_node : TreeNode
            The leaf node reached by traversal

        Raises
        ------
        ValueError
            If missing value encountered and strategy is 'error'
        """
        # Base case: reached a leaf
        if node.is_leaf:
            return node

        # Get feature value for this node's split
        feature_value = sample[node.feature_idx]

        # Handle missing value
        if pd.isna(feature_value) or (isinstance(feature_value, float) and np.isnan(feature_value)):
            if self.missing_value_strategy == 'error':
                raise ValueError(
                    f"Missing value at feature index {node.feature_idx} "
                    f"(feature name: {node.feature_name})"
                )
            elif self.missing_value_strategy == 'zero':
                feature_value = False  # Treat missing as False (absent)
            elif self.missing_value_strategy == 'majority':
                # Follow direction with more training samples
                # Compare left and right child sample counts
                left_samples = node.left.n_samples
                right_samples = node.right.n_samples

                if left_samples >= right_samples:
                    return self._traverse_to_leaf(sample, node.left)
                else:
                    return self._traverse_to_leaf(sample, node.right)

        # Standard traversal based on feature value
        # Convention: False (0) -> left, True (1) -> right
        if feature_value:  # True or 1
            return self._traverse_to_leaf(sample, node.right)
        else:  # False or 0
            return self._traverse_to_leaf(sample, node.left)


def predict_tree(
    root: Union[TreeNode, Tree],
    X: NDArray,
    missing_value_strategy: str = 'majority'
) -> NDArray[np.int64]:
    """
    Convenience function to predict without creating TreeTraverser instance.

    Parameters
    ----------
    root : TreeNode or Tree
        Root of decision tree
    X : array of shape (n_samples, n_features)
        Binary features
    missing_value_strategy : str, default='majority'
        How to handle missing values

    Returns
    -------
    predictions : array of shape (n_samples,)
        Predicted class labels

    Examples
    --------
    >>> predictions = predict_tree(root_node, X_test)
    """
    traverser = TreeTraverser(root, missing_value_strategy)
    return traverser.predict(X)


def predict_proba_tree(
    root: Union[TreeNode, Tree],
    X: NDArray,
    missing_value_strategy: str = 'majority'
) -> NDArray[np.float64]:
    """
    Convenience function to predict probabilities without creating TreeTraverser instance.

    Parameters
    ----------
    root : TreeNode or Tree
        Root of decision tree
    X : array of shape (n_samples, n_features)
        Binary features
    missing_value_strategy : str, default='majority'
        How to handle missing values

    Returns
    -------
    probabilities : array of shape (n_samples, n_classes)
        Class probabilities

    Examples
    --------
    >>> proba = predict_proba_tree(root_node, X_test)
    """
    traverser = TreeTraverser(root, missing_value_strategy)
    return traverser.predict_proba(X)
