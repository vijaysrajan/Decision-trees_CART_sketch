"""
Feature importance calculation for decision trees.

This module implements the standard feature importance algorithm based on
weighted impurity decrease for each feature used in the tree.
"""

from typing import Dict, List
import numpy as np
from numpy.typing import NDArray

from .tree_structure import TreeNode


class FeatureImportanceCalculator:
    """
    Calculates feature importances from a fitted decision tree.

    Feature importance is computed as the sum of weighted impurity decreases
    for all splits using that feature, normalized by the total number of samples.
    """

    def __init__(self, feature_names: List[str]):
        """
        Initialize calculator.

        Parameters
        ----------
        feature_names : list of str
            Names of all features in the dataset
        """
        self.feature_names = feature_names
        self.n_features = len(feature_names)

    def compute_importances(self, root: TreeNode) -> NDArray:
        """
        Compute feature importances from a fitted tree.

        The importance of feature i is computed as:
            importance[i] = sum over all splits using feature i of:
                (n_samples_parent / n_samples_root) *
                (parent_impurity - weighted_child_impurity)

        Parameters
        ----------
        root : TreeNode
            Root node of the fitted decision tree

        Returns
        -------
        importances : NDArray of shape (n_features,)
            Feature importances (normalized to sum to 1.0)

        Examples
        --------
        >>> calculator = FeatureImportanceCalculator(['age>30', 'income>50k'])
        >>> importances = calculator.compute_importances(root_node)
        >>> print(importances)  # [0.7, 0.3] for example
        """
        # Initialize importance array
        importances = np.zeros(self.n_features)

        # Get total samples at root for normalization
        total_samples = root.n_samples

        # Recursively compute importances
        self._compute_node_importance(root, total_samples, importances)

        # Normalize to sum to 1.0
        total_importance = np.sum(importances)
        if total_importance > 0:
            importances = importances / total_importance
        else:
            # If no splits (single node tree), all features have equal importance
            importances = np.ones(self.n_features) / self.n_features

        return importances

    def _compute_node_importance(
        self,
        node: TreeNode,
        total_samples: float,
        importances: NDArray
    ) -> None:
        """
        Recursively compute importance contribution from this node and its subtree.

        Parameters
        ----------
        node : TreeNode
            Current node being processed
        total_samples : float
            Total samples at root (for normalization)
        importances : NDArray
            Array to accumulate importance values (modified in-place)
        """
        # Base case: leaf node contributes no feature importance
        if node.is_leaf:
            return

        # Get feature index for this split
        feature_name = node.feature_name
        if feature_name not in self.feature_names:
            # Skip unknown features (shouldn't happen in practice)
            return

        feature_idx = self.feature_names.index(feature_name)

        # Compute weighted impurity decrease for this split
        parent_impurity = node.impurity
        parent_samples = node.n_samples

        left_samples = node.left.n_samples
        right_samples = node.right.n_samples

        left_impurity = node.left.impurity
        right_impurity = node.right.impurity

        # Weighted average of child impurities
        total_child_samples = left_samples + right_samples
        if total_child_samples > 0:
            weighted_child_impurity = (
                (left_samples / total_child_samples) * left_impurity +
                (right_samples / total_child_samples) * right_impurity
            )
        else:
            weighted_child_impurity = parent_impurity  # No improvement

        # Compute impurity decrease
        impurity_decrease = parent_impurity - weighted_child_impurity

        # Weight by number of samples reaching this node
        sample_weight = parent_samples / total_samples
        weighted_importance = sample_weight * impurity_decrease

        # Feature importances should always be non-negative
        # Negative values can occur due to estimation errors in mock sketches
        weighted_importance = max(0.0, weighted_importance)

        # Add to feature's total importance
        importances[feature_idx] += weighted_importance

        # Recurse on children
        self._compute_node_importance(node.left, total_samples, importances)
        self._compute_node_importance(node.right, total_samples, importances)

    def get_feature_importance_dict(self, importances: NDArray) -> Dict[str, float]:
        """
        Convert importance array to feature name -> importance mapping.

        Parameters
        ----------
        importances : NDArray
            Feature importance array from compute_importances()

        Returns
        -------
        dict
            Mapping from feature names to importance scores

        Examples
        --------
        >>> importances = np.array([0.7, 0.3])
        >>> calculator.get_feature_importance_dict(importances)
        {'age>30': 0.7, 'income>50k': 0.3}
        """
        return {
            feature_name: float(importance)
            for feature_name, importance in zip(self.feature_names, importances)
        }

    def get_top_features(
        self,
        importances: NDArray,
        top_k: int = 5
    ) -> List[tuple]:
        """
        Get top k most important features.

        Parameters
        ----------
        importances : NDArray
            Feature importance array
        top_k : int, default=5
            Number of top features to return

        Returns
        -------
        list of tuple
            List of (feature_name, importance) tuples, sorted by importance

        Examples
        --------
        >>> top_features = calculator.get_top_features(importances, top_k=3)
        >>> print(top_features)
        [('age>30', 0.7), ('income>50k', 0.3)]
        """
        # Create list of (feature_name, importance) tuples
        feature_importance_pairs = [
            (feature_name, importance)
            for feature_name, importance in zip(self.feature_names, importances)
        ]

        # Sort by importance (descending)
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        return feature_importance_pairs[:top_k]


def compute_feature_importances(
    root: TreeNode,
    feature_names: List[str]
) -> NDArray:
    """
    Convenience function to compute feature importances.

    Parameters
    ----------
    root : TreeNode
        Root of the fitted decision tree
    feature_names : list of str
        Names of all features

    Returns
    -------
    importances : NDArray
        Feature importance scores (normalized to sum to 1.0)

    Examples
    --------
    >>> importances = compute_feature_importances(tree.tree_, tree.feature_names_in_)
    >>> print(importances)
    [0.7 0.3]
    """
    calculator = FeatureImportanceCalculator(feature_names)
    return calculator.compute_importances(root)
