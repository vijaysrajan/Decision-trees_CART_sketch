"""
Feature importance calculation for theta sketch decision trees.

This module provides clean separation of feature importance logic
from the main classifier class with comprehensive validation and logging.
"""

from typing import List
import numpy as np
from numpy.typing import NDArray

from .tree_structure import TreeNode
from .validation_utils import ValidationError
from .logging_utils import TreeLogger


class FeatureImportanceCalculator:
    """
    Calculates feature importances using weighted impurity decrease method.

    This is the standard approach used by scikit-learn's decision trees,
    where importance is measured by the total decrease in node impurity
    weighted by the probability of reaching that node.
    """

    @staticmethod
    def compute_importances(
        tree_root: TreeNode,
        feature_names: NDArray,
        n_features: int,
        verbose: int = 0
    ) -> NDArray:
        """
        Compute feature importances from a fitted tree.

        Parameters
        ----------
        tree_root : TreeNode
            Root node of the fitted decision tree
        feature_names : ndarray
            Array of feature names
        n_features : int
            Number of features
        verbose : int, optional
            Verbosity level for logging

        Returns
        -------
        importances : ndarray
            Feature importance scores normalized to sum to 1.0

        Raises
        ------
        ValidationError
            If inputs are invalid
        """
        # Validate inputs
        if tree_root is None:
            raise ValidationError("tree_root cannot be None")
        if n_features <= 0:
            raise ValidationError(f"n_features must be positive, got {n_features}")
        if len(feature_names) != n_features:
            raise ValidationError(f"feature_names length ({len(feature_names)}) != n_features ({n_features})")

        logger = TreeLogger("FeatureImportance", verbose)
        logger.debug(f"Computing feature importances for {n_features} features")

        # Initialize importance array
        importances = np.zeros(n_features)

        # Get total samples at root for normalization
        total_samples = tree_root.n_samples
        logger.debug(f"Total samples at root: {total_samples}")

        # Recursively compute importances
        FeatureImportanceCalculator._compute_node_importance(
            tree_root, total_samples, importances, feature_names
        )

        # Normalize to sum to 1.0
        total_importance = np.sum(importances)
        if total_importance > 0:
            importances = importances / total_importance
            logger.debug(f"Normalized importances, total: {total_importance:.6f}")
        else:
            # If no splits (single node tree), all features have equal importance
            importances = np.ones(n_features) / n_features
            logger.debug("No splits found, using uniform feature importance")

        return importances

    @staticmethod
    def _compute_node_importance(
        node: TreeNode,
        total_samples: float,
        importances: NDArray,
        feature_names: NDArray
    ) -> None:
        """
        Recursively compute importance contribution from this node and its subtree.

        Parameters
        ----------
        node : TreeNode
            Current tree node
        total_samples : float
            Total samples at root (for normalization)
        importances : ndarray
            Array to accumulate importance scores (modified in-place)
        feature_names : ndarray
            Array of feature names
        """
        # Base case: leaf node contributes no feature importance
        if node.is_leaf:
            return

        # Get feature index for this split
        feature_name = node.feature_name
        feature_names_list = list(feature_names)

        if feature_name not in feature_names_list:
            # Skip unknown features (shouldn't happen in practice)
            return

        feature_idx = feature_names_list.index(feature_name)

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
        weighted_importance = max(0.0, weighted_importance)

        # Add to feature's total importance
        importances[feature_idx] += weighted_importance

        # Recurse on children
        FeatureImportanceCalculator._compute_node_importance(
            node.left, total_samples, importances, feature_names
        )
        FeatureImportanceCalculator._compute_node_importance(
            node.right, total_samples, importances, feature_names
        )


# Convenience function for backward compatibility
def compute_feature_importances(
    tree_root: TreeNode,
    feature_names: NDArray,
    n_features: int
) -> NDArray:
    """
    Convenience function to compute feature importances.

    Parameters
    ----------
    tree_root : TreeNode
        Root node of the fitted decision tree
    feature_names : ndarray
        Array of feature names
    n_features : int
        Number of features

    Returns
    -------
    importances : ndarray
        Feature importance scores normalized to sum to 1.0
    """
    return FeatureImportanceCalculator.compute_importances(
        tree_root, feature_names, n_features
    )