"""
Split evaluation logic for theta sketch decision trees.

This module provides clean separation of split finding logic
from the main tree builder class.
"""

from typing import Dict, Set, List, Any, Optional, Tuple, NamedTuple
import numpy as np
from numpy.typing import NDArray
from .logging_utils import TreeLogger


class SplitResult(NamedTuple):
    """
    Result of split evaluation containing all necessary information.
    """
    feature_name: str
    score: float
    left_sketch_pos: Any
    left_sketch_neg: Any
    right_sketch_pos: Any
    right_sketch_neg: Any
    left_counts: NDArray
    right_counts: NDArray


class SplitFinder:
    """
    Finds the best split for a tree node using theta sketches.

    This class encapsulates all split evaluation logic including:
    - Feature filtering and availability checking
    - Sketch intersection computation
    - Split quality evaluation
    - Minimum samples constraint validation
    - Categorical mutual exclusivity constraints
    """

    def __init__(
        self,
        criterion: Any,
        min_samples_leaf: int = 1,
        verbose: int = 0
    ):
        """
        Initialize split finder.

        Parameters
        ----------
        criterion : Criterion
            Split evaluation criterion (gini, entropy, etc.)
        min_samples_leaf : int
            Minimum samples required in leaf nodes
        verbose : int
            Verbosity level for logging
        """
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.verbose = verbose
        self.logger = TreeLogger(__name__)

    def find_best_split(
        self,
        parent_sketch_pos: Any,
        parent_sketch_neg: Any,
        parent_class_counts: NDArray,
        parent_impurity: float,
        sketch_dict: Dict[str, Dict[str, Any]],
        feature_names: List[str],
        already_used: Set[str]
    ) -> Optional[SplitResult]:
        """
        Find the best split for the current node.

        Parameters
        ----------
        parent_sketch_pos : ThetaSketch
            Parent positive class sketch
        parent_sketch_neg : ThetaSketch
            Parent negative class sketch
        parent_class_counts : ndarray
            Class counts at parent node [neg_count, pos_count]
        parent_impurity : float
            Impurity at parent node
        sketch_dict : dict
            Global sketch dictionary
        feature_names : list
            Available feature names
        already_used : set
            Features already used in this path

        Returns
        -------
        split_result : SplitResult or None
            Best split found, or None if no valid split exists
        """
        best_split = None
        best_score = float('inf')  # Lower is better

        # Get available features
        available_features = self._get_available_features(
            feature_names, already_used, sketch_dict
        )

        if not available_features:
            if self.verbose >= 2:
                self.logger.info("No features available for splitting", level=2)
            return None

        # Evaluate each feature
        for feature_name in available_features:
            split_result = self._evaluate_feature_split(
                feature_name,
                parent_sketch_pos,
                parent_sketch_neg,
                parent_class_counts,
                parent_impurity,
                sketch_dict
            )

            if split_result is not None and split_result.score < best_score:
                best_score = split_result.score
                best_split = split_result

                if self.verbose >= 3:
                    self.logger.debug(f"  New best split: {feature_name} (score: {best_score:.4f})")

        return best_split

    def _get_available_features(
        self,
        feature_names: List[str],
        already_used: Set[str],
        sketch_dict: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Filter features to only those available and not already used.

        Handles mutual exclusivity ONLY for 'variable=value' format features.
        Standalone features (like ICD codes) are treated as independent.

        Parameters
        ----------
        feature_names : list
            All feature names
        already_used : set
            Features already used in this path
        sketch_dict : dict
            Global sketch dictionary

        Returns
        -------
        available_features : list
            List of available feature names
        """
        # Filter available features
        available_features = []
        for feature_name in feature_names:
            # Skip if feature not in sketch data
            if (feature_name not in sketch_dict.get('positive', {}) or
                feature_name not in sketch_dict.get('negative', {})):
                continue

            # Skip if exact feature already used
            if feature_name in already_used:
                continue

            # Feature is available
            available_features.append(feature_name)

        return available_features

    def _evaluate_feature_split(
        self,
        feature_name: str,
        parent_sketch_pos: Any,
        parent_sketch_neg: Any,
        parent_class_counts: NDArray,
        parent_impurity: float,
        sketch_dict: Dict[str, Dict[str, Any]]
    ) -> Optional[SplitResult]:
        """
        Evaluate a single feature for splitting.

        Parameters
        ----------
        feature_name : str
            Name of feature to evaluate
        parent_sketch_pos : ThetaSketch
            Parent positive class sketch
        parent_sketch_neg : ThetaSketch
            Parent negative class sketch
        parent_class_counts : ndarray
            Class counts at parent node
        parent_impurity : float
            Impurity at parent node
        sketch_dict : dict
            Global sketch dictionary

        Returns
        -------
        split_result : SplitResult or None
            Split result or None if invalid
        """
        # Get feature sketches from the global dictionary
        try:
            feature_present_pos = sketch_dict['positive'][feature_name][0]
            feature_absent_pos = sketch_dict['positive'][feature_name][1]
            feature_present_neg = sketch_dict['negative'][feature_name][0]
            feature_absent_neg = sketch_dict['negative'][feature_name][1]
        except (KeyError, IndexError):
            if self.verbose >= 3:
                self.logger.debug(f"  Skipping {feature_name}: invalid sketch structure")
            return None

        # Compute child sketches using intersection (∩)
        left_sketch_pos = parent_sketch_pos.intersection(feature_absent_pos)
        left_sketch_neg = parent_sketch_neg.intersection(feature_absent_neg)

        right_sketch_pos = parent_sketch_pos.intersection(feature_present_pos)
        right_sketch_neg = parent_sketch_neg.intersection(feature_present_neg)

        # Get child estimates
        left_pos = left_sketch_pos.get_estimate()
        left_neg = left_sketch_neg.get_estimate()
        right_pos = right_sketch_pos.get_estimate()
        right_neg = right_sketch_neg.get_estimate()

        # Skip if either child is empty
        if (left_pos + left_neg == 0) or (right_pos + right_neg == 0):
            if self.verbose >= 3:
                self.logger.debug(f"  Skipping {feature_name}: empty child node")
            return None

        # Calculate class counts
        left_counts = np.array([left_neg, left_pos])
        right_counts = np.array([right_neg, right_pos])

        # Check minimum samples constraint
        if not self._validate_min_samples(left_counts, right_counts):
            if self.verbose >= 3:
                self.logger.debug(f"  Skipping {feature_name}: min_samples_leaf constraint")
            return None

        # Evaluate split using criterion (lower is better)
        score = self.criterion.evaluate_split(
            parent_class_counts, left_counts, right_counts, parent_impurity
        )

        return SplitResult(
            feature_name=feature_name,
            score=score,
            left_sketch_pos=left_sketch_pos,
            left_sketch_neg=left_sketch_neg,
            right_sketch_pos=right_sketch_pos,
            right_sketch_neg=right_sketch_neg,
            left_counts=left_counts,
            right_counts=right_counts
        )

    def _validate_min_samples(
        self,
        left_counts: NDArray,
        right_counts: NDArray
    ) -> bool:
        """
        Validate minimum samples constraint for child nodes.

        Parameters
        ----------
        left_counts : ndarray
            Class counts for left child
        right_counts : ndarray
            Class counts for right child

        Returns
        -------
        valid : bool
            True if both children satisfy minimum samples constraint
        """
        left_total = np.sum(left_counts)
        right_total = np.sum(right_counts)

        return (left_total >= self.min_samples_leaf and
                right_total >= self.min_samples_leaf)


# Convenience function for backward compatibility
def find_best_split(
    parent_sketch_pos: Any,
    parent_sketch_neg: Any,
    parent_class_counts: NDArray,
    parent_impurity: float,
    sketch_dict: Dict[str, Dict[str, Any]],
    feature_names: List[str],
    already_used: Set[str],
    criterion: Any,
    min_samples_leaf: int = 1,
    verbose: int = 0
) -> Optional[SplitResult]:
    """
    Convenience function to find best split.

    Parameters
    ----------
    parent_sketch_pos : ThetaSketch
        Parent positive class sketch
    parent_sketch_neg : ThetaSketch
        Parent negative class sketch
    parent_class_counts : ndarray
        Class counts at parent node
    parent_impurity : float
        Impurity at parent node
    sketch_dict : dict
        Global sketch dictionary
    feature_names : list
        Available feature names
    already_used : set
        Features already used in this path
    criterion : Criterion
        Split evaluation criterion
    min_samples_leaf : int
        Minimum samples in leaf nodes
    verbose : int
        Verbosity level

    Returns
    -------
    split_result : SplitResult or None
        Best split found, or None if no valid split exists
    """
    finder = SplitFinder(criterion, min_samples_leaf, verbose)
    return finder.find_best_split(
        parent_sketch_pos,
        parent_sketch_neg,
        parent_class_counts,
        parent_impurity,
        sketch_dict,
        feature_names,
        already_used
    )