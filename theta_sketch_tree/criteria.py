"""
Simplified criteria module for theta sketch decision trees.

This module provides split criteria with clean inheritance design while eliminating
unnecessary complexity. Maintains the same interface as the original but with
dramatically reduced line count.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict
import numpy as np
from numpy.typing import NDArray


class BaseCriterion(ABC):
    """Abstract base class for split criteria.

    Provides common functionality and interface for all split criteria.
    Each criterion must implement compute_impurity() and evaluate_split().
    """

    def __init__(self, class_weight: Optional[Dict[int, float]] = None):
        self.class_weight = class_weight

    @abstractmethod
    def compute_impurity(self, class_counts: NDArray[np.float64]) -> float:
        """Compute impurity for a node given class counts."""
        pass

    @abstractmethod
    def evaluate_split(self, parent_counts: NDArray[np.float64],
                      left_counts: NDArray[np.float64],
                      right_counts: NDArray[np.float64],
                      parent_impurity: Optional[float] = None) -> float:
        """Evaluate quality of a split. Lower scores indicate better splits."""
        pass

    def _apply_class_weights(self, counts: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply class weights to counts if specified."""
        if self.class_weight is None:
            return counts
        weighted = counts.copy()
        for class_idx, weight in self.class_weight.items():
            if class_idx < len(weighted):
                weighted[class_idx] *= weight
        return weighted


class GiniCriterion(BaseCriterion):
    """Gini impurity criterion: Gini(t) = 1 - Σ(p_i²)"""

    def compute_impurity(self, class_counts: NDArray[np.float64]) -> float:
        counts = self._apply_class_weights(class_counts)
        total = np.sum(counts)
        if total == 0:
            return 0.0
        proportions = counts / total
        return 1.0 - np.sum(proportions ** 2)

    def evaluate_split(self, parent_counts: NDArray[np.float64],
                      left_counts: NDArray[np.float64],
                      right_counts: NDArray[np.float64],
                      parent_impurity: Optional[float] = None) -> float:
        """Returns negative weighted impurity decrease (lower = better)."""
        if parent_impurity is None:
            parent_impurity = self.compute_impurity(parent_counts)
        total = np.sum(parent_counts)
        if total == 0:
            return 0.0

        left_weight = np.sum(left_counts) / total
        right_weight = np.sum(right_counts) / total

        weighted_impurity = (left_weight * self.compute_impurity(left_counts) +
                           right_weight * self.compute_impurity(right_counts))

        return weighted_impurity - parent_impurity  # Negative decrease


class EntropyCriterion(BaseCriterion):
    """Shannon entropy criterion: H(t) = -Σ(p_i * log2(p_i))"""

    def compute_impurity(self, class_counts: NDArray[np.float64]) -> float:
        counts = self._apply_class_weights(class_counts)
        total = np.sum(counts)
        if total == 0:
            return 0.0
        proportions = counts / total
        # Avoid log(0) by filtering out zero proportions
        proportions = proportions[proportions > 0]
        return -np.sum(proportions * np.log2(proportions))

    def evaluate_split(self, parent_counts: NDArray[np.float64],
                      left_counts: NDArray[np.float64],
                      right_counts: NDArray[np.float64],
                      parent_impurity: Optional[float] = None) -> float:
        """Returns negative information gain (lower = better)."""
        if parent_impurity is None:
            parent_entropy = self.compute_impurity(parent_counts)
        else:
            parent_entropy = parent_impurity
        total = np.sum(parent_counts)
        if total == 0:
            return 0.0

        left_weight = np.sum(left_counts) / total
        right_weight = np.sum(right_counts) / total

        weighted_entropy = (left_weight * self.compute_impurity(left_counts) +
                          right_weight * self.compute_impurity(right_counts))

        return weighted_entropy - parent_entropy  # Negative gain


class GainRatioCriterion(EntropyCriterion):
    """Gain ratio criterion: GainRatio = InformationGain / SplitInfo

    Inherits entropy computation from EntropyCriterion and adds split info normalization.
    """

    def evaluate_split(self, parent_counts: NDArray[np.float64],
                      left_counts: NDArray[np.float64],
                      right_counts: NDArray[np.float64],
                      parent_impurity: Optional[float] = None) -> float:
        """Returns negative gain ratio (lower = better)."""
        # Get information gain from parent class
        info_gain = -super().evaluate_split(parent_counts, left_counts, right_counts, parent_impurity)

        # Calculate split information
        total = np.sum(parent_counts)
        if total == 0:
            return 0.0

        left_prop = np.sum(left_counts) / total
        right_prop = np.sum(right_counts) / total

        # Avoid log(0)
        split_info = 0.0
        if left_prop > 0:
            split_info -= left_prop * np.log2(left_prop)
        if right_prop > 0:
            split_info -= right_prop * np.log2(right_prop)

        # Avoid division by zero
        if split_info == 0:
            return float('-inf')  # Perfect split with no split info

        gain_ratio = info_gain / split_info
        return -gain_ratio  # Return negative for consistency


class BinomialCriterion(BaseCriterion):
    """Binomial test criterion for statistical significance."""

    def __init__(self, class_weight: Optional[Dict[int, float]] = None, min_pvalue: float = 0.05):
        super().__init__(class_weight)
        self.min_pvalue = min_pvalue

    def compute_impurity(self, class_counts: NDArray[np.float64]) -> float:
        """Returns p-value from binomial test."""
        from scipy.stats import binomtest
        counts = self._apply_class_weights(class_counts)
        total = np.sum(counts)
        if total == 0 or len(counts) < 2:
            return 1.0
        successes = counts[1]  # Positive class
        try:
            result = binomtest(int(successes), int(total), 0.5)
            return result.pvalue
        except (ValueError, AttributeError):
            # Fallback for older scipy versions
            try:
                from scipy.stats import binom_test
                return binom_test(successes, int(total), 0.5)
            except ImportError:
                # Manual binomial test calculation
                from scipy.stats import binom
                p_value = 2 * min(binom.cdf(successes, total, 0.5),
                                 1 - binom.cdf(successes - 1, total, 0.5))
                return p_value

    def evaluate_split(self, parent_counts: NDArray[np.float64],
                      left_counts: NDArray[np.float64],
                      right_counts: NDArray[np.float64],
                      parent_impurity: Optional[float] = None) -> float:
        """Returns p-value (lower = better split)."""
        return min(self.compute_impurity(left_counts), self.compute_impurity(right_counts))


class ChiSquareCriterion(BaseCriterion):
    """Chi-square test criterion for independence."""

    def __init__(self, class_weight: Optional[Dict[int, float]] = None, min_pvalue: float = 0.05):
        super().__init__(class_weight)
        self.min_pvalue = min_pvalue

    def compute_impurity(self, class_counts: NDArray[np.float64]) -> float:
        """Chi-square doesn't have node-level impurity, return 0."""
        return 0.0

    def evaluate_split(self, parent_counts: NDArray[np.float64],
                      left_counts: NDArray[np.float64],
                      right_counts: NDArray[np.float64],
                      parent_impurity: Optional[float] = None) -> float:
        """Returns chi-square test p-value (lower = better split)."""
        from scipy.stats import chi2_contingency

        # Create contingency table
        contingency = np.array([left_counts, right_counts])

        # Handle edge cases
        if np.any(np.sum(contingency, axis=0) == 0) or np.any(np.sum(contingency, axis=1) == 0):
            return 1.0

        try:
            _, p_value, _, _ = chi2_contingency(contingency)
            return p_value
        except ValueError:
            return 1.0


def get_criterion(criterion_name: str, **kwargs) -> BaseCriterion:
    """Factory function to create criterion instances.

    Args:
        criterion_name: One of 'gini', 'entropy', 'gain_ratio', 'binomial', 'chi_square'
        **kwargs: Arguments passed to criterion constructor

    Returns:
        BaseCriterion: Criterion instance

    Raises:
        ValueError: If criterion_name is not recognized
    """
    criteria = {
        'gini': GiniCriterion,
        'entropy': EntropyCriterion,
        'gain_ratio': GainRatioCriterion,
        'binomial': BinomialCriterion,
        'chi_square': ChiSquareCriterion,
        'binomial_chi': BinomialCriterion,  # Alias
    }

    name = criterion_name.lower()
    if name not in criteria:
        raise ValueError(f"Unknown criterion: '{criterion_name}'. Valid: {list(criteria.keys())}")

    return criteria[name](**kwargs)