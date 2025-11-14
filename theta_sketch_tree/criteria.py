"""
Criteria module.

This module implements split criteria (Gini, Entropy, Gain Ratio, Binomial, Chi-Square)
for evaluating node splits in decision trees.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict
import numpy as np
from numpy.typing import NDArray


class BaseCriterion(ABC):
    """
    Abstract base class for split criteria.

    All criterion classes must implement:
    - compute_impurity(): Calculate node impurity
    - evaluate_split(): Score quality of a split

    Parameters
    ----------
    class_weight : dict, optional
        Weights for each class {class_label: weight}
    min_pvalue : float, default=0.05
        Minimum p-value for statistical tests (Binomial, Chi-Square)
    use_bonferroni : bool, default=True
        Whether to use Bonferroni correction for multiple testing
    """

    def __init__(
        self,
        class_weight: Optional[Dict[int, float]] = None,
        min_pvalue: float = 0.05,
        use_bonferroni: bool = True
    ) -> None:
        """Initialize criterion."""
        self.class_weight = class_weight
        self.min_pvalue = min_pvalue
        self.use_bonferroni = use_bonferroni

    @abstractmethod
    def compute_impurity(
        self,
        class_counts: NDArray[np.float64]
    ) -> float:
        """
        Compute impurity for a node.

        Parameters
        ----------
        class_counts : array of shape (n_classes,)
            Count of samples per class [n_class_0, n_class_1]

        Returns
        -------
        impurity : float
            Impurity value (interpretation depends on criterion)
        """
        pass

    @abstractmethod
    def evaluate_split(
        self,
        parent_counts: NDArray[np.float64],
        left_counts: NDArray[np.float64],
        right_counts: NDArray[np.float64],
        parent_impurity: Optional[float] = None
    ) -> float:
        """
        Evaluate quality of a split.

        Parameters
        ----------
        parent_counts : array of shape (n_classes,)
            Class counts at parent node
        left_counts : array of shape (n_classes,)
            Class counts at left child
        right_counts : array of shape (n_classes,)
            Class counts at right child
        parent_impurity : float, optional
            Pre-computed impurity of parent node (can be None)

        Returns
        -------
        score : float
            Split score (lower is better for most criteria)
            - For Gini/Entropy: Returns negative impurity decrease
            - For statistical tests: Returns p-value
        """
        pass


class GiniCriterion(BaseCriterion):
    """
    Gini impurity criterion.

    Gini(t) = 1 - Σ(p_i²)

    For weighted Gini:
    Gini_weighted(t) = 1 - Σ((w_i * p_i)²) / (Σw_i * p_i)²

    Lower Gini values indicate purer nodes.
    """

    def compute_impurity(
        self,
        class_counts: NDArray[np.float64]
    ) -> float:
        """
        Compute Gini impurity.

        Parameters
        ----------
        class_counts : array of shape (n_classes,)
            Count of samples per class

        Returns
        -------
        gini : float
            Gini impurity value in [0, 1]
        """
        total = np.sum(class_counts)
        if total == 0:
            return 0.0

        if self.class_weight is None:
            # Standard Gini
            probabilities = class_counts / total
            return 1.0 - np.sum(probabilities ** 2)
        else:
            # Weighted Gini
            weights = np.array([self.class_weight.get(i, 1.0) for i in range(len(class_counts))])
            weighted_counts = class_counts * weights
            total_weighted = np.sum(weighted_counts)
            if total_weighted == 0:
                return 0.0
            weighted_probs = weighted_counts / total_weighted
            return 1.0 - np.sum(weighted_probs ** 2)

    def evaluate_split(
        self,
        parent_counts: NDArray[np.float64],
        left_counts: NDArray[np.float64],
        right_counts: NDArray[np.float64],
        parent_impurity: Optional[float] = None
    ) -> float:
        """
        Evaluate split using weighted impurity decrease.

        Returns negative impurity decrease (lower is better for maximizing gain).

        Parameters
        ----------
        parent_counts : array
            Class counts at parent
        left_counts : array
            Class counts at left child
        right_counts : array
            Class counts at right child
        parent_impurity : float, optional
            Pre-computed parent impurity

        Returns
        -------
        score : float
            Negative impurity decrease (lower = better split)
        """
        n_parent = np.sum(parent_counts)
        n_left = np.sum(left_counts)
        n_right = np.sum(right_counts)

        if n_left == 0 or n_right == 0:
            return 0.0  # Invalid split

        # Compute impurities
        if parent_impurity is None:
            parent_impurity = self.compute_impurity(parent_counts)
        left_impurity = self.compute_impurity(left_counts)
        right_impurity = self.compute_impurity(right_counts)

        # Weighted average of child impurities
        weighted_impurity = (n_left / n_parent) * left_impurity + (n_right / n_parent) * right_impurity

        # Impurity decrease (higher is better, so return negative)
        impurity_decrease = parent_impurity - weighted_impurity
        return -impurity_decrease


class EntropyCriterion(BaseCriterion):
    """
    Shannon entropy criterion (Information Gain).

    Entropy(t) = -Σ(p_i * log2(p_i))
    Information Gain = Entropy(parent) - Weighted_Avg(Entropy(children))

    Lower entropy indicates purer nodes.
    """

    def compute_impurity(
        self,
        class_counts: NDArray[np.float64]
    ) -> float:
        """
        Compute Shannon entropy.

        Parameters
        ----------
        class_counts : array of shape (n_classes,)
            Count of samples per class

        Returns
        -------
        entropy : float
            Shannon entropy value
        """
        total = np.sum(class_counts)
        if total == 0:
            return 0.0

        probabilities = class_counts / total
        # Avoid log(0) by filtering zero probabilities
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))

    def evaluate_split(
        self,
        parent_counts: NDArray[np.float64],
        left_counts: NDArray[np.float64],
        right_counts: NDArray[np.float64],
        parent_impurity: Optional[float] = None
    ) -> float:
        """
        Evaluate split using information gain (return negative).

        Parameters
        ----------
        parent_counts : array
            Class counts at parent
        left_counts : array
            Class counts at left child
        right_counts : array
            Class counts at right child
        parent_impurity : float, optional
            Pre-computed parent impurity

        Returns
        -------
        score : float
            Negative information gain (lower = better split)
        """
        n_parent = np.sum(parent_counts)
        n_left = np.sum(left_counts)
        n_right = np.sum(right_counts)

        if n_left == 0 or n_right == 0:
            return 0.0  # Invalid split

        # Compute entropies
        if parent_impurity is None:
            parent_impurity = self.compute_impurity(parent_counts)
        left_impurity = self.compute_impurity(left_counts)
        right_impurity = self.compute_impurity(right_counts)

        # Weighted average entropy of children
        weighted_impurity = (n_left / n_parent) * left_impurity + (n_right / n_parent) * right_impurity

        # Information gain
        information_gain = parent_impurity - weighted_impurity

        return -information_gain


class GainRatioCriterion(EntropyCriterion):
    """
    Gain Ratio criterion (C4.5).

    GainRatio = InformationGain / SplitInfo
    where SplitInfo = -Σ(|child|/|parent| * log2(|child|/|parent|))

    Normalizes information gain by split entropy to prevent bias toward
    multi-way splits. Inherits entropy computation from EntropyCriterion.
    """

    def evaluate_split(
        self,
        parent_counts: NDArray[np.float64],
        left_counts: NDArray[np.float64],
        right_counts: NDArray[np.float64],
        parent_impurity: Optional[float] = None
    ) -> float:
        """
        Evaluate split using gain ratio (return negative).

        Parameters
        ----------
        parent_counts : array
            Class counts at parent
        left_counts : array
            Class counts at left child
        right_counts : array
            Class counts at right child
        parent_impurity : float, optional
            Pre-computed parent impurity

        Returns
        -------
        score : float
            Negative gain ratio (lower = better split)
        """
        n_parent = np.sum(parent_counts)
        n_left = np.sum(left_counts)
        n_right = np.sum(right_counts)

        if n_left == 0 or n_right == 0:
            return 0.0  # Invalid split

        # Compute information gain
        if parent_impurity is None:
            parent_impurity = self.compute_impurity(parent_counts)
        left_impurity = self.compute_impurity(left_counts)
        right_impurity = self.compute_impurity(right_counts)
        weighted_impurity = (n_left / n_parent) * left_impurity + (n_right / n_parent) * right_impurity
        information_gain = parent_impurity - weighted_impurity

        # Compute split information
        p_left = n_left / n_parent
        p_right = n_right / n_parent

        # Avoid log(0)
        split_info = 0.0
        if p_left > 0:
            split_info -= p_left * np.log2(p_left)
        if p_right > 0:
            split_info -= p_right * np.log2(p_right)

        if split_info == 0:
            return 0.0  # Avoid division by zero

        gain_ratio = information_gain / split_info
        return -gain_ratio


class BinomialCriterion(BaseCriterion):
    """
    Binomial statistical test criterion.

    Tests whether class proportions in children are significantly different from parent.
    Uses binomial test p-value as split criterion (lower p-value = more significant split).

    This criterion is useful for preventing overfitting by requiring statistical
    significance for splits.
    """

    def compute_impurity(
        self,
        class_counts: NDArray[np.float64]
    ) -> float:
        """
        Not used for statistical tests, return Gini as fallback.

        Parameters
        ----------
        class_counts : array
            Class counts

        Returns
        -------
        impurity : float
            Gini impurity
        """
        total = np.sum(class_counts)
        if total == 0:
            return 0.0
        probabilities = class_counts / total
        return 1.0 - np.sum(probabilities ** 2)

    def evaluate_split(
        self,
        parent_counts: NDArray[np.float64],
        left_counts: NDArray[np.float64],
        right_counts: NDArray[np.float64],
        parent_impurity: Optional[float] = None
    ) -> float:
        """
        Evaluate split using binomial test.

        Tests if the proportion of positive class in each child is significantly
        different from the parent proportion.

        Parameters
        ----------
        parent_counts : array
            Class counts at parent [n_class_0, n_class_1]
        left_counts : array
            Class counts at left child
        right_counts : array
            Class counts at right child
        parent_impurity : float, optional
            Not used

        Returns
        -------
        p_value : float
            Minimum p-value from children (lower = better split)
        """
        from scipy.stats import binomtest

        n_parent = np.sum(parent_counts)
        n_left = np.sum(left_counts)
        n_right = np.sum(right_counts)

        if n_left == 0 or n_right == 0:
            return 1.0  # Not significant

        # Proportion of positive class in parent
        p_parent = parent_counts[1] / n_parent if n_parent > 0 else 0.5

        # Test left child
        k_left = int(left_counts[1])
        n_left_int = int(n_left)
        p_value_left = binomtest(k_left, n_left_int, p_parent, alternative='two-sided').pvalue

        # Test right child
        k_right = int(right_counts[1])
        n_right_int = int(n_right)
        p_value_right = binomtest(k_right, n_right_int, p_parent, alternative='two-sided').pvalue

        # Use minimum p-value (most significant child)
        p_value = min(p_value_left, p_value_right)

        # Bonferroni correction if enabled
        if self.use_bonferroni:
            # Note: Correction factor could be number of features tested
            # For now, we return raw p-value (correction applied at tree level)
            pass

        return p_value


class ChiSquareCriterion(BaseCriterion):
    """
    Chi-square test criterion.

    Tests independence between split and class label using chi-square test.
    Lower p-value indicates stronger dependence (better split).

    Contingency table format:
                     Class_0  Class_1
        Left_child   n_00     n_01
        Right_child  n_10     n_11
    """

    def compute_impurity(
        self,
        class_counts: NDArray[np.float64]
    ) -> float:
        """
        Not used for statistical tests, return Gini as fallback.

        Parameters
        ----------
        class_counts : array
            Class counts

        Returns
        -------
        impurity : float
            Gini impurity
        """
        total = np.sum(class_counts)
        if total == 0:
            return 0.0
        probabilities = class_counts / total
        return 1.0 - np.sum(probabilities ** 2)

    def evaluate_split(
        self,
        parent_counts: NDArray[np.float64],
        left_counts: NDArray[np.float64],
        right_counts: NDArray[np.float64],
        parent_impurity: Optional[float] = None
    ) -> float:
        """
        Evaluate split using chi-square test.

        Parameters
        ----------
        parent_counts : array
            Class counts at parent (not used)
        left_counts : array
            Class counts at left child [n_class_0, n_class_1]
        right_counts : array
            Class counts at right child [n_class_0, n_class_1]
        parent_impurity : float, optional
            Not used

        Returns
        -------
        p_value : float
            Chi-square test p-value (lower = better split)
        """
        from scipy.stats import chi2_contingency

        # Contingency table: [left_class0, left_class1]
        #                     [right_class0, right_class1]
        contingency_table = np.array([
            left_counts,
            right_counts
        ])

        # Avoid issues with zero counts
        if np.any(np.sum(contingency_table, axis=0) == 0) or np.any(np.sum(contingency_table, axis=1) == 0):
            return 1.0  # Not significant

        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            return p_value
        except ValueError:
            # Handle edge cases where chi2_contingency fails
            return 1.0


def get_criterion(criterion_name: str, **kwargs) -> BaseCriterion:
    """
    Factory function to get criterion instance by name.

    Parameters
    ----------
    criterion_name : str
        Name of criterion: 'gini', 'entropy', 'gain_ratio', 'binomial', 'chi_square'
    **kwargs : dict
        Additional arguments passed to criterion constructor

    Returns
    -------
    criterion : BaseCriterion
        Criterion instance

    Raises
    ------
    ValueError
        If criterion_name is not recognized

    Examples
    --------
    >>> criterion = get_criterion('gini')
    >>> criterion = get_criterion('entropy')
    >>> criterion = get_criterion('binomial', min_pvalue=0.01)
    """
    criterion_map = {
        'gini': GiniCriterion,
        'entropy': EntropyCriterion,
        'gain_ratio': GainRatioCriterion,
        'binomial': BinomialCriterion,
        'chi_square': ChiSquareCriterion,
        'binomial_chi': BinomialCriterion,  # Alias
    }

    criterion_name = criterion_name.lower()
    if criterion_name not in criterion_map:
        raise ValueError(
            f"Unknown criterion: '{criterion_name}'. "
            f"Valid options: {list(criterion_map.keys())}"
        )

    return criterion_map[criterion_name](**kwargs)
