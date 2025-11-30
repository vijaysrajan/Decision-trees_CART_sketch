"""
Interfaces and base classes for theta sketch tree components.

Defines clear contracts and type interfaces for extensibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, Set, List, Any, Optional, Protocol, runtime_checkable
import numpy as np
from numpy.typing import NDArray

from .tree_structure import TreeNode


@runtime_checkable
class ThetaSketch(Protocol):
    """Protocol for theta sketch objects."""

    def get_estimate(self) -> float:
        """Get cardinality estimate from sketch."""
        ...

    def intersect(self, other: 'ThetaSketch') -> 'ThetaSketch':
        """Compute intersection with another sketch."""
        ...


class BaseCriterion(ABC):
    """
    Abstract base class for split evaluation criteria.

    All criteria implementations must inherit from this class.
    """

    @abstractmethod
    def compute_impurity(self, class_counts: NDArray) -> float:
        """
        Compute impurity for given class distribution.

        Parameters
        ----------
        class_counts : ndarray
            Class counts [n_class_0, n_class_1, ...]

        Returns
        -------
        impurity : float
            Impurity measure (0 = pure, higher = more impure)
        """
        pass

    @abstractmethod
    def compute_split_score(
        self,
        parent_impurity: float,
        left_class_counts: NDArray,
        right_class_counts: NDArray
    ) -> float:
        """
        Compute score for a split.

        Parameters
        ----------
        parent_impurity : float
            Impurity of parent node
        left_class_counts : ndarray
            Class counts in left child
        right_class_counts : ndarray
            Class counts in right child

        Returns
        -------
        score : float
            Split score (higher = better split)
        """
        pass


class BaseSplitFinder(ABC):
    """
    Abstract base class for split finding strategies.
    """

    @abstractmethod
    def find_best_split(
        self,
        parent_sketch_pos: ThetaSketch,
        parent_sketch_neg: ThetaSketch,
        parent_class_counts: NDArray,
        parent_impurity: float,
        sketch_dict: Dict[str, Dict[str, Any]],
        feature_names: List[str],
        already_used: Set[str]
    ) -> Optional['SplitResult']:
        """
        Find the best split for current node.

        Parameters
        ----------
        parent_sketch_pos : ThetaSketch
            Positive class sketch at this node
        parent_sketch_neg : ThetaSketch
            Negative class sketch at this node
        parent_class_counts : ndarray
            Class counts at this node
        parent_impurity : float
            Impurity at this node
        sketch_dict : dict
            Global feature sketches
        feature_names : list
            Available feature names
        already_used : set
            Features already used in path from root

        Returns
        -------
        split_result : SplitResult or None
            Best split information, or None if no valid split found
        """
        pass


class BaseStoppingCriteria(ABC):
    """
    Abstract base class for tree stopping criteria.
    """

    @abstractmethod
    def should_stop_splitting(
        self,
        depth: int,
        n_samples: float,
        impurity: float,
        class_counts: NDArray,
        available_features: List[str]
    ) -> tuple[bool, str]:
        """
        Check if node splitting should stop.

        Parameters
        ----------
        depth : int
            Current tree depth
        n_samples : float
            Number of samples at this node
        impurity : float
            Node impurity
        class_counts : ndarray
            Class counts at this node
        available_features : list
            Available features for splitting

        Returns
        -------
        should_stop : bool
            True if splitting should stop
        reason : str
            Reason for stopping
        """
        pass


class BasePruner(ABC):
    """
    Abstract base class for tree pruning strategies.
    """

    @abstractmethod
    def prune_tree(
        self,
        tree_root: TreeNode,
        **kwargs
    ) -> TreeNode:
        """
        Apply pruning to tree.

        Parameters
        ----------
        tree_root : TreeNode
            Root of tree to prune
        **kwargs
            Additional pruning parameters

        Returns
        -------
        pruned_root : TreeNode
            Root of pruned tree
        """
        pass


class BaseTreeTraverser(ABC):
    """
    Abstract base class for tree traversal strategies.
    """

    @abstractmethod
    def predict(self, X: NDArray) -> NDArray:
        """
        Predict class labels.

        Parameters
        ----------
        X : ndarray
            Input features

        Returns
        -------
        y_pred : ndarray
            Predicted class labels
        """
        pass

    @abstractmethod
    def predict_proba(self, X: NDArray) -> NDArray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : ndarray
            Input features

        Returns
        -------
        y_proba : ndarray
            Class probabilities
        """
        pass


class ComponentFactory:
    """
    Factory for creating tree components with proper validation.
    """

    @staticmethod
    def create_criterion(criterion_name: str) -> BaseCriterion:
        """
        Create criterion instance from name.

        Parameters
        ----------
        criterion_name : str
            Name of criterion ('gini', 'entropy', etc.)

        Returns
        -------
        criterion : BaseCriterion
            Criterion instance
        """
        from .criteria import get_criterion
        from .validation_utils import ParameterValidator

        ParameterValidator.validate_criterion(criterion_name)
        return get_criterion(criterion_name)

    @staticmethod
    def create_split_finder(
        criterion: BaseCriterion,
        min_samples_leaf: int,
        verbose: int = 0
    ) -> BaseSplitFinder:
        """
        Create split finder instance.

        Parameters
        ----------
        criterion : BaseCriterion
            Split evaluation criterion
        min_samples_leaf : int
            Minimum samples in leaf node
        verbose : int
            Verbosity level

        Returns
        -------
        split_finder : BaseSplitFinder
            Split finder instance
        """
        from .split_finder import SplitFinder
        from .validation_utils import ParameterValidator

        ParameterValidator.validate_positive_integer(min_samples_leaf, "min_samples_leaf")
        ParameterValidator.validate_verbose_level(verbose)

        return SplitFinder(criterion, min_samples_leaf, verbose)

    @staticmethod
    def create_stopping_criteria(
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        verbose: int = 0
    ) -> BaseStoppingCriteria:
        """
        Create stopping criteria instance.

        Parameters
        ----------
        max_depth : int, optional
            Maximum tree depth
        min_samples_split : int
            Minimum samples to split
        min_samples_leaf : int
            Minimum samples in leaf
        verbose : int
            Verbosity level

        Returns
        -------
        stopping_criteria : BaseStoppingCriteria
            Stopping criteria instance
        """
        from .tree_orchestrator import StoppingCriteria
        from .validation_utils import TreeValidator

        TreeValidator.validate_tree_hyperparameters(
            max_depth, min_samples_split, min_samples_leaf, 0.0, 0.2
        )

        return StoppingCriteria(max_depth, min_samples_split, min_samples_leaf, verbose)