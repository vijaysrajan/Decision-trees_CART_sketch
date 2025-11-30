"""
TreeBuilder module.

Simplified tree builder that delegates to TreeOrchestrator for actual construction.
Maintains backward compatibility while using improved architecture.
"""

from typing import Dict, Set, List, Any, Optional
import numpy as np
from numpy.typing import NDArray

from .tree_structure import TreeNode
from .tree_orchestrator import TreeOrchestrator


class TreeBuilder:
    """
    Simplified tree builder that delegates to TreeOrchestrator.

    This class maintains backward compatibility while using the new
    modular architecture with separated concerns.
    """

    def __init__(
        self,
        criterion: Any,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        pruner: Optional[Any] = None,
        feature_mapping: Optional[Dict[str, int]] = None,
        verbose: int = 0,
    ):
        """
        Initialize tree builder.

        Parameters
        ----------
        criterion : BaseCriterion
            Split criterion (gini, entropy, etc.)
        max_depth : int, optional
            Maximum tree depth
        min_samples_split : int
            Minimum samples to split a node
        min_samples_leaf : int
            Minimum samples in a leaf
        pruner : Pruner, optional
            Pre-pruning implementation (deprecated, use post-pruning)
        feature_mapping : dict, optional
            Maps feature names to column indices
        verbose : int
            Verbosity level
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.pruner = pruner  # Kept for compatibility, but not used
        self.feature_mapping = feature_mapping or {}
        self.verbose = verbose

        # Create internal orchestrator
        self.orchestrator = TreeOrchestrator(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            feature_mapping=feature_mapping,
            verbose=verbose
        )

    def build_tree(
        self,
        parent_sketch_pos: Any = None,
        parent_sketch_neg: Any = None,
        parent_pos_count: float = None,
        parent_neg_count: float = None,
        sketch_dict: Dict[str, Dict[str, Any]] = None,
        feature_names: List[str] = None,
        already_used: Set[str] = None,
        depth: int = 0
    ) -> TreeNode:
        """
        Build decision tree using theta sketches.

        Parameters
        ----------
        parent_sketch_pos : ThetaSketch
            Positive class population at this node
        parent_sketch_neg : ThetaSketch
            Negative/all class population at this node
        parent_pos_count : float, optional
            Positive count (not used, for compatibility)
        parent_neg_count : float, optional
            Negative count (not used, for compatibility)
        sketch_dict : dict
            Global feature sketches: {'positive': {...}, 'negative': {...}}
        feature_names : list
            Available feature names
        already_used : set
            Binary features already used in path from root
        depth : int
            Current depth in tree

        Returns
        -------
        node : TreeNode
            Root of (sub)tree
        """
        # Update orchestrator parameters if they've changed since initialization
        # This handles cases where test code modifies attributes after __init__
        if (self.orchestrator.stopping_criteria.max_depth != self.max_depth or
            self.orchestrator.stopping_criteria.min_samples_split != self.min_samples_split or
            self.orchestrator.stopping_criteria.min_samples_leaf != self.min_samples_leaf):

            # Recreate orchestrator with current parameters
            self.orchestrator = TreeOrchestrator(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                feature_mapping=self.feature_mapping,
                verbose=self.verbose
            )

        # Delegate to orchestrator for actual tree building
        return self.orchestrator.build_tree(
            parent_sketch_pos=parent_sketch_pos,
            parent_sketch_neg=parent_sketch_neg,
            sketch_dict=sketch_dict,
            feature_names=feature_names,
            already_used=already_used,
            depth=depth
        )


    @staticmethod
    def calculate_tree_depth(root: TreeNode) -> int:
        """Calculate tree depth recursively."""
        def _depth(node):
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))
        return _depth(root)

    @staticmethod
    def count_tree_nodes(root: TreeNode) -> int:
        """Count total nodes in tree."""
        def _count(node):
            if node is None:
                return 0
            return 1 + _count(node.left) + _count(node.right)
        return _count(root)

    @staticmethod
    def count_tree_leaves(root: TreeNode) -> int:
        """Count leaf nodes in tree."""
        def _count_leaves(node):
            if node is None:
                return 0
            if node.is_leaf:
                return 1
            return _count_leaves(node.left) + _count_leaves(node.right)
        return _count_leaves(root)