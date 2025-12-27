"""
Tree construction orchestrator for theta sketch decision trees.

This module provides high-level orchestration of tree building,
pruning, and post-processing operations.
"""

from typing import Dict, Set, List, Any, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from .tree_structure import TreeNode
from .split_finder import SplitFinder
from .logging_utils import TreeLogger


class StoppingCriteria:
    """
    Encapsulates tree stopping criteria logic.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        verbose: int = 0
    ):
        """
        Initialize stopping criteria.

        Parameters
        ----------
        max_depth : int, optional
            Maximum tree depth
        min_samples_split : int
            Minimum samples to split a node
        min_samples_leaf : int
            Minimum samples in leaf node
        verbose : int
            Verbosity level
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.verbose = verbose

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
        # Pure node (impurity = 0)
        if impurity == 0:
            return True, "pure_node"

        # Max depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            return True, "max_depth"

        # Insufficient samples to split
        if n_samples < self.min_samples_split:
            return True, "min_samples_split"

        # Only one class present
        if np.sum(class_counts > 0) <= 1:
            return True, "single_class"

        # No features available
        if not available_features:
            return True, "no_features"

        return False, "continue"


class NodeBuilder:
    """
    Builds individual tree nodes from sketch data.
    """

    def __init__(self, criterion: Any, logger: TreeLogger):
        """
        Initialize node builder.

        Parameters
        ----------
        criterion : Criterion
            Split evaluation criterion
        logger : TreeLogger
            Logger instance for output
        """
        self.criterion = criterion
        self.logger = logger

    def create_node(
        self,
        parent_sketch_pos: Any,
        parent_sketch_neg: Any,
        depth: int
    ) -> TreeNode:
        """
        Create a tree node from sketch data.

        Parameters
        ----------
        parent_sketch_pos : ThetaSketch
            Positive class sketch
        parent_sketch_neg : ThetaSketch
            Negative class sketch
        depth : int
            Node depth

        Returns
        -------
        node : TreeNode
            Created tree node
        """
        # Extract class counts from sketches
        pos_count = parent_sketch_pos.get_estimate()
        neg_count = parent_sketch_neg.get_estimate()
        n_samples = pos_count + neg_count
        class_counts = np.array([neg_count, pos_count])

        # Calculate impurity using the provided criterion
        parent_impurity = self.criterion.compute_impurity(class_counts)

        # Create current node
        node = TreeNode(
            depth=depth,
            n_samples=n_samples,
            class_counts=class_counts,
            impurity=parent_impurity
        )

        self.logger.log_node_creation(depth, n_samples, class_counts, parent_impurity)

        return node

    def finalize_leaf_node(self, node: TreeNode, reason: str) -> None:
        """
        Finalize a node as a leaf.

        Parameters
        ----------
        node : TreeNode
            Node to make into leaf
        reason : str
            Reason for making it a leaf
        """
        node.make_leaf()
        self.logger.log_leaf_creation(node.depth, reason, node.prediction)

    def configure_split_node(
        self,
        node: TreeNode,
        split_result,
        feature_mapping: Dict[str, int]
    ) -> None:
        """
        Configure a node for splitting.

        Parameters
        ----------
        node : TreeNode
            Node to configure
        split_result : SplitResult
            Best split information
        feature_mapping : dict
            Maps feature names to indices
        """
        node.is_leaf = False
        node.feature_name = split_result.feature_name
        node.feature_idx = feature_mapping.get(split_result.feature_name, -1)

        self.logger.log_best_split(node.depth, split_result.feature_name, split_result.score)


class TreeOrchestrator:
    """
    Orchestrates the complete tree building process.

    This class coordinates between different components to build
    decision trees in a clean, modular way.
    """

    def __init__(
        self,
        criterion: Any,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        feature_mapping: Optional[Dict[str, int]] = None,
        max_features: Optional[Any] = None,
        random_state: Optional[int] = None,
        verbose: int = 0
    ):
        """
        Initialize tree orchestrator.

        Parameters
        ----------
        criterion : Criterion
            Split evaluation criterion
        max_depth : int, optional
            Maximum tree depth
        min_samples_split : int
            Minimum samples to split a node
        min_samples_leaf : int
            Minimum samples in leaf node
        feature_mapping : dict, optional
            Maps feature names to indices
        max_features : int, float, str, or None, optional
            Number of features to consider for each split
        random_state : int, optional
            Random seed for feature subsampling
        verbose : int
            Verbosity level
        """
        self.logger = TreeLogger("TreeOrchestrator", verbose)
        self.stopping_criteria = StoppingCriteria(
            max_depth, min_samples_split, min_samples_leaf, verbose
        )
        self.split_finder = SplitFinder(criterion, min_samples_leaf, verbose)
        self.node_builder = NodeBuilder(criterion, self.logger)
        self.feature_mapping = feature_mapping or {}
        self.max_features = max_features
        self.random_state = random_state
        self.verbose = verbose

    def build_tree(
        self,
        parent_sketch_pos: Any,
        parent_sketch_neg: Any,
        sketch_dict: Dict[str, Dict[str, Any]],
        feature_names: List[str],
        already_used: Optional[Set[str]] = None,
        depth: int = 0
    ) -> TreeNode:
        """
        Build a decision tree recursively.

        Parameters
        ----------
        parent_sketch_pos : ThetaSketch
            Positive class population at this node
        parent_sketch_neg : ThetaSketch
            Negative class population at this node
        sketch_dict : dict
            Global feature sketches
        feature_names : list
            Available feature names
        already_used : set, optional
            Binary features already used in path from root
        depth : int
            Current depth in tree

        Returns
        -------
        node : TreeNode
            Root of constructed (sub)tree
        """
        if already_used is None:
            already_used = set()

        # Step 1: Create node from sketch data
        node = self.node_builder.create_node(
            parent_sketch_pos, parent_sketch_neg, depth
        )

        # Step 2: Check stopping conditions
        available_features = self.split_finder._get_available_features(
            feature_names, already_used, sketch_dict
        )

        should_stop, reason = self.stopping_criteria.should_stop_splitting(
            depth, node.n_samples, node.impurity, node.class_counts, available_features
        )

        if should_stop:
            self.node_builder.finalize_leaf_node(node, reason)
            return node

        # Step 3: Find best split
        split_result = self.split_finder.find_best_split(
            parent_sketch_pos,
            parent_sketch_neg,
            node.class_counts,
            node.impurity,
            sketch_dict,
            feature_names,
            already_used,
            max_features=self.max_features,
            random_state=self.random_state
        )

        if split_result is None:
            self.node_builder.finalize_leaf_node(node, "no_valid_split")
            return node

        # Step 4: Configure split and build children
        self.node_builder.configure_split_node(node, split_result, self.feature_mapping)

        # Create different already_used sets for left and right children
        # For categorical features (variable=value), handle mutual exclusivity correctly
        already_used_left, already_used_right = self._create_child_already_used_sets(
            already_used, split_result.feature_name, feature_names
        )

        # Recursively build children
        node.left = self.build_tree(
            split_result.left_sketch_pos,
            split_result.left_sketch_neg,
            sketch_dict,
            feature_names,
            already_used_left,
            depth + 1
        )

        node.right = self.build_tree(
            split_result.right_sketch_pos,
            split_result.right_sketch_neg,
            sketch_dict,
            feature_names,
            already_used_right,
            depth + 1
        )

        # Set parent references
        node.left.parent = node
        node.right.parent = node

        return node

    def _create_child_already_used_sets(
        self,
        parent_already_used: Set[str],
        split_feature: str,
        all_features: List[str]
    ) -> Tuple[Set[str], Set[str]]:
        """
        Create appropriate already_used sets for left and right children.

        For categorical features (variable=value format):
        - Left child (absent): Can still use other values of same categorical variable
        - Right child (present): Cannot use ANY value of same categorical variable

        For standalone features:
        - Both children: Just add the current feature to already_used

        Parameters
        ----------
        parent_already_used : set
            Features already used in parent path
        split_feature : str
            Feature used for current split
        all_features : list
            All available feature names

        Returns
        -------
        already_used_left : set
            Features to exclude for left child (absent)
        already_used_right : set
            Features to exclude for right child (present)
        """
        # Start with parent's already_used for both children
        already_used_left = parent_already_used.copy()
        already_used_right = parent_already_used.copy()

        # Add the current split feature to both
        already_used_left.add(split_feature)
        already_used_right.add(split_feature)

        # For categorical features (variable=value), handle mutual exclusivity
        if '=' in split_feature:
            categorical_var = split_feature.split('=')[0]

            # For RIGHT child (present): exclude ALL other values of same categorical variable
            # This ensures mutual exclusivity: if city=Bangalore is TRUE,
            # then city=Mumbai, city=Delhi, etc. are impossible
            for feature in all_features:
                if ('=' in feature and
                    feature.split('=')[0] == categorical_var and
                    feature != split_feature):
                    already_used_right.add(feature)

            # For LEFT child (absent): only exclude the current feature
            # The left child can still use other values of the same categorical variable
            # because if city≠Bangalore, it could still be Mumbai, Delhi, etc.

        # For standalone features (no '='), both children just exclude the current feature
        # No additional exclusions needed

        return already_used_left, already_used_right

    def build_tree_with_postprocessing(
        self,
        parent_sketch_pos: Any,
        parent_sketch_neg: Any,
        sketch_dict: Dict[str, Dict[str, Any]],
        feature_names: List[str],
        pruning_method: str = "none",
        **pruning_kwargs
    ) -> TreeNode:
        """
        Build tree and apply post-processing (pruning, etc.).

        Parameters
        ----------
        parent_sketch_pos : ThetaSketch
            Positive class sketch at root
        parent_sketch_neg : ThetaSketch
            Negative class sketch at root
        sketch_dict : dict
            Global feature sketches
        feature_names : list
            Available feature names
        pruning_method : str
            Pruning method to apply
        **pruning_kwargs
            Additional pruning parameters

        Returns
        -------
        tree_root : TreeNode
            Root of the final tree (after post-processing)
        """
        # Build the initial tree
        tree_root = self.build_tree(
            parent_sketch_pos,
            parent_sketch_neg,
            sketch_dict,
            feature_names
        )

        # Apply pruning if requested
        if pruning_method != "none":
            tree_root = self._apply_pruning(
                tree_root, pruning_method, **pruning_kwargs
            )

        return tree_root

    def _apply_pruning(
        self,
        tree_root: TreeNode,
        method: str,
        **kwargs
    ) -> TreeNode:
        """
        Apply pruning to the tree.

        Parameters
        ----------
        tree_root : TreeNode
            Root of tree to prune
        method : str
            Pruning method
        **kwargs
            Additional pruning parameters

        Returns
        -------
        pruned_root : TreeNode
            Root of pruned tree
        """
        self.logger.info(f"Applying {method} pruning...")

        from .pruning import prune_tree, get_pruning_summary
        from .tree_builder import TreeBuilder

        # Count nodes before pruning
        nodes_before = TreeBuilder.count_tree_nodes(tree_root)

        # Apply pruning
        pruned_root = prune_tree(tree_root, method, **kwargs)

        nodes_after = TreeBuilder.count_tree_nodes(pruned_root)
        self.logger.log_pruning(method, nodes_before, nodes_after)

        return pruned_root


# Convenience function for backward compatibility
def build_tree(
    parent_sketch_pos: Any,
    parent_sketch_neg: Any,
    sketch_dict: Dict[str, Dict[str, Any]],
    feature_names: List[str],
    criterion: Any,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    feature_mapping: Optional[Dict[str, int]] = None,
    verbose: int = 0
) -> TreeNode:
    """
    Convenience function to build a tree.

    Parameters
    ----------
    parent_sketch_pos : ThetaSketch
        Positive class sketch at root
    parent_sketch_neg : ThetaSketch
        Negative class sketch at root
    sketch_dict : dict
        Global feature sketches
    feature_names : list
        Available feature names
    criterion : Criterion
        Split evaluation criterion
    max_depth : int, optional
        Maximum tree depth
    min_samples_split : int
        Minimum samples to split a node
    min_samples_leaf : int
        Minimum samples in leaf node
    feature_mapping : dict, optional
        Maps feature names to indices
    verbose : int
        Verbosity level

    Returns
    -------
    tree_root : TreeNode
        Root of constructed tree
    """
    orchestrator = TreeOrchestrator(
        criterion, max_depth, min_samples_split, min_samples_leaf,
        feature_mapping, verbose
    )
    return orchestrator.build_tree(
        parent_sketch_pos, parent_sketch_neg, sketch_dict, feature_names
    )