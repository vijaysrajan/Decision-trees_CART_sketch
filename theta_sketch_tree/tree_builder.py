"""
TreeBuilder module.

Implements recursive decision tree construction using theta sketches with intersection-based approach.
"""

from typing import Dict, Set, List, Any, Optional
import numpy as np
from numpy.typing import NDArray

from .tree_structure import TreeNode


class TreeBuilder:
    """
    Builds decision tree using theta sketches with intersection-based algorithm.

    This class implements the intersection-based algorithm where:
    - Global feature sketches (loaded from CSV) are passed unchanged to all recursive calls
    - Each node receives parent_sketch_pos/neg (accumulated intersection from root)
    - Binary feature optimization: features in already_used set are skipped
    - Split evaluation is integrated directly into build_tree
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
            Pre-pruning implementation
        feature_mapping : dict, optional
            Maps feature names to column indices
        verbose : int
            Verbosity level
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.pruner = pruner
        self.feature_mapping = feature_mapping or {}
        self.verbose = verbose

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
        Recursively build decision tree using theta sketches.

        Parameters
        ----------
        parent_sketch_pos : ThetaSketch
            Positive class population at this node
        parent_sketch_neg : ThetaSketch
            Negative/all class population at this node
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
        # Use intersection-based algorithm
        return self._build_tree_intersection(
            parent_sketch_pos, parent_sketch_neg,
            sketch_dict, feature_names, already_used, depth
        )

    def _build_tree_intersection(
        self,
        parent_sketch_pos: Any,
        parent_sketch_neg: Any,
        sketch_dict: Dict[str, Dict[str, Any]],
        feature_names: List[str],
        already_used: Set[str],
        depth: int = 0
    ) -> TreeNode:
        """Intersection-based tree building algorithm."""

        # ========== Step 1: Compute THIS NODE's Statistics ==========

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

        if self.verbose >= 2:
            print(f"Depth {depth}: n_samples={n_samples}, impurity={parent_impurity:.4f}, class_counts={class_counts}")

        # ========== Step 2: Check Stopping Conditions ==========

        # Pure node (impurity = 0)
        if parent_impurity == 0:
            node.make_leaf()
            if self.verbose >= 2:
                print(f"Created pure leaf at depth {depth}: prediction={node.prediction}")
            return node

        # Max depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            node.make_leaf()
            if self.verbose >= 2:
                print(f"Created leaf at max depth {depth}: prediction={node.prediction}")
            return node

        # Insufficient samples to split
        if n_samples < self.min_samples_split:
            node.make_leaf()
            if self.verbose >= 2:
                print(f"Created leaf (insufficient samples {n_samples}): prediction={node.prediction}")
            return node

        # Only one class present
        if np.sum(class_counts > 0) <= 1:
            node.make_leaf()
            if self.verbose >= 2:
                print(f"Created leaf (single class): prediction={node.prediction}")
            return node

        # ========== Step 3: Find Best Split ==========

        best_feature = None
        best_score = float('inf')  # Lower is better for all criteria
        best_left_sketch_pos = None
        best_left_sketch_neg = None
        best_right_sketch_pos = None
        best_right_sketch_neg = None

        # Filter features to only those available in sketch dict and not already used
        available_features = [
            f for f in feature_names
            if f not in already_used
            and f in sketch_dict.get('positive', {})
            and f in sketch_dict.get('negative', {})
        ]

        if not available_features:
            node.make_leaf()
            if self.verbose >= 2:
                print(f"Created leaf (no features available): prediction={node.prediction}")
            return node

        for feature_name in available_features:
            # Get feature sketches from the global dictionary
            feature_present_pos = sketch_dict['positive'][feature_name][0]
            feature_absent_pos = sketch_dict['positive'][feature_name][1]
            feature_present_neg = sketch_dict['negative'][feature_name][0]
            feature_absent_neg = sketch_dict['negative'][feature_name][1]

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
                continue

            # Calculate class counts
            left_counts = np.array([left_neg, left_pos])
            right_counts = np.array([right_neg, right_pos])

            # Check minimum samples constraint
            if (left_pos + left_neg < self.min_samples_leaf or
                right_pos + right_neg < self.min_samples_leaf):
                continue

            # Evaluate split using criterion (lower is better)
            score = self.criterion.evaluate_split(class_counts, left_counts, right_counts, parent_impurity)

            if score < best_score:
                best_score = score
                best_feature = feature_name
                best_left_sketch_pos = left_sketch_pos
                best_left_sketch_neg = left_sketch_neg
                best_right_sketch_pos = right_sketch_pos
                best_right_sketch_neg = right_sketch_neg

        # ========== Step 4: Apply Best Split or Create Leaf ==========

        if best_feature is None:
            node.make_leaf()
            if self.verbose >= 2:
                print(f"No valid split found at depth {depth}, creating leaf")
            return node

        # Configure internal node
        node.is_leaf = False
        node.feature_name = best_feature
        node.feature_idx = self.feature_mapping.get(best_feature, -1)

        if self.verbose >= 2:
            print(f"Best split at depth {depth}: feature='{best_feature}', score={best_score:.4f}")

        # Update already_used set for children
        already_used_for_children = already_used.copy()
        already_used_for_children.add(best_feature)

        # Recursively build children
        node.left = self._build_tree_intersection(
            best_left_sketch_pos, best_left_sketch_neg,
            sketch_dict, feature_names, already_used_for_children, depth + 1
        )

        node.right = self._build_tree_intersection(
            best_right_sketch_pos, best_right_sketch_neg,
            sketch_dict, feature_names, already_used_for_children, depth + 1
        )

        # Set parent references
        node.left.parent = node
        node.right.parent = node

        return node

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