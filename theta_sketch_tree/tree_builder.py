"""
TreeBuilder module.

Implements recursive decision tree construction using theta sketches.
"""

from typing import Dict, Set, List, Any, Optional
import numpy as np
from numpy.typing import NDArray

from .tree_structure import TreeNode


class TreeBuilder:
    """
    Builds decision tree using theta sketches.

    This class implements the corrected algorithm where:
    - Global feature sketches (loaded from CSV) are passed unchanged to all recursive calls
    - Each node receives parent_sketch_pos/neg (accumulated intersection from root)
    - Binary feature optimization: features in already_used set are skipped
    - Split evaluation is integrated directly into build_tree (no separate find_best_split)
    """

    def __init__(
        self,
        criterion: Any,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        pruner: Optional[Any] = None,
        feature_mapping: Optional[Dict[str, int]] = None,
        mode: str = "intersection",
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
        mode : str, default="intersection"
            Tree building algorithm: "intersection" or "ratio_based"
        verbose : int
            Verbosity level
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.pruner = pruner
        self.feature_mapping = feature_mapping or {}
        self.mode = mode
        self.verbose = verbose

        # Validate mode parameter
        if self.mode not in ["intersection", "ratio_based"]:
            raise ValueError(f"Unknown mode: {self.mode}. Must be 'intersection' or 'ratio_based'")

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

        CRITICAL - Corrected Algorithm:
        ================================
        - parent_sketch_pos/neg: Accumulated intersection from root to this node
        - sketch_dict: Global feature sketches (loaded from CSV, never changes)
        - already_used: Binary features already used in path from root
        - Split evaluation is integrated directly (no separate find_best_split)

        Parameters
        ----------
        parent_sketch_pos : ThetaSketch
            Positive class population at this node
        parent_sketch_neg : ThetaSketch
            Negative/all class population at this node
        sketch_dict : dict
            Global feature sketches: {'positive': {...}, 'negative': {...}}
        feature_names : list
            All feature names to consider
        already_used : set
            Features already used in path from root
        depth : int
            Current depth (root = 0)

        Returns
        -------
        node : TreeNode
            Root of (sub)tree
        """
        # Delegate to appropriate algorithm based on mode
        if self.mode == "ratio_based":
            return self._build_tree_ratio_based(
                parent_pos_count, parent_neg_count,
                sketch_dict, feature_names, already_used, depth
            )
        elif self.mode == "intersection":
            return self._build_tree_intersection(
                parent_sketch_pos, parent_sketch_neg,
                sketch_dict, feature_names, already_used, depth
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _build_tree_intersection(
        self,
        parent_sketch_pos: Any,
        parent_sketch_neg: Any,
        sketch_dict: Dict[str, Dict[str, Any]],
        feature_names: List[str],
        already_used: Set[str],
        depth: int = 0
    ) -> TreeNode:
        """Original intersection-based tree building algorithm."""

        # ========== Step 1: Compute THIS NODE's Statistics ==========
        n_pos_parent = parent_sketch_pos.get_estimate()
        n_neg_parent = parent_sketch_neg.get_estimate()
        n_total = n_pos_parent + n_neg_parent
        parent_counts = np.array([n_neg_parent, n_pos_parent])

        # Compute this node's impurity
        parent_impurity = self.criterion.compute_impurity(parent_counts)

        # Create node object
        node = TreeNode(
            depth=depth,
            n_samples=n_total,
            class_counts=parent_counts,
            impurity=parent_impurity,
            parent=None  # Set later via set_split()
        )

        if self.verbose >= 2:
            print(f"Depth {depth}: n_samples={n_total:.0f}, impurity={parent_impurity:.4f}, "
                  f"class_counts={parent_counts}")

        # ========== Step 2: Check Stopping Criteria ==========
        should_stop = self._should_stop_splitting(
            n_samples=n_total,
            depth=depth,
            impurity=parent_impurity,
            class_counts=parent_counts
        )

        if should_stop:
            node.make_leaf()
            if self.verbose >= 2:
                print(f"Created leaf at depth {depth}: prediction={node.prediction}")
            return node

        # ========== Step 3: Find Best Split ==========
        best_score = float('inf')
        best_feature = None
        best_left_sketch_pos = None
        best_left_sketch_neg = None
        best_right_sketch_pos = None
        best_right_sketch_neg = None

        # Loop through all features to find best split
        for feature_name in feature_names:
            # OPTIMIZATION: Skip features already used (binary feature optimization)
            if feature_name in already_used:
                continue

            # Get global feature sketches
            if feature_name not in sketch_dict['positive'] or \
               feature_name not in sketch_dict['negative']:
                continue

            feature_present_pos, feature_absent_pos = sketch_dict['positive'][feature_name]
            feature_present_neg, feature_absent_neg = sketch_dict['negative'][feature_name]

            # Compute child sketches by intersecting parent with global features
            # NOTE: At level 1, this intersection is redundant but done for uniformity
            left_sketch_pos = parent_sketch_pos.intersection(feature_absent_pos)
            left_sketch_neg = parent_sketch_neg.intersection(feature_absent_neg)

            right_sketch_pos = parent_sketch_pos.intersection(feature_present_pos)
            right_sketch_neg = parent_sketch_neg.intersection(feature_present_neg)

            # Compute child statistics
            n_pos_left = left_sketch_pos.get_estimate()
            n_neg_left = left_sketch_neg.get_estimate()
            n_pos_right = right_sketch_pos.get_estimate()
            n_neg_right = right_sketch_neg.get_estimate()

            left_counts = np.array([n_neg_left, n_pos_left])
            right_counts = np.array([n_neg_right, n_pos_right])

            # Check for invalid split (empty child or violates min_samples_leaf)
            left_n_samples = np.sum(left_counts)
            right_n_samples = np.sum(right_counts)

            if (left_n_samples == 0 or right_n_samples == 0 or
                left_n_samples < self.min_samples_leaf or
                right_n_samples < self.min_samples_leaf):
                continue

            # Evaluate split quality using criterion
            # Pass parent_impurity to avoid recomputation
            score = self.criterion.evaluate_split(
                parent_counts=parent_counts,
                left_counts=left_counts,
                right_counts=right_counts,
                parent_impurity=parent_impurity
            )

            if self.verbose >= 3:
                print(f"  Feature '{feature_name}': score={score:.4f}, "
                      f"left={left_counts}, right={right_counts}")

            # Check if this is best split so far
            if score < best_score:  # Lower is better for most criteria
                best_score = score
                best_feature = feature_name
                best_left_sketch_pos = left_sketch_pos
                best_left_sketch_neg = left_sketch_neg
                best_right_sketch_pos = right_sketch_pos
                best_right_sketch_neg = right_sketch_neg

        # Check if no valid split found
        if best_feature is None:
            node.make_leaf()
            if self.verbose >= 2:
                print(f"No valid split found at depth {depth}, creating leaf")
            return node

        if self.verbose >= 2:
            print(f"Best split at depth {depth}: feature='{best_feature}', score={best_score:.4f}")

        # ========== Step 4: Check Pre-Pruning ==========
        if self.pruner is not None:
            # Compute impurity decrease for pruning decision
            n_left_pos = best_left_sketch_pos.get_estimate()
            n_left_neg = best_left_sketch_neg.get_estimate()
            n_right_pos = best_right_sketch_pos.get_estimate()
            n_right_neg = best_right_sketch_neg.get_estimate()

            n_left = n_left_pos + n_left_neg
            n_right = n_right_pos + n_right_neg

            left_counts = np.array([n_left_neg, n_left_pos])
            right_counts = np.array([n_right_neg, n_right_pos])

            left_impurity = self.criterion.compute_impurity(left_counts)
            right_impurity = self.criterion.compute_impurity(right_counts)

            weighted_child_impurity = (n_left / n_total) * left_impurity + \
                                     (n_right / n_total) * right_impurity
            impurity_decrease = parent_impurity - weighted_child_impurity

            if self.pruner.should_prune(node, best_feature, impurity_decrease):
                node.make_leaf()
                if self.verbose >= 2:
                    print(f"Pre-pruning at depth {depth}: "
                          f"impurity_decrease={impurity_decrease:.4f} < threshold")
                return node

        # ========== Step 5: Update already_used Set ==========
        already_used_for_children = already_used.copy()
        already_used_for_children.add(best_feature)

        # ========== Step 6: Recurse with Best Feature's Sketches ==========
        # LEFT child (feature = False)
        left_child = self.build_tree(
            parent_sketch_pos=best_left_sketch_pos,
            parent_sketch_neg=best_left_sketch_neg,
            sketch_dict=sketch_dict,  # Same global dict
            feature_names=feature_names,
            already_used=already_used_for_children,  # Updated set
            depth=depth + 1
        )

        # RIGHT child (feature = True)
        right_child = self.build_tree(
            parent_sketch_pos=best_right_sketch_pos,
            parent_sketch_neg=best_right_sketch_neg,
            sketch_dict=sketch_dict,  # Same global dict
            feature_names=feature_names,
            already_used=already_used_for_children,  # Updated set
            depth=depth + 1
        )

        # ========== Step 7: Set Split on This Node ==========
        feature_idx = self.feature_mapping.get(best_feature, -1)
        node.set_split(
            feature_idx=feature_idx,
            feature_name=best_feature,
            left_child=left_child,
            right_child=right_child
        )

        return node

    def _should_stop_splitting(
        self,
        n_samples: float,
        depth: int,
        impurity: float,
        class_counts: NDArray
    ) -> bool:
        """
        Check stopping criteria.

        Parameters computed from parent_sketch_pos and parent_sketch_neg.
        """
        # Pure node (impurity = 0)
        if impurity == 0:
            return True

        # Max depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            return True

        # Insufficient samples to split
        if n_samples < self.min_samples_split:
            return True

        # Only one class present
        if np.sum(class_counts > 0) <= 1:
            return True

        return False

    def _build_tree_ratio_based(
        self,
        parent_pos_count: float,
        parent_neg_count: float,
        sketch_dict: Dict[str, Dict[str, Any]],
        feature_names: List[str],
        already_used: Set[str],
        depth: int = 0
    ) -> TreeNode:
        """
        Build decision tree using ratio-based split estimation.

        This method uses direct feature ratio calculations instead of sketch
        intersections, ensuring perfect sample conservation at each level.
        """
        n_samples = parent_pos_count + parent_neg_count
        class_counts = np.array([parent_neg_count, parent_pos_count])  # [negative, positive]

        if self.verbose >= 2:
            print(f"Building node at depth {depth}: {n_samples:.1f} samples "
                  f"({parent_neg_count:.1f} neg, {parent_pos_count:.1f} pos)")

        # Create node with current statistics
        parent_impurity = self.criterion.compute_impurity(class_counts)
        node = TreeNode(
            depth=depth,
            n_samples=n_samples,
            class_counts=class_counts,
            impurity=parent_impurity
        )

        # Check stopping conditions
        if self._should_stop_splitting_ratio(n_samples, class_counts, depth, parent_impurity):
            node.make_leaf()
            if self.verbose >= 2:
                print(f"Created leaf at depth {depth}: prediction={node.prediction}")
            return node

        # Find best feature split using ratio-based estimation
        best_feature, best_left_pos, best_left_neg, best_right_pos, best_right_neg = (
            self._find_best_ratio_split(
                parent_pos_count, parent_neg_count,
                sketch_dict, feature_names, already_used,
                class_counts, parent_impurity, depth
            )
        )

        # Check if valid split found
        if best_feature is None:
            node.make_leaf()
            if self.verbose >= 2:
                print(f"No valid split found at depth {depth}, creating leaf")
            return node

        # Update used features set for children
        already_used_for_children = already_used.copy()
        already_used_for_children.add(best_feature)

        # Build child nodes recursively using estimated counts
        if self.verbose >= 3:
            print(f"Splitting on '{best_feature}': "
                  f"left({best_left_pos:.1f}+, {best_left_neg:.1f}-), "
                  f"right({best_right_pos:.1f}+, {best_right_neg:.1f}-)")

        left_child = self._build_tree_ratio_based(
            parent_pos_count=best_left_pos,
            parent_neg_count=best_left_neg,
            sketch_dict=sketch_dict,
            feature_names=feature_names,
            already_used=already_used_for_children,
            depth=depth + 1
        )

        right_child = self._build_tree_ratio_based(
            parent_pos_count=best_right_pos,
            parent_neg_count=best_right_neg,
            sketch_dict=sketch_dict,
            feature_names=feature_names,
            already_used=already_used_for_children,
            depth=depth + 1
        )

        # Set split on this node
        feature_idx = self.feature_mapping.get(best_feature, -1)
        node.set_split(
            feature_idx=feature_idx,
            feature_name=best_feature,
            left_child=left_child,
            right_child=right_child
        )

        return node

    def _find_best_ratio_split(
        self,
        parent_pos: float,
        parent_neg: float,
        sketch_dict: Dict[str, Dict[str, Any]],
        feature_names: List[str],
        already_used: Set[str],
        parent_counts: NDArray,
        parent_impurity: float,
        depth: int
    ) -> tuple:
        """Find best feature split using ratio-based estimation."""
        best_score = float('inf')
        best_feature = None
        best_left_pos = best_left_neg = best_right_pos = best_right_neg = 0

        if self.verbose >= 3:
            print(f"Evaluating {len(feature_names)} features at depth {depth}")

        for feature_name in feature_names:
            # Skip already used features
            if feature_name in already_used:
                continue

            # Get global feature sketches
            if (feature_name not in sketch_dict['positive'] or
                feature_name not in sketch_dict['negative']):
                continue

            feature_present_pos, feature_absent_pos = sketch_dict['positive'][feature_name]
            feature_present_neg, feature_absent_neg = sketch_dict['negative'][feature_name]

            # Get feature sketch estimates
            pos_present = feature_present_pos.get_estimate()
            pos_absent = feature_absent_pos.get_estimate()
            neg_present = feature_present_neg.get_estimate()
            neg_absent = feature_absent_neg.get_estimate()

            # Use ratio-based split estimation
            left_pos, left_neg, right_pos, right_neg = self._ratio_based_split(
                parent_pos, parent_neg,
                pos_present, neg_present,
                pos_absent, neg_absent
            )

            # Check min_samples_leaf constraint
            left_n_samples = left_pos + left_neg
            right_n_samples = right_pos + right_neg

            if (left_n_samples < self.min_samples_leaf or
                right_n_samples < self.min_samples_leaf or
                left_n_samples == 0 or right_n_samples == 0):
                continue

            # Evaluate split quality
            left_counts = np.array([left_neg, left_pos])
            right_counts = np.array([right_neg, right_pos])

            score = self.criterion.evaluate_split(
                parent_counts=parent_counts,
                left_counts=left_counts,
                right_counts=right_counts,
                parent_impurity=parent_impurity
            )

            if self.verbose >= 3:
                print(f"  Feature '{feature_name}': score={score:.4f}, "
                      f"left={left_counts}, right={right_counts}")

            # Check if this is the best split so far
            if score < best_score:
                best_score = score
                best_feature = feature_name
                best_left_pos = left_pos
                best_left_neg = left_neg
                best_right_pos = right_pos
                best_right_neg = right_neg

        return (best_feature, best_left_pos, best_left_neg, best_right_pos, best_right_neg)

    @staticmethod
    def _ratio_based_split(
        parent_pos: float, parent_neg: float,
        feature_present_pos: float, feature_present_neg: float,
        feature_absent_pos: float, feature_absent_neg: float
    ) -> tuple:
        """Compute split estimates using direct ratio estimation from feature sketches."""
        # Total feature sketch estimates
        total_feature_present = feature_present_pos + feature_present_neg
        total_feature_absent = feature_absent_pos + feature_absent_neg
        total_feature_estimate = total_feature_present + total_feature_absent

        # Avoid division by zero
        if total_feature_estimate == 0:
            return (0, 0, 0, 0)

        # Compute ratios from feature sketches
        ratio_absent = total_feature_absent / total_feature_estimate
        ratio_present = total_feature_present / total_feature_estimate

        # Estimate child populations using ratios
        left_pos = parent_pos * ratio_absent
        left_neg = parent_neg * ratio_absent
        right_pos = parent_pos * ratio_present
        right_neg = parent_neg * ratio_present

        return (left_pos, left_neg, right_pos, right_neg)

    def _should_stop_splitting_ratio(
        self,
        n_samples: float,
        class_counts: NDArray,
        depth: int,
        impurity: float
    ) -> bool:
        """Check if node splitting should stop (for ratio-based mode)."""
        # Pure node (impurity = 0)
        if impurity == 0:
            return True

        # Max depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            return True

        # Insufficient samples to split
        if n_samples < self.min_samples_split:
            return True

        # Only one class present
        if np.sum(class_counts > 0) <= 1:
            return True

        return False

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

