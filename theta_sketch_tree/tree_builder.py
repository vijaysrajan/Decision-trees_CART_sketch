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
        parent_sketch_pos: Any,
        parent_sketch_neg: Any,
        sketch_dict: Dict[str, Dict[str, Any]],
        feature_names: List[str],
        already_used: Set[str],
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

            # Check for invalid split (empty child)
            if np.sum(left_counts) == 0 or np.sum(right_counts) == 0:
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
