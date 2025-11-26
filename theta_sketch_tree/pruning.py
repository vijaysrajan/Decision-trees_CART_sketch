"""
Tree Pruning Module.

Implements various pruning strategies to reduce overfitting and improve
model generalization in theta sketch decision trees.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from numpy.typing import NDArray
import copy
from .tree_structure import TreeNode


class TreePruner:
    """
    Advanced tree pruning with multiple strategies.

    Supports:
    - Post-pruning with validation data
    - Cost-complexity pruning (minimal error pruning)
    - Reduced error pruning
    - Rule post-pruning
    """

    def __init__(
        self,
        method: str = "validation",
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        validation_fraction: float = 0.2,
        random_state: Optional[int] = None
    ):
        """
        Initialize tree pruner.

        Parameters
        ----------
        method : str, default="validation"
            Pruning method: "validation", "cost_complexity", "reduced_error", "none"
        min_samples_leaf : int, default=1
            Minimum samples required in leaf after pruning
        min_impurity_decrease : float, default=0.0
            Minimum impurity decrease required to keep a split
        validation_fraction : float, default=0.2
            Fraction of training data to use for validation-based pruning
        random_state : int, optional
            Random seed for reproducible validation splits
        """
        self.method = method
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.validation_fraction = validation_fraction
        self.random_state = random_state

        # Statistics tracking
        self.pruning_stats = {
            'nodes_before': 0,
            'nodes_after': 0,
            'leaves_before': 0,
            'leaves_after': 0,
            'accuracy_before': 0.0,
            'accuracy_after': 0.0
        }

    def prune_tree(
        self,
        tree_root: TreeNode,
        X_val: Optional[NDArray] = None,
        y_val: Optional[NDArray] = None,
        feature_mapping: Optional[Dict[str, int]] = None
    ) -> TreeNode:
        """
        Prune the decision tree using the specified method.

        Parameters
        ----------
        tree_root : TreeNode
            Root node of the tree to prune
        X_val : NDArray, optional
            Validation features for validation-based pruning
        y_val : NDArray, optional
            Validation targets for validation-based pruning
        feature_mapping : dict, optional
            Feature mapping for prediction

        Returns
        -------
        pruned_root : TreeNode
            Root of the pruned tree
        """
        if self.method == "none":
            return tree_root

        # Collect initial statistics
        self._collect_tree_stats(tree_root, before=True)

        # Create a copy to avoid modifying the original
        pruned_tree = self._deep_copy_tree(tree_root)

        if self.method == "validation" and X_val is not None and y_val is not None:
            pruned_tree = self._validation_based_pruning(pruned_tree, X_val, y_val, feature_mapping)
        elif self.method == "cost_complexity":
            pruned_tree = self._cost_complexity_pruning(pruned_tree)
        elif self.method == "reduced_error" and X_val is not None and y_val is not None:
            pruned_tree = self._reduced_error_pruning(pruned_tree, X_val, y_val, feature_mapping)
        elif self.method == "min_impurity":
            pruned_tree = self._min_impurity_pruning(pruned_tree)

        # Collect final statistics
        self._collect_tree_stats(pruned_tree, before=False)

        return pruned_tree

    def _validation_based_pruning(
        self,
        tree_root: TreeNode,
        X_val: NDArray,
        y_val: NDArray,
        feature_mapping: Dict[str, int]
    ) -> TreeNode:
        """
        Prune tree using validation accuracy as the criterion.

        Implements bottom-up pruning where we replace subtrees with leaves
        if it improves or doesn't hurt validation accuracy.
        """
        from .tree_traverser import TreeTraverser

        # Get initial accuracy
        traverser = TreeTraverser(tree_root)
        initial_predictions = traverser.predict(X_val)
        initial_accuracy = np.mean(initial_predictions == y_val)

        # Bottom-up pruning
        improved = True
        iterations = 0
        max_iterations = 100  # Prevent infinite loops

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            # Find candidates for pruning (internal nodes with only leaf children)
            candidates = self._find_pruning_candidates(tree_root)

            for candidate in candidates:
                # Try pruning this subtree
                candidate_backup = self._backup_node(candidate)

                # Convert to leaf with majority class
                self._convert_to_leaf(candidate)

                # Test accuracy
                new_predictions = traverser.predict(X_val)
                new_accuracy = np.mean(new_predictions == y_val)

                # Keep pruning if accuracy doesn't decrease significantly
                if new_accuracy >= initial_accuracy - 0.01:  # Allow small decrease
                    improved = True
                    initial_accuracy = new_accuracy
                    print(f"  Pruned subtree at depth {candidate.depth}, accuracy: {new_accuracy:.4f}")
                else:
                    # Restore the subtree
                    self._restore_node(candidate, candidate_backup)

        return tree_root

    def _cost_complexity_pruning(self, tree_root: TreeNode) -> TreeNode:
        """
        Implement cost-complexity pruning (minimal error pruning).

        Finds the optimal subtree by minimizing cost-complexity measure:
        R(T) + alpha * |T|
        where R(T) is the error rate and |T| is the number of leaves.
        """
        # Build sequence of pruned trees with different alpha values
        alpha_sequence = []
        tree_sequence = [self._deep_copy_tree(tree_root)]

        current_tree = self._deep_copy_tree(tree_root)

        while self._count_internal_nodes(current_tree) > 0:
            # Find the subtree with smallest cost-complexity
            best_alpha = float('inf')
            best_node = None

            internal_nodes = self._get_internal_nodes(current_tree)

            for node in internal_nodes:
                # Calculate cost-complexity for this subtree
                subtree_error = self._calculate_subtree_error(node)
                leaf_error = self._calculate_leaf_error(node)
                num_leaves = self._count_leaves_in_subtree(node)

                if num_leaves > 1:
                    alpha = (leaf_error - subtree_error) / (num_leaves - 1)
                    if alpha < best_alpha:
                        best_alpha = alpha
                        best_node = node

            if best_node is not None:
                alpha_sequence.append(best_alpha)
                self._convert_to_leaf(best_node)
                tree_sequence.append(self._deep_copy_tree(current_tree))
            else:
                break

        # For now, return the tree with moderate pruning (middle of sequence)
        if len(tree_sequence) > 1:
            mid_index = len(tree_sequence) // 2
            return tree_sequence[mid_index]

        return tree_root

    def _reduced_error_pruning(
        self,
        tree_root: TreeNode,
        X_val: NDArray,
        y_val: NDArray,
        feature_mapping: Dict[str, int]
    ) -> TreeNode:
        """
        Reduced error pruning: prune subtrees that don't hurt validation accuracy.
        """
        from .tree_traverser import TreeTraverser

        traverser = TreeTraverser(tree_root)

        # Bottom-up traversal
        improved = True
        while improved:
            improved = False
            candidates = self._find_pruning_candidates(tree_root)

            for candidate in candidates:
                # Calculate error before pruning
                predictions_before = traverser.predict(X_val)
                error_before = np.sum(predictions_before != y_val)

                # Temporarily prune
                backup = self._backup_node(candidate)
                self._convert_to_leaf(candidate)

                # Calculate error after pruning
                predictions_after = traverser.predict(X_val)
                error_after = np.sum(predictions_after != y_val)

                # Keep pruning if error doesn't increase
                if error_after <= error_before:
                    improved = True
                    print(f"  Reduced error pruning at depth {candidate.depth}")
                else:
                    # Restore
                    self._restore_node(candidate, backup)

        return tree_root

    def _min_impurity_pruning(self, tree_root: TreeNode) -> TreeNode:
        """
        Prune subtrees that don't provide sufficient impurity decrease.
        """
        def _prune_recursive(node):
            if node.is_leaf:
                return

            # Recursively prune children first
            if node.left:
                _prune_recursive(node.left)
            if node.right:
                _prune_recursive(node.right)

            # Calculate impurity decrease
            if hasattr(node, 'impurity') and node.left and node.right:
                parent_impurity = node.impurity * node.n_samples
                left_impurity = getattr(node.left, 'impurity', 0) * node.left.n_samples
                right_impurity = getattr(node.right, 'impurity', 0) * node.right.n_samples

                weighted_child_impurity = (left_impurity + right_impurity)
                impurity_decrease = parent_impurity - weighted_child_impurity

                # Prune if impurity decrease is too small
                if impurity_decrease < self.min_impurity_decrease * node.n_samples:
                    self._convert_to_leaf(node)
                    print(f"  Min impurity pruning at depth {node.depth}")

        _prune_recursive(tree_root)
        return tree_root

    def _find_pruning_candidates(self, tree_root: TreeNode) -> List[TreeNode]:
        """Find nodes that are candidates for pruning (internal nodes with leaf children)."""
        candidates = []

        def _traverse(node):
            if node.is_leaf:
                return

            # Check if this is a pruning candidate
            left_is_leaf = node.left is None or node.left.is_leaf
            right_is_leaf = node.right is None or node.right.is_leaf

            if left_is_leaf and right_is_leaf:
                candidates.append(node)

            # Continue traversal
            if node.left and not node.left.is_leaf:
                _traverse(node.left)
            if node.right and not node.right.is_leaf:
                _traverse(node.right)

        _traverse(tree_root)
        return candidates

    def _convert_to_leaf(self, node: TreeNode) -> None:
        """Convert an internal node to a leaf with majority class prediction."""
        if hasattr(node, 'class_counts') and node.class_counts is not None:
            # Predict majority class
            node.prediction = np.argmax(node.class_counts)

            # Calculate probabilities
            total_samples = np.sum(node.class_counts)
            if total_samples > 0:
                node.probabilities = node.class_counts / total_samples
            else:
                node.probabilities = np.array([0.5, 0.5])
        else:
            # Default prediction if class counts unavailable
            node.prediction = 0
            node.probabilities = np.array([1.0, 0.0])

        # Remove children and split information
        node.left = None
        node.right = None
        node.feature_name = None
        node.feature_idx = None
        node.is_leaf = True

    def _backup_node(self, node: TreeNode) -> Dict[str, Any]:
        """Create a backup of node state for restoration."""
        return {
            'is_leaf': node.is_leaf,
            'left': node.left,
            'right': node.right,
            'feature_name': getattr(node, 'feature_name', None),
            'feature_idx': getattr(node, 'feature_idx', None),
            'prediction': getattr(node, 'prediction', None),
            'probabilities': getattr(node, 'probabilities', None)
        }

    def _restore_node(self, node: TreeNode, backup: Dict[str, Any]) -> None:
        """Restore node state from backup."""
        node.is_leaf = backup['is_leaf']
        node.left = backup['left']
        node.right = backup['right']
        node.feature_name = backup['feature_name']
        node.feature_idx = backup['feature_idx']
        node.prediction = backup['prediction']
        node.probabilities = backup['probabilities']

    def _deep_copy_tree(self, node: TreeNode) -> TreeNode:
        """Create a deep copy of the tree."""
        return copy.deepcopy(node)

    def _count_internal_nodes(self, tree_root: TreeNode) -> int:
        """Count internal nodes in tree."""
        if tree_root is None or tree_root.is_leaf:
            return 0

        count = 1  # This node
        if tree_root.left:
            count += self._count_internal_nodes(tree_root.left)
        if tree_root.right:
            count += self._count_internal_nodes(tree_root.right)

        return count

    def _get_internal_nodes(self, tree_root: TreeNode) -> List[TreeNode]:
        """Get all internal nodes."""
        nodes = []

        def _traverse(node):
            if node is None or node.is_leaf:
                return
            nodes.append(node)
            _traverse(node.left)
            _traverse(node.right)

        _traverse(tree_root)
        return nodes

    def _calculate_subtree_error(self, node: TreeNode) -> float:
        """Calculate error rate for the entire subtree."""
        if node.is_leaf:
            if hasattr(node, 'class_counts') and node.class_counts is not None:
                total = np.sum(node.class_counts)
                if total > 0:
                    majority_count = np.max(node.class_counts)
                    return (total - majority_count) / total
            return 0.0

        # Weighted error from children
        left_error = self._calculate_subtree_error(node.left) if node.left else 0
        right_error = self._calculate_subtree_error(node.right) if node.right else 0

        left_weight = node.left.n_samples if node.left else 0
        right_weight = node.right.n_samples if node.right else 0
        total_weight = left_weight + right_weight

        if total_weight > 0:
            return (left_error * left_weight + right_error * right_weight) / total_weight
        return 0.0

    def _calculate_leaf_error(self, node: TreeNode) -> float:
        """Calculate error if this node becomes a leaf."""
        if hasattr(node, 'class_counts') and node.class_counts is not None:
            total = np.sum(node.class_counts)
            if total > 0:
                majority_count = np.max(node.class_counts)
                return (total - majority_count) / total
        return 0.0

    def _count_leaves_in_subtree(self, node: TreeNode) -> int:
        """Count leaves in subtree rooted at node."""
        if node is None:
            return 0
        if node.is_leaf:
            return 1

        left_leaves = self._count_leaves_in_subtree(node.left)
        right_leaves = self._count_leaves_in_subtree(node.right)
        return left_leaves + right_leaves

    def _collect_tree_stats(self, tree_root: TreeNode, before: bool = True) -> None:
        """Collect statistics about the tree."""
        def _count_nodes(node):
            if node is None:
                return 0, 0
            if node.is_leaf:
                return 1, 1  # total nodes, leaf nodes

            left_total, left_leaves = _count_nodes(node.left)
            right_total, right_leaves = _count_nodes(node.right)

            return 1 + left_total + right_total, left_leaves + right_leaves

        total_nodes, total_leaves = _count_nodes(tree_root)

        if before:
            self.pruning_stats['nodes_before'] = total_nodes
            self.pruning_stats['leaves_before'] = total_leaves
        else:
            self.pruning_stats['nodes_after'] = total_nodes
            self.pruning_stats['leaves_after'] = total_leaves

    def get_pruning_summary(self) -> Dict[str, Any]:
        """Get summary of pruning results."""
        return {
            'method': self.method,
            'nodes_removed': self.pruning_stats['nodes_before'] - self.pruning_stats['nodes_after'],
            'leaves_removed': self.pruning_stats['leaves_before'] - self.pruning_stats['leaves_after'],
            'compression_ratio': self.pruning_stats['nodes_after'] / max(1, self.pruning_stats['nodes_before']),
            'stats': self.pruning_stats.copy()
        }