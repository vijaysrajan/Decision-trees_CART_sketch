"""
Minimal tree pruning functions.

4 core pruning algorithms with no bloat:
- validation_prune: accuracy-based pruning
- cost_complexity_prune: impurity decrease pruning
- reduced_error_prune: error-based pruning
- min_impurity_prune: threshold-based pruning
"""

from typing import Dict, List, Optional
import numpy as np
from numpy.typing import NDArray
import copy
from .tree_structure import TreeNode


def prune_tree(tree_root: TreeNode, method: str = "none", **kwargs) -> TreeNode:
    """Prune tree using specified method."""
    if method == "none":
        return tree_root

    pruned_tree = copy.deepcopy(tree_root)

    if method == "validation":
        return validation_prune(pruned_tree, kwargs.get('X_val'), kwargs.get('y_val'),
                               kwargs.get('feature_mapping'))
    elif method == "cost_complexity":
        return cost_complexity_prune(pruned_tree, kwargs.get('min_impurity_decrease', 0.0))
    elif method == "reduced_error":
        return reduced_error_prune(pruned_tree, kwargs.get('X_val'), kwargs.get('y_val'),
                                  kwargs.get('feature_mapping'))
    elif method == "min_impurity":
        return min_impurity_prune(pruned_tree, kwargs.get('min_impurity_decrease', 0.0))
    else:
        raise ValueError(f"Unknown pruning method: {method}")


def validation_prune(tree_root: TreeNode, X_val: Optional[NDArray], y_val: Optional[NDArray],
                    feature_mapping: Optional[Dict[str, int]]) -> TreeNode:
    """Validation-based pruning using accuracy."""
    if X_val is None or y_val is None:
        return tree_root

    from .tree_traverser import TreeTraverser

    improved = True
    iterations = 0

    while improved and iterations < 20:
        improved = False
        iterations += 1

        # Find internal nodes with leaf children
        candidates = []
        def find_candidates(node):
            if node.is_leaf:
                return
            if node.left.is_leaf and node.right.is_leaf:
                candidates.append(node)
            else:
                find_candidates(node.left)
                find_candidates(node.right)

        find_candidates(tree_root)
        if not candidates:
            break

        # Get current accuracy
        traverser = TreeTraverser(tree_root)
        current_acc = np.mean(traverser.predict(X_val) == y_val)

        # Try pruning each candidate
        for candidate in candidates:
            # Save state
            orig_leaf, orig_left, orig_right = candidate.is_leaf, candidate.left, candidate.right

            # Make leaf
            candidate.is_leaf = True
            candidate.prediction = int(np.argmax(candidate.class_counts))
            candidate.left = candidate.right = None

            # Test accuracy
            new_acc = np.mean(traverser.predict(X_val) == y_val)

            if new_acc >= current_acc:
                improved = True
                break
            else:
                # Restore
                candidate.is_leaf, candidate.left, candidate.right = orig_leaf, orig_left, orig_right

    return tree_root


def cost_complexity_prune(tree_root: TreeNode, min_impurity_decrease: float = 0.0) -> TreeNode:
    """Cost-complexity pruning based on impurity decrease."""
    def prune_recursive(node):
        if node.is_leaf:
            return

        # Recurse first
        prune_recursive(node.left)
        prune_recursive(node.right)

        # Calculate impurity decrease
        if node.left.is_leaf and node.right.is_leaf:
            left_weight = node.left.n_samples / node.n_samples
            right_weight = node.right.n_samples / node.n_samples
            weighted_child_impurity = (left_weight * node.left.impurity +
                                      right_weight * node.right.impurity)
            decrease = node.impurity - weighted_child_impurity

            # Prune if decrease is below threshold
            if decrease < min_impurity_decrease:
                node.is_leaf = True
                node.prediction = int(np.argmax(node.class_counts))
                node.left = node.right = None

    prune_recursive(tree_root)
    return tree_root


def reduced_error_prune(tree_root: TreeNode, X_val: Optional[NDArray], y_val: Optional[NDArray],
                       feature_mapping: Optional[Dict[str, int]]) -> TreeNode:
    """Reduced error pruning using validation error."""
    if X_val is None or y_val is None:
        return tree_root

    from .tree_traverser import TreeTraverser
    traverser = TreeTraverser(tree_root)

    # Find all internal nodes (bottom-up)
    internal_nodes = []
    def collect_internal(node):
        if node.is_leaf:
            return
        collect_internal(node.left)
        collect_internal(node.right)
        internal_nodes.append(node)

    collect_internal(tree_root)

    # Try pruning each node
    for node in internal_nodes:
        if node.is_leaf:  # May have been pruned already
            continue

        # Calculate error before pruning
        error_before = 1.0 - np.mean(traverser.predict(X_val) == y_val)

        # Save state and prune
        orig_leaf, orig_left, orig_right = node.is_leaf, node.left, node.right
        node.is_leaf = True
        node.prediction = int(np.argmax(node.class_counts))
        node.left = node.right = None

        # Calculate error after pruning
        error_after = 1.0 - np.mean(traverser.predict(X_val) == y_val)

        # Keep pruning if error doesn't increase
        if error_after > error_before:
            # Restore
            node.is_leaf, node.left, node.right = orig_leaf, orig_left, orig_right

    return tree_root


def min_impurity_prune(tree_root: TreeNode, min_impurity_decrease: float = 0.0) -> TreeNode:
    """Min impurity decrease pruning."""
    def prune_recursive(node):
        if node.is_leaf:
            return

        # Recurse first
        prune_recursive(node.left)
        prune_recursive(node.right)

        # Check if should prune based on impurity
        if (hasattr(node, 'impurity') and node.impurity < min_impurity_decrease):
            node.is_leaf = True
            node.prediction = int(np.argmax(node.class_counts))
            node.left = node.right = None

    prune_recursive(tree_root)
    return tree_root


def get_pruning_summary(method: str, nodes_before: int, nodes_after: int) -> Dict:
    """Get minimal pruning summary."""
    return {
        'method': method,
        'nodes_removed': nodes_before - nodes_after,
        'compression_ratio': nodes_after / max(1, nodes_before)
    }