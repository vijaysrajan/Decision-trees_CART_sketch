"""
Tree data structures.

This module contains TreeNode and Tree classes for storing
the decision tree structure.
"""

from typing import Optional, Dict, Any
import numpy as np
from numpy.typing import NDArray


class TreeNode:
    """
    Single node in the decision tree.

    Can be either an internal node (with split) or a leaf node (with prediction).
    """

    def __init__(self, depth: int, n_samples: float, class_counts: NDArray, impurity: float):
        """Initialize tree node."""
        # TODO: Implement as per docs/02_low_level_design.md
        pass

    def make_leaf(self) -> None:
        """Convert this node to a leaf."""
        # TODO: Implement
        pass

    def set_split(
        self, feature_idx: int, feature_name: str, left_child: "TreeNode", right_child: "TreeNode"
    ) -> None:
        """Set split information for internal node."""
        # TODO: Implement
        pass


class Tree:
    """
    Complete decision tree structure.
    """

    def __init__(self, root: TreeNode):
        """Initialize tree with root node."""
        # TODO: Implement as per docs/02_low_level_design.md
        pass
