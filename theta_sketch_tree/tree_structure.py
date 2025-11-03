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

    Attributes
    ----------
    depth : int
        Depth of this node in tree (root = 0)
    n_samples : float
        Number of samples at this node (from sketch estimates)
    class_counts : NDArray
        Count of samples per class [n_class_0, n_class_1]
    impurity : float
        Impurity value at this node
    parent : Optional[TreeNode]
        Parent node (None for root node)
    """

    def __init__(
        self,
        depth: int,
        n_samples: float,
        class_counts: NDArray,
        impurity: float,
        parent: Optional["TreeNode"] = None,
    ):
        """
        Initialize tree node.

        Parameters
        ----------
        depth : int
            Depth of this node in tree (root = 0)
        n_samples : float
            Number of samples at this node
        class_counts : NDArray
            Count of samples per class [n_class_0, n_class_1]
        impurity : float
            Impurity value at this node
        parent : Optional[TreeNode], default=None
            Parent node (None for root node)
        """
        self.depth = depth
        self.n_samples = n_samples
        self.class_counts = class_counts
        self.impurity = impurity
        self.parent = parent

        # Internal node attributes (set via set_split)
        self.feature_idx: Optional[int] = None
        self.feature_name: Optional[str] = None
        self.left: Optional[TreeNode] = None
        self.right: Optional[TreeNode] = None

        # Leaf node attributes (set via make_leaf)
        self.is_leaf: bool = False
        self.prediction: Optional[int] = None
        self.probabilities: Optional[NDArray] = None

    def make_leaf(self) -> None:
        """
        Convert this node to a leaf.

        Sets is_leaf=True, computes prediction and probabilities.
        """
        self.is_leaf = True
        self.prediction = int(np.argmax(self.class_counts))
        total = np.sum(self.class_counts)
        self.probabilities = self.class_counts / total if total > 0 else np.array([0.5, 0.5])

    def set_split(
        self, feature_idx: int, feature_name: str, left_child: "TreeNode", right_child: "TreeNode"
    ) -> None:
        """
        Set split information for internal node.

        IMPORTANT: Automatically sets parent references on children:
            - left_child.parent = self
            - right_child.parent = self
        This enables upward tree traversal for pruning and other operations.

        Parameters
        ----------
        feature_idx : int
            Column index in X for this feature
        feature_name : str
            Name of the feature (e.g., "age>30")
        left_child : TreeNode
            Node for samples where feature=False
        right_child : TreeNode
            Node for samples where feature=True
        """
        self.feature_idx = feature_idx
        self.feature_name = feature_name

        # Set parent references on children for upward traversal
        left_child.parent = self
        right_child.parent = self

        self.left = left_child
        self.right = right_child


class Tree:
    """
    Complete decision tree structure.
    """

    def __init__(self, root: TreeNode):
        """Initialize tree with root node."""
        # TODO: Implement as per docs/02_low_level_design.md
        pass
