"""
Centralized logging utilities for theta sketch tree.

Provides consistent logging interface across all components.
"""

import logging
import sys
from typing import Optional, Any


class TreeLogger:
    """
    Centralized logger for theta sketch tree operations.

    Provides verbose-level based logging with consistent formatting
    and component identification.
    """

    def __init__(self, component_name: str, verbose: int = 0):
        """
        Initialize logger for component.

        Parameters
        ----------
        component_name : str
            Name of the component (e.g., 'SplitFinder', 'TreeOrchestrator')
        verbose : int
            Verbosity level (0=silent, 1=info, 2=detailed, 3=debug)
        """
        self.component_name = component_name
        self.verbose = verbose

        # Set up Python logger
        self.logger = logging.getLogger(f"theta_sketch_tree.{component_name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                f'[{component_name}] %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

    def info(self, message: str, level: int = 1) -> None:
        """Log info message at specified verbose level."""
        if self.verbose >= level:
            self.logger.info(message)

    def debug(self, message: str, level: int = 2) -> None:
        """Log debug message at specified verbose level."""
        if self.verbose >= level:
            self.logger.debug(message)

    def trace(self, message: str, level: int = 3) -> None:
        """Log trace message at specified verbose level."""
        if self.verbose >= level:
            self.logger.debug(f"TRACE: {message}")

    def log_split_evaluation(self, feature_name: str, score: float, level: int = 2) -> None:
        """Log split evaluation details."""
        self.debug(f"Evaluating split on '{feature_name}': score={score:.4f}", level)

    def log_node_creation(self, depth: int, n_samples: float, class_counts: Any,
                         impurity: float, level: int = 2) -> None:
        """Log node creation details."""
        self.debug(f"Depth {depth}: n_samples={n_samples}, impurity={impurity:.4f}, "
                  f"class_counts={class_counts}", level)

    def log_leaf_creation(self, depth: int, reason: str, prediction: Any, level: int = 2) -> None:
        """Log leaf node creation."""
        self.debug(f"Created leaf at depth {depth} ({reason}): prediction={prediction}", level)

    def log_best_split(self, depth: int, feature_name: str, score: float, level: int = 2) -> None:
        """Log best split selection."""
        self.debug(f"Best split at depth {depth}: feature='{feature_name}', score={score:.4f}", level)

    def log_tree_building(self, n_features: int, criterion: str, level: int = 1) -> None:
        """Log tree building start."""
        self.info(f"Building decision tree with {n_features} features using {criterion} criterion", level)

    def log_pruning(self, method: str, nodes_before: int, nodes_after: int, level: int = 1) -> None:
        """Log pruning results."""
        nodes_removed = nodes_before - nodes_after
        compression_ratio = nodes_after / nodes_before if nodes_before > 0 else 1.0
        self.info(f"Pruning ({method}): {nodes_removed} nodes removed, "
                 f"compression ratio: {compression_ratio:.3f}", level)