"""
Main classifier implementation.

This module contains the ThetaSketchDecisionTreeClassifier class,
the main API for the theta sketch decision tree.
"""

from typing import Dict, Optional, Union, Tuple, Any
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin

# TODO: Implement ThetaSketchDecisionTreeClassifier
# See docs/02_low_level_design.md for detailed specifications
# See docs/05_api_design.md for API documentation


class ThetaSketchDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    CART Decision Tree Classifier trained on theta sketches.

    This classifier trains on theta sketches but performs inference on
    binary tabular data (0/1 features).

    Parameters
    ----------
    criterion : str, default='gini'
        Split criterion: 'gini', 'entropy', 'gain_ratio', 'binomial', 'binomial_chi'
    max_depth : int or None, default=None
        Maximum tree depth
    min_samples_split : int, default=2
        Minimum samples to split internal node
    min_samples_leaf : int, default=1
        Minimum samples in leaf node
    # ... (see full API in docs/05_api_design.md)

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels [0, 1]
    n_classes_ : int
        Number of classes (always 2)
    n_features_in_ : int
        Number of features
    feature_names_in_ : ndarray
        Feature names
    tree_ : Tree
        The underlying tree structure
    feature_importances_ : ndarray
        Feature importance scores

    Examples
    --------
    >>> from theta_sketch_tree import load_sketches, load_config
    >>> # Load data
    >>> sketch_data = load_sketches('target_yes.csv', 'target_no.csv')
    >>> config = load_config('config.yaml')
    >>> # Fit model
    >>> clf = ThetaSketchDecisionTreeClassifier(**config['hyperparameters'])
    >>> clf.fit(sketch_data, config['feature_mapping'])
    >>> predictions = clf.predict(X_test)
    >>>
    >>> # Or use convenience method
    >>> clf = ThetaSketchDecisionTreeClassifier.fit_from_csv(
    ...     positive_csv='target_yes.csv',
    ...     negative_csv='target_no.csv',
    ...     config_path='config.yaml'
    ... )
    >>> predictions = clf.predict(X_test)
    """

    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        # ... more parameters
        verbose=0,
    ):
        # Store parameters
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.verbose = verbose

    def fit(
        self,
        sketch_data: Dict[str, Dict[str, Union[Any, Tuple[Any, Any]]]],
        feature_mapping: Dict[str, int],
        sample_weight: Optional[NDArray] = None,
    ):
        """
        Build decision tree from theta sketch data.

        CRITICAL - Corrected Implementation:
        ====================================
        Uses the corrected algorithm where:
        - sketch_dict loaded ONCE from CSV, passed unchanged to all recursive calls
        - Root uses sketch_dict['positive']['total'] as starting parent_sketch
        - Binary feature optimization via already_used set
        - No nested sketch_dict structures or feature propagation

        Parameters
        ----------
        sketch_data : dict
            Dictionary with keys 'positive' and 'negative' (or 'all'), each containing:
            - 'total': ThetaSketch for the class population (required)
            - '<feature_name>': Tuple (sketch_feature_present, sketch_feature_absent)

        feature_mapping : dict
            Maps feature names to column indices for inference.
            Example: {'age>30': 0, 'income>50k': 1, 'has_diabetes': 2}

        sample_weight : array-like, optional
            Sample weights (not used in sketch-based training)

        Returns
        -------
        self : ThetaSketchDecisionTreeClassifier

        See docs/03_algorithms.md for detailed algorithm specification.
        """
        # ========== Step 1: Validate sketch_data structure ==========
        if 'positive' not in sketch_data:
            raise ValueError("sketch_data must contain 'positive' key")
        if 'negative' not in sketch_data:
            raise ValueError("sketch_data must contain 'negative' key (or 'all' for one-vs-all mode)")

        if 'total' not in sketch_data['positive']:
            raise ValueError("sketch_data['positive'] must contain 'total' key")
        if 'total' not in sketch_data['negative']:
            raise ValueError("sketch_data['negative'] must contain 'total' key")

        # ========== Step 2: Extract feature names ==========
        feature_names = [k for k in sketch_data['positive'].keys() if k != 'total']

        if not feature_names:
            raise ValueError("No features found in sketch_data (besides 'total')")

        # ========== Step 3: Initialize components ==========
        # TODO: Implement proper Criterion classes (gini, entropy, etc.)
        # For now, create a minimal stub
        from .tree_builder import TreeBuilder

        # Minimal criterion stub (TODO: Replace with full implementation)
        class MinimalCriterion:
            def compute_impurity(self, class_counts):
                """Gini impurity."""
                n = np.sum(class_counts)
                if n == 0:
                    return 0.0
                p = class_counts / n
                return 1.0 - np.sum(p ** 2)

            def evaluate_split(self, parent_counts, left_counts, right_counts):
                """Weighted impurity decrease (lower is better)."""
                n_parent = np.sum(parent_counts)
                n_left = np.sum(left_counts)
                n_right = np.sum(right_counts)

                parent_impurity = self.compute_impurity(parent_counts)
                left_impurity = self.compute_impurity(left_counts)
                right_impurity = self.compute_impurity(right_counts)

                weighted_child_impurity = (n_left / n_parent) * left_impurity + \
                                         (n_right / n_parent) * right_impurity

                # Return weighted child impurity (lower is better)
                return weighted_child_impurity

        criterion = MinimalCriterion()

        # TODO: Implement Pruner class
        pruner = None  # No pruning for now

        # ========== Step 4: Build tree using corrected algorithm ==========
        tree_builder = TreeBuilder(
            criterion=criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            pruner=pruner,
            feature_mapping=feature_mapping,
            verbose=self.verbose
        )

        if self.verbose >= 1:
            print("Building decision tree...")
            print(f"Features: {len(feature_names)}")
            print(f"Criterion: {self.criterion}")

        # CRITICAL: Call build_tree with corrected signature
        # - parent_sketch from 'total' key (at root)
        # - sketch_dict passed unchanged
        # - already_used starts as empty set
        root_node = tree_builder.build_tree(
            parent_sketch_pos=sketch_data['positive']['total'],
            parent_sketch_neg=sketch_data['negative']['total'],
            sketch_dict=sketch_data,  # Global features (unchanged for all recursive calls)
            feature_names=feature_names,
            already_used=set(),  # Empty set at root
            depth=0
        )

        # ========== Step 5: Set sklearn attributes ==========
        self.classes_ = np.array([0, 1])
        self.n_classes_ = 2
        self.n_features_in_ = len(feature_names)
        self.feature_names_in_ = np.array(feature_names)
        self.tree_ = root_node  # Store root node (TODO: Wrap in Tree class)

        # Store internal state
        self._sketch_dict = sketch_data
        self._feature_mapping = feature_mapping
        self._is_fitted = True

        # TODO: Implement feature importances calculation
        self._feature_importances = None

        if self.verbose >= 1:
            print(f"Tree built successfully")

        return self

    @classmethod
    def fit_from_csv(
        cls,
        positive_csv: str,
        negative_csv: str,
        config_path: str,
        csv_path: Optional[str] = None,
    ) -> "ThetaSketchDecisionTreeClassifier":
        """
        Convenience method: load sketches from CSV and fit model in one call.

        Parameters
        ----------
        positive_csv : str
            Path to CSV with positive class sketches.
        negative_csv : str
            Path to CSV with negative class sketches.
        config_path : str
            Path to YAML config file.
        csv_path : str, optional
            If provided, uses single CSV mode instead (for backward compatibility).

        Returns
        -------
        clf : ThetaSketchDecisionTreeClassifier
            Fitted classifier.

        Examples
        --------
        >>> clf = ThetaSketchDecisionTreeClassifier.fit_from_csv(
        ...     positive_csv='target_yes.csv',
        ...     negative_csv='target_no.csv',
        ...     config_path='config.yaml'
        ... )
        >>> predictions = clf.predict(X_test)
        """
        # Import here to avoid circular imports
        from theta_sketch_tree import load_sketches, load_config

        # Load data
        if csv_path:
            config = load_config(config_path)
            sketch_data = load_sketches(
                csv_path=csv_path,
                target_positive=config["targets"]["positive"],
                target_negative=config["targets"]["negative"],
            )
        else:
            sketch_data = load_sketches(positive_csv, negative_csv)

        config = load_config(config_path)

        # Initialize with hyperparameters
        clf = cls(**config["hyperparameters"])

        # Fit model
        clf.fit(sketch_data, config["feature_mapping"])

        return clf

    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels for binary feature data."""
        raise NotImplementedError("To be implemented in Week 4")

    def predict_proba(self, X: NDArray) -> NDArray:
        """Predict class probabilities."""
        raise NotImplementedError("To be implemented in Week 4")

    # ... more methods to be implemented
