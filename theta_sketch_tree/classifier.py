"""
Main classifier implementation.

This module contains the ThetaSketchDecisionTreeClassifier class,
the main API for the theta sketch decision tree.
"""

from typing import Dict, Optional, Union
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
    >>> clf = ThetaSketchDecisionTreeClassifier(max_depth=5)
    >>> clf.fit('sketches.csv', 'config.yaml')
    >>> predictions = clf.predict(X_test)
    """

    def __init__(
        self,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        # ... more parameters
        verbose=0
    ):
        # Store parameters
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.verbose = verbose

    def fit(self, csv_path: str, config_path: str):
        """Train decision tree from theta sketches."""
        raise NotImplementedError("To be implemented in Week 1-4")

    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels for binary feature data."""
        raise NotImplementedError("To be implemented in Week 4")

    def predict_proba(self, X: NDArray) -> NDArray:
        """Predict class probabilities."""
        raise NotImplementedError("To be implemented in Week 4")

    # ... more methods to be implemented
