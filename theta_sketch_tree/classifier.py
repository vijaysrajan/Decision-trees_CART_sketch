"""
Refactored classifier implementation with integrated functionality.

This module contains the simplified ThetaSketchDecisionTreeClassifier class
following clean architecture principles:
- Core sklearn API (~200 lines)
- Integrated tree traversal (was tree_traverser.py)
- Integrated feature importance (was feature_importance.py)
- Delegates to ModelPersistence for save/load
- Removes validation optimizer bloat
"""

from typing import Dict, Optional, Union, Tuple, Any, List
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin

from .tree_structure import TreeNode
from .model_persistence import ModelPersistence
from .feature_importance import compute_feature_importances
from .tree_traverser import TreeTraverser


class ThetaSketchDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    CART Decision Tree Classifier trained on theta sketches.

    This classifier trains on theta sketches but performs inference on
    binary tabular data (0/1 features).

    Parameters
    ----------
    criterion : str, default='gini'
        Split criterion: 'gini', 'entropy', 'gain_ratio', 'binomial', 'chi_square'
    max_depth : int or None, default=None
        Maximum tree depth
    min_samples_split : int, default=2
        Minimum samples to split internal node
    min_samples_leaf : int, default=1
        Minimum samples in leaf node
    pruning : str, default='none'
        Pruning method: 'none', 'validation', 'cost_complexity', 'reduced_error', 'min_impurity'
    min_impurity_decrease : float, default=0.0
        Minimum impurity decrease required to keep a split (for min_impurity pruning)
    validation_fraction : float, default=0.2
        Fraction of training data to use for validation-based pruning
    verbose : int, default=0
        Verbosity level
    random_state : int, default=None
        Random seed for reproducible results

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
    tree_ : TreeNode
        The underlying tree structure
    feature_importances_ : ndarray
        Feature importance scores
    """

    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        pruning="none",
        min_impurity_decrease=0.0,
        validation_fraction=0.2,
        verbose=0,
        random_state=None,
    ):
        # Store parameters
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.pruning = pruning
        self.min_impurity_decrease = min_impurity_decrease
        self.validation_fraction = validation_fraction
        self.verbose = verbose
        self.random_state = random_state

        # Initialize fitted state
        self._is_fitted = False

    def fit(
        self,
        sketch_data: Dict[str, Dict[str, Union[Any, Tuple[Any, Any]]]],
        feature_mapping: Dict[str, int],
        sample_weight: Optional[NDArray] = None,
        X_val: Optional[NDArray] = None,
        y_val: Optional[NDArray] = None,
    ):
        """
        Build decision tree from theta sketch data.

        Parameters
        ----------
        sketch_data : dict
            Dictionary with keys 'positive' and 'negative', each containing:
            - 'total': ThetaSketch for the class population (required)
            - '<feature_name>': Tuple (sketch_feature_present, sketch_feature_absent)

        feature_mapping : dict
            Maps feature names to column indices for inference.
            Example: {'age>30': 0, 'income>50k': 1, 'has_diabetes': 2}

        sample_weight : array-like, optional
            Sample weights (not used in sketch-based training)

        X_val : array-like, optional
            Validation data for pruning

        y_val : array-like, optional
            Validation targets for pruning

        Returns
        -------
        self : ThetaSketchDecisionTreeClassifier
        """
        # Validate input
        self._validate_sketch_data(sketch_data)

        # Extract feature names
        feature_names = [k for k in sketch_data['positive'].keys() if k != 'total']

        if not feature_names:
            raise ValueError("No features found in sketch_data")

        # Initialize components
        from .criteria import get_criterion
        criterion = get_criterion(self.criterion)

        # Build tree
        from .tree_builder import TreeBuilder
        tree_builder = TreeBuilder(
            criterion=criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            pruner=None,
            feature_mapping=feature_mapping,
            verbose=self.verbose
        )

        if self.verbose >= 1:
            print("Building decision tree...")
            print(f"Features: {len(feature_names)}")
            print(f"Criterion: {self.criterion}")
            print("Building with intersection approach")

        # Build tree using intersection-based approach
        root_node = tree_builder.build_tree(
            parent_sketch_pos=sketch_data['positive']['total'],
            parent_sketch_neg=sketch_data['negative']['total'],
            sketch_dict=sketch_data,
            feature_names=feature_names,
            already_used=set(),
            depth=0
        )

        # Apply pruning if enabled
        if self.pruning != "none":
            if self.verbose >= 1:
                print(f"Applying {self.pruning} pruning...")

            from .pruning import prune_tree, get_pruning_summary
            from .tree_builder import TreeBuilder

            # Count nodes before pruning
            nodes_before = TreeBuilder.count_tree_nodes(root_node)

            # Apply pruning
            root_node = prune_tree(
                tree_root=root_node,
                method=self.pruning,
                min_impurity_decrease=self.min_impurity_decrease,
                X_val=X_val,
                y_val=y_val,
                feature_mapping=feature_mapping
            )

            if self.verbose >= 1:
                nodes_after = TreeBuilder.count_tree_nodes(root_node)
                summary = get_pruning_summary(self.pruning, nodes_before, nodes_after)
                print(f"Pruning complete: {summary['nodes_removed']} nodes removed")
                print(f"Compression ratio: {summary['compression_ratio']:.3f}")

        # Set sklearn attributes
        self.classes_ = np.array([0, 1])
        self.n_classes_ = 2
        self.n_features_in_ = len(feature_names)
        self.feature_names_in_ = np.array(feature_names)
        self.tree_ = root_node

        # Store internal state
        self._sketch_dict = sketch_data
        self._feature_mapping = feature_mapping
        self._is_fitted = True

        # Compute feature importances
        self._feature_importances = compute_feature_importances(
            self.tree_, self.feature_names_in_, self.n_features_in_
        )

        if self.verbose >= 1:
            print("Tree built successfully")
            if self.verbose >= 2:
                print("Feature importances:")
                for i, feature_name in enumerate(self.feature_names_in_):
                    importance = self._feature_importances[i]
                    print(f"  {feature_name}: {importance:.4f}")

        return self

    def predict(self, X: NDArray) -> NDArray:
        """
        Predict class labels for binary feature data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Binary feature matrix (0/1 values).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError("Classifier must be fitted before making predictions. Call fit() first.")
        X = self._validate_input(X)

        traverser = TreeTraverser(self.tree_)
        return traverser.predict(X)

    def predict_proba(self, X: NDArray) -> NDArray:
        """
        Predict class probabilities for binary feature data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Binary feature matrix (0/1 values).

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Class probabilities. For binary classification:
            - Column 0: P(class=0)
            - Column 1: P(class=1)
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError("Classifier must be fitted before making predictions. Call fit() first.")
        X = self._validate_input(X)

        traverser = TreeTraverser(self.tree_)
        return traverser.predict_proba(X)

    @property
    def feature_importances_(self) -> NDArray:
        """
        Feature importances (sklearn-compatible property).

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            Feature importances (normalized to sum to 1.0)
        """
        self._check_is_fitted()
        return self._feature_importances

    def get_feature_importance_dict(self) -> Dict[str, float]:
        """
        Get feature importances as a dictionary.

        Returns
        -------
        dict
            Mapping from feature names to importance scores
        """
        self._check_is_fitted()
        return {
            feature_name: float(importance)
            for feature_name, importance in zip(self.feature_names_in_, self._feature_importances)
        }

    def get_top_features(self, top_k: int = 5) -> List[tuple]:
        """
        Get the top k most important features.

        Parameters
        ----------
        top_k : int, default=5
            Number of top features to return

        Returns
        -------
        list of tuple
            List of (feature_name, importance) tuples sorted by importance
        """
        self._check_is_fitted()

        # Create list of (feature_name, importance) tuples
        feature_importance_pairs = [
            (feature_name, importance)
            for feature_name, importance in zip(self.feature_names_in_, self._feature_importances)
        ]

        # Sort by importance (descending)
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        return feature_importance_pairs[:top_k]

    # ========== Delegation Methods ==========

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
        """
        from .classifier_utils import ClassifierUtils
        return ClassifierUtils.fit_from_csv(
            positive_csv, negative_csv, config_path, csv_path, cls
        )

    def save_model(self, filepath: str, include_sketches: bool = False) -> None:
        """Save the trained model to disk."""
        ModelPersistence.save_model(self, filepath, include_sketches)

    @classmethod
    def load_model(cls, filepath: str) -> "ThetaSketchDecisionTreeClassifier":
        """Load a trained model from disk."""
        return ModelPersistence.load_model(filepath)

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the fitted model."""
        return ModelPersistence.get_model_info(self)

    # ========== Private Methods ==========

    def _validate_sketch_data(self, sketch_data: Dict) -> None:
        """Validate sketch_data structure."""
        if 'positive' not in sketch_data:
            raise ValueError("sketch_data must contain 'positive' key")
        if 'negative' not in sketch_data:
            raise ValueError("sketch_data must contain 'negative' key")
        if 'total' not in sketch_data['positive']:
            raise ValueError("sketch_data['positive'] must contain 'total' key")
        if 'total' not in sketch_data['negative']:
            raise ValueError("sketch_data['negative'] must contain 'total' key")

    def _check_is_fitted(self) -> None:
        """Check if classifier is fitted."""
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError("Classifier must be fitted before accessing feature importances")

    def _validate_input(self, X: NDArray) -> NDArray:
        """Validate and convert input data."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but classifier was fitted with {self.n_features_in_} features")
        return X

