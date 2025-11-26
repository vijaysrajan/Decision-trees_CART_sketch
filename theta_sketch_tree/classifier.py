"""
Main classifier implementation.

This module contains the ThetaSketchDecisionTreeClassifier class,
the main API for the theta sketch decision tree.
"""

from typing import Dict, Optional, Union, Tuple, Any, List
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
import pickle
import json
import os
from pathlib import Path

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
        tree_builder="intersection",  # New parameter: "intersection" or "ratio_based"
        verbose=0,
    ):
        # Store parameters
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_builder = tree_builder
        self.verbose = verbose

        # Initialize fitted state
        self._is_fitted = False

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
        from .criteria import get_criterion

        # Create criterion instance based on self.criterion parameter
        criterion = get_criterion(self.criterion)

        # TODO: Implement Pruner class
        pruner = None  # No pruning for now

        # ========== Step 4: Build tree using selected algorithm ==========
        from .tree_builder import TreeBuilder
        tree_builder = TreeBuilder(
            criterion=criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            pruner=pruner,
            feature_mapping=feature_mapping,
            mode=self.tree_builder,  # Pass tree_builder mode to TreeBuilder
            verbose=self.verbose
        )

        if self.verbose >= 1:
            print("Building decision tree...")
            print(f"Features: {len(feature_names)}")
            print(f"Criterion: {self.criterion}")

        # Build tree using unified TreeBuilder interface
        if self.verbose >= 1:
            print(f"Building with {self.tree_builder} approach")

        if self.tree_builder == "ratio_based":
            # Ratio-based approach uses count estimates
            pos_total_count = sketch_data['positive']['total'].get_estimate()
            neg_total_count = sketch_data['negative']['total'].get_estimate()

            root_node = tree_builder.build_tree(
                parent_pos_count=pos_total_count,
                parent_neg_count=neg_total_count,
                sketch_dict=sketch_data,
                feature_names=feature_names,
                already_used=set(),
                depth=0
            )
        else:
            # Intersection-based approach uses sketch objects
            root_node = tree_builder.build_tree(
                parent_sketch_pos=sketch_data['positive']['total'],
                parent_sketch_neg=sketch_data['negative']['total'],
                sketch_dict=sketch_data,
                feature_names=feature_names,
                already_used=set(),
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

        # Compute feature importances
        from .feature_importance import compute_feature_importances
        self._feature_importances = compute_feature_importances(
            root=root_node,
            feature_names=list(self.feature_names_in_)
        )

        if self.verbose >= 1:
            print(f"Tree built successfully")
            if self.verbose >= 2:
                print("Feature importances:")
                for i, feature_name in enumerate(self.feature_names_in_):
                    importance = self._feature_importances[i]
                    print(f"  {feature_name}: {importance:.4f}")

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
        """
        Predict class labels for binary feature data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Binary feature matrix (0/1 values).
            Features must match the feature_mapping used during fit().

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).

        Examples
        --------
        >>> X_test = np.array([[1, 0, 1], [0, 1, 0]])  # age>30, income>50k, has_diabetes
        >>> predictions = clf.predict(X_test)
        >>> print(predictions)  # [1, 0]
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError("Classifier must be fitted before making predictions. Call fit() first.")

        # Convert to numpy array and validate
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but classifier was fitted with {self.n_features_in_} features")

        # Use TreeTraverser for prediction
        from .tree_traverser import TreeTraverser
        traverser = TreeTraverser(self.tree_, missing_value_strategy='majority')
        return traverser.predict(X)

    def predict_proba(self, X: NDArray) -> NDArray:
        """
        Predict class probabilities for binary feature data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Binary feature matrix (0/1 values).
            Features must match the feature_mapping used during fit().

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Class probabilities. For binary classification:
            - Column 0: P(class=0)
            - Column 1: P(class=1)

        Examples
        --------
        >>> X_test = np.array([[1, 0, 1], [0, 1, 0]])
        >>> probabilities = clf.predict_proba(X_test)
        >>> print(probabilities)  # [[0.2, 0.8], [0.9, 0.1]]
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError("Classifier must be fitted before making predictions. Call fit() first.")

        # Convert to numpy array and validate
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but classifier was fitted with {self.n_features_in_} features")

        # Use TreeTraverser for prediction
        from .tree_traverser import TreeTraverser
        traverser = TreeTraverser(self.tree_, missing_value_strategy='majority')
        return traverser.predict_proba(X)

    @property
    def feature_importances_(self) -> NDArray:
        """
        Feature importances (sklearn-compatible property).

        The importance of a feature is computed as the (normalized) total reduction
        of impurity brought by that feature. Features that do not appear in the
        tree have zero importance.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            Feature importances (normalized to sum to 1.0)

        Examples
        --------
        >>> clf.fit(sketch_data, feature_mapping)
        >>> importances = clf.feature_importances_
        >>> print(importances)  # [0.7, 0.3]
        """
        if not self._is_fitted:
            raise ValueError("Classifier must be fitted before accessing feature importances")

        return self._feature_importances

    def get_feature_importance_dict(self) -> Dict[str, float]:
        """
        Get feature importances as a dictionary mapping feature names to importance scores.

        Returns
        -------
        dict
            Mapping from feature names to importance scores

        Examples
        --------
        >>> clf.fit(sketch_data, feature_mapping)
        >>> importance_dict = clf.get_feature_importance_dict()
        >>> print(importance_dict)
        {'age>30': 0.7, 'income>50k': 0.3}
        """
        if not self._is_fitted:
            raise ValueError("Classifier must be fitted before accessing feature importances")

        from .feature_importance import FeatureImportanceCalculator
        calculator = FeatureImportanceCalculator(list(self.feature_names_in_))
        return calculator.get_feature_importance_dict(self._feature_importances)

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

        Examples
        --------
        >>> clf.fit(sketch_data, feature_mapping)
        >>> top_features = clf.get_top_features(top_k=3)
        >>> print(top_features)
        [('age>30', 0.7), ('income>50k', 0.3)]
        """
        if not self._is_fitted:
            raise ValueError("Classifier must be fitted before accessing feature importances")

        from .feature_importance import FeatureImportanceCalculator
        calculator = FeatureImportanceCalculator(list(self.feature_names_in_))
        return calculator.get_top_features(self._feature_importances, top_k)

    def save_model(self, filepath: str, include_sketches: bool = False) -> None:
        """
        Save the trained model to disk.

        Parameters
        ----------
        filepath : str
            Path to save the model (will add .pkl extension if not present)
        include_sketches : bool, default=False
            Whether to include the original sketch data (can be large)
            If False, only saves the trained tree structure and metadata

        Examples
        --------
        >>> clf.fit(sketch_data, feature_mapping)
        >>> clf.save_model('my_model.pkl')
        >>> # Later...
        >>> clf_loaded = ThetaSketchDecisionTreeClassifier.load_model('my_model.pkl')
        """
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted classifier. Call fit() first.")

        # Ensure .pkl extension
        filepath = str(Path(filepath).with_suffix('.pkl'))

        # Prepare model data for serialization
        model_data = {
            'version': '1.0',
            'hyperparameters': {
                'criterion': self.criterion,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'tree_builder': self.tree_builder,
                'verbose': self.verbose
            },
            'fitted_attributes': {
                'classes_': self.classes_,
                'n_classes_': self.n_classes_,
                'n_features_in_': self.n_features_in_,
                'feature_names_in_': self.feature_names_in_,
                'feature_importances': self._feature_importances
            },
            'tree_structure': self._serialize_tree(self.tree_),
            'feature_mapping': self._feature_mapping,
            'is_fitted': self._is_fitted
        }

        # Optionally include sketch data (can be very large)
        if include_sketches:
            print("⚠️  Sketch serialization not yet implemented (DataSketches objects cannot be pickled)")
            print("   Model will be saved without sketch data (prediction-only mode)")
            # TODO: Implement sketch serialization using DataSketches serialize/deserialize methods
            # model_data['sketch_data'] = self._serialize_sketches(self._sketch_dict)

        # Save to file
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Model saved successfully to: {filepath}")

            # Print file size info
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            print(f"File size: {file_size:.2f} MB")

        except Exception as e:
            raise IOError(f"Failed to save model to {filepath}: {e}")

    @classmethod
    def load_model(cls, filepath: str) -> "ThetaSketchDecisionTreeClassifier":
        """
        Load a trained model from disk.

        Parameters
        ----------
        filepath : str
            Path to the saved model file

        Returns
        -------
        classifier : ThetaSketchDecisionTreeClassifier
            Loaded and fitted classifier ready for predictions

        Examples
        --------
        >>> clf = ThetaSketchDecisionTreeClassifier.load_model('my_model.pkl')
        >>> predictions = clf.predict(X_test)
        """
        # Ensure .pkl extension
        filepath = str(Path(filepath).with_suffix('.pkl'))

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            # Validate model data
            if not isinstance(model_data, dict) or 'version' not in model_data:
                raise ValueError("Invalid model file format")

            print(f"Loading model version {model_data['version']} from: {filepath}")

            # Create new classifier instance with saved hyperparameters
            clf = cls(**model_data['hyperparameters'])

            # Restore fitted attributes
            fitted_attrs = model_data['fitted_attributes']
            clf.classes_ = fitted_attrs['classes_']
            clf.n_classes_ = fitted_attrs['n_classes_']
            clf.n_features_in_ = fitted_attrs['n_features_in_']
            clf.feature_names_in_ = fitted_attrs['feature_names_in_']
            clf._feature_importances = fitted_attrs['feature_importances']

            # Restore tree structure
            clf.tree_ = cls._deserialize_tree(model_data['tree_structure'])

            # Restore other internal state
            clf._feature_mapping = model_data['feature_mapping']
            clf._is_fitted = model_data['is_fitted']

            # Restore sketch data if available
            if 'sketch_data' in model_data:
                clf._sketch_dict = model_data['sketch_data']
                print("✅ Sketch data loaded (model can be retrained)")
            else:
                clf._sketch_dict = None
                print("⚠️  Sketch data not available (model is prediction-only)")

            print("✅ Model loaded successfully")
            return clf

        except Exception as e:
            raise IOError(f"Failed to load model from {filepath}: {e}")

    def _serialize_tree(self, node) -> Dict:
        """Serialize tree structure to dictionary."""
        if node is None:
            return None

        tree_dict = {
            'n_samples': node.n_samples,
            'is_leaf': node.is_leaf,
            'depth': getattr(node, 'depth', 0),
            'class_counts': node.class_counts.tolist() if hasattr(node, 'class_counts') else None,
            'impurity': getattr(node, 'impurity', None),
        }

        if node.is_leaf:
            tree_dict.update({
                'prediction': node.prediction,
                'probabilities': getattr(node, 'probabilities', None)
            })
        else:
            tree_dict.update({
                'feature_name': node.feature_name,
                'feature_idx': node.feature_idx,
                'left': self._serialize_tree(node.left),
                'right': self._serialize_tree(node.right)
            })

        return tree_dict

    @classmethod
    def _deserialize_tree(cls, tree_dict, depth: int = 0) -> Any:
        """Deserialize tree structure from dictionary."""
        if tree_dict is None:
            return None

        from .tree_structure import TreeNode

        # Create node with required parameters
        class_counts = np.array(tree_dict['class_counts']) if tree_dict['class_counts'] is not None else np.array([0.0, 0.0])
        impurity = tree_dict['impurity'] if tree_dict['impurity'] is not None else 0.0
        node_depth = tree_dict.get('depth', depth)  # Use stored depth if available

        node = TreeNode(
            depth=node_depth,
            n_samples=tree_dict['n_samples'],
            class_counts=class_counts,
            impurity=impurity
        )

        node.is_leaf = tree_dict['is_leaf']

        if node.is_leaf:
            node.prediction = tree_dict['prediction']
            if tree_dict['probabilities'] is not None:
                node.probabilities = tree_dict['probabilities']
        else:
            node.feature_name = tree_dict['feature_name']
            node.feature_idx = tree_dict['feature_idx']
            node.left = cls._deserialize_tree(tree_dict['left'], depth + 1)
            node.right = cls._deserialize_tree(tree_dict['right'], depth + 1)

            # Set parent references
            if node.left:
                node.left.parent = node
            if node.right:
                node.right.parent = node

        return node

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the fitted model.

        Returns
        -------
        info : dict
            Dictionary containing model metadata and statistics

        Examples
        --------
        >>> info = clf.get_model_info()
        >>> print(f"Model has {info['n_features']} features and depth {info['tree_depth']}")
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before getting info")

        info = {
            'hyperparameters': {
                'criterion': self.criterion,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'tree_builder': self.tree_builder
            },
            'n_features': self.n_features_in_,
            'n_classes': self.n_classes_,
            'feature_names': list(self.feature_names_in_),
            'tree_depth': getattr(self.tree_, 'depth', self._calculate_tree_depth()),
            'tree_nodes': self._count_tree_nodes(),
            'tree_leaves': self._count_tree_leaves(),
            'has_sketch_data': self._sketch_dict is not None,
        }

        return info

    def _calculate_tree_depth(self) -> int:
        """Calculate tree depth recursively."""
        def _depth(node):
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))
        return _depth(self.tree_)

    def _count_tree_nodes(self) -> int:
        """Count total nodes in tree."""
        def _count(node):
            if node is None:
                return 0
            return 1 + _count(node.left) + _count(node.right)
        return _count(self.tree_)

    def _count_tree_leaves(self) -> int:
        """Count leaf nodes in tree."""
        def _count_leaves(node):
            if node is None:
                return 0
            if node.is_leaf:
                return 1
            return _count_leaves(node.left) + _count_leaves(node.right)
        return _count_leaves(self.tree_)

    # ... more methods to be implemented
