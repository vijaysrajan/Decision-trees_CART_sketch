"""
Model persistence functionality for ThetaSketchDecisionTreeClassifier.

This module handles saving and loading of trained models,
separated from the main classifier for cleaner architecture.
"""

import pickle
import os
from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .classifier import ThetaSketchDecisionTreeClassifier

from .tree_structure import TreeNode
from .logging_utils import TreeLogger


class ModelPersistence:
    """
    Handles saving and loading of fitted decision tree models.

    This class is responsible for:
    - Serializing/deserializing tree structures
    - Handling model metadata and hyperparameters
    - Managing file I/O operations
    - Validating model file formats
    """

    _logger = TreeLogger(__name__)

    @staticmethod
    def save_model(classifier: 'ThetaSketchDecisionTreeClassifier',
                  filepath: str,
                  include_sketches: bool = False) -> None:
        """
        Save the trained model to disk.

        Parameters
        ----------
        classifier : ThetaSketchDecisionTreeClassifier
            Fitted classifier to save
        filepath : str
            Path to save the model (will add .pkl extension if not present)
        include_sketches : bool, default=False
            Whether to include the original sketch data (can be large)
            If False, only saves the trained tree structure and metadata

        Examples
        --------
        >>> ModelPersistence.save_model(clf, 'my_model.pkl')
        """
        if not classifier._is_fitted:
            raise ValueError("Cannot save unfitted classifier. Call fit() first.")

        # Ensure .pkl extension
        filepath = str(Path(filepath).with_suffix('.pkl'))

        # Prepare model data for serialization
        model_data = {
            'version': '1.0',
            'hyperparameters': {
                'criterion': classifier.criterion,
                'max_depth': classifier.max_depth,
                'min_samples_split': classifier.min_samples_split,
                'min_samples_leaf': classifier.min_samples_leaf,
                'pruning': classifier.pruning,
                'min_impurity_decrease': classifier.min_impurity_decrease,
                'validation_fraction': classifier.validation_fraction,
                'verbose': classifier.verbose,
                'random_state': classifier.random_state
            },
            'fitted_attributes': {
                'classes_': classifier.classes_,
                'n_classes_': classifier.n_classes_,
                'n_features_in_': classifier.n_features_in_,
                'feature_names_in_': classifier.feature_names_in_,
                'feature_importances': classifier._feature_importances
            },
            'tree_structure': ModelPersistence._serialize_tree(classifier.tree_),
            'feature_mapping': classifier._feature_mapping,
            'is_fitted': classifier._is_fitted
        }

        # Optionally include sketch data (can be very large)
        if include_sketches:
            ModelPersistence._logger.warning("⚠️  Sketch serialization not yet implemented (DataSketches objects cannot be pickled)")
            ModelPersistence._logger.warning("   Model will be saved without sketch data (prediction-only mode)")
            # TODO: Implement sketch serialization using DataSketches serialize/deserialize methods

        # Save to file
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            ModelPersistence._logger.info(f"Model saved successfully to: {filepath}")

            # Print file size info
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            ModelPersistence._logger.info(f"File size: {file_size:.2f} MB")

        except Exception as e:
            raise IOError(f"Failed to save model to {filepath}: {e}")

    @staticmethod
    def load_model(filepath: str) -> 'ThetaSketchDecisionTreeClassifier':
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
        >>> clf = ModelPersistence.load_model('my_model.pkl')
        >>> predictions = clf.predict(X_test)
        """
        from .classifier import ThetaSketchDecisionTreeClassifier

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

            ModelPersistence._logger.info(f"Loading model version {model_data['version']} from: {filepath}")

            # Create new classifier instance with saved hyperparameters
            clf = ThetaSketchDecisionTreeClassifier(**model_data['hyperparameters'])

            # Restore fitted attributes
            fitted_attrs = model_data['fitted_attributes']
            clf.classes_ = fitted_attrs['classes_']
            clf.n_classes_ = fitted_attrs['n_classes_']
            clf.n_features_in_ = fitted_attrs['n_features_in_']
            clf.feature_names_in_ = fitted_attrs['feature_names_in_']
            clf._feature_importances = fitted_attrs['feature_importances']

            # Restore tree structure
            clf.tree_ = ModelPersistence._deserialize_tree(model_data['tree_structure'])

            # Restore other internal state
            clf._feature_mapping = model_data['feature_mapping']
            clf._is_fitted = model_data['is_fitted']

            # Restore sketch data if available
            if 'sketch_data' in model_data:
                clf._sketch_dict = model_data['sketch_data']
                ModelPersistence._logger.info("✅ Sketch data loaded (model can be retrained)")
            else:
                clf._sketch_dict = None
                ModelPersistence._logger.warning("⚠️  Sketch data not available (model is prediction-only)")

            ModelPersistence._logger.info("✅ Model loaded successfully")
            return clf

        except Exception as e:
            raise IOError(f"Failed to load model from {filepath}: {e}")

    @staticmethod
    def _serialize_tree(node: TreeNode) -> Dict:
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
                'left': ModelPersistence._serialize_tree(node.left),
                'right': ModelPersistence._serialize_tree(node.right)
            })

        return tree_dict

    @staticmethod
    def _deserialize_tree(tree_dict: Dict, depth: int = 0) -> TreeNode:
        """Deserialize tree structure from dictionary."""
        if tree_dict is None:
            return None

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
            node.left = ModelPersistence._deserialize_tree(tree_dict['left'], depth + 1)
            node.right = ModelPersistence._deserialize_tree(tree_dict['right'], depth + 1)

            # Set parent references
            if node.left:
                node.left.parent = node
            if node.right:
                node.right.parent = node

        return node

    @staticmethod
    def get_model_info(classifier: 'ThetaSketchDecisionTreeClassifier') -> Dict[str, Any]:
        """
        Get comprehensive information about the fitted model.

        Parameters
        ----------
        classifier : ThetaSketchDecisionTreeClassifier
            Fitted classifier

        Returns
        -------
        info : dict
            Dictionary containing model metadata and statistics

        Examples
        --------
        >>> info = ModelPersistence.get_model_info(clf)
        >>> print(f"Model has {info['n_features']} features and depth {info['tree_depth']}")
        """
        if not classifier._is_fitted:
            raise ValueError("Model must be fitted before getting info")

        info = {
            'hyperparameters': {
                'criterion': classifier.criterion,
                'max_depth': classifier.max_depth,
                'min_samples_split': classifier.min_samples_split,
                'min_samples_leaf': classifier.min_samples_leaf,
                'pruning': classifier.pruning,
                'min_impurity_decrease': classifier.min_impurity_decrease,
                'validation_fraction': classifier.validation_fraction
            },
            'n_features': classifier.n_features_in_,
            'n_classes': classifier.n_classes_,
            'feature_names': list(classifier.feature_names_in_),
            'tree_depth': getattr(classifier.tree_, 'depth', ModelPersistence._calculate_tree_depth(classifier.tree_)),
            'tree_nodes': ModelPersistence._count_tree_nodes(classifier.tree_),
            'tree_leaves': ModelPersistence._count_tree_leaves(classifier.tree_),
            'has_sketch_data': classifier._sketch_dict is not None,
        }

        return info

    @staticmethod
    def _calculate_tree_depth(root: TreeNode) -> int:
        """Calculate tree depth recursively."""
        def _depth(node):
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))
        return _depth(root)

    @staticmethod
    def _count_tree_nodes(root: TreeNode) -> int:
        """Count total nodes in tree."""
        def _count(node):
            if node is None:
                return 0
            return 1 + _count(node.left) + _count(node.right)
        return _count(root)

    @staticmethod
    def _count_tree_leaves(root: TreeNode) -> int:
        """Count leaf nodes in tree."""
        def _count_leaves(node):
            if node is None:
                return 0
            if node.is_leaf:
                return 1
            return _count_leaves(node.left) + _count_leaves(node.right)
        return _count_leaves(root)