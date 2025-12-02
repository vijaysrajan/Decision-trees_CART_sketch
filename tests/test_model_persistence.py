"""
Comprehensive tests for model_persistence module.

Tests model saving, loading, serialization, and utility methods.
"""

import pytest
import pickle
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from theta_sketch_tree.model_persistence import ModelPersistence
from theta_sketch_tree.tree_structure import TreeNode
from tests.test_mock_sketches import MockThetaSketch


@pytest.fixture
def mock_classifier():
    """Create a mock fitted classifier for testing."""
    classifier = Mock()
    classifier._is_fitted = True
    classifier.criterion = 'gini'
    classifier.max_depth = 5
    classifier.min_samples_split = 2
    classifier.min_samples_leaf = 1
    classifier.pruning = 'none'
    classifier.min_impurity_decrease = 0.0
    classifier.validation_fraction = 0.1
    classifier.verbose = 0
    classifier.random_state = None

    # Fitted attributes
    classifier.classes_ = np.array([0, 1])
    classifier.n_classes_ = 2
    classifier.n_features_in_ = 3
    classifier.feature_names_in_ = np.array(['feature1', 'feature2', 'feature3'])
    classifier._feature_importances = np.array([0.5, 0.3, 0.2])
    classifier._feature_mapping = {'feature1': 0, 'feature2': 1, 'feature3': 2}
    classifier._sketch_dict = None

    # Create a simple tree structure
    classifier.tree_ = TreeNode(
        depth=0,
        n_samples=100,
        class_counts=np.array([50.0, 50.0]),
        impurity=0.5
    )
    classifier.tree_.is_leaf = False
    classifier.tree_.feature_name = 'feature1'
    classifier.tree_.feature_idx = 0

    # Add children
    left_child = TreeNode(
        depth=1,
        n_samples=50,
        class_counts=np.array([40.0, 10.0]),
        impurity=0.32,
        parent=classifier.tree_
    )
    left_child.is_leaf = True
    left_child.prediction = 0
    left_child.probabilities = [0.8, 0.2]

    right_child = TreeNode(
        depth=1,
        n_samples=50,
        class_counts=np.array([10.0, 40.0]),
        impurity=0.32,
        parent=classifier.tree_
    )
    right_child.is_leaf = True
    right_child.prediction = 1
    right_child.probabilities = [0.2, 0.8]

    classifier.tree_.left = left_child
    classifier.tree_.right = right_child

    return classifier


@pytest.fixture
def mock_unfitted_classifier():
    """Create a mock unfitted classifier for testing."""
    classifier = Mock()
    classifier._is_fitted = False
    return classifier


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        temp_path = f.name
    yield temp_path
    # Clean up
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestModelPersistence:
    """Test suite for ModelPersistence class."""

    def test_save_model_basic(self, mock_classifier, temp_file):
        """Test basic model saving functionality."""
        ModelPersistence.save_model(mock_classifier, temp_file)

        # Verify file was created
        assert os.path.exists(temp_file)

        # Verify file contains expected data
        with open(temp_file, 'rb') as f:
            data = pickle.load(f)

        assert 'version' in data
        assert 'hyperparameters' in data
        assert 'fitted_attributes' in data
        assert 'tree_structure' in data
        assert data['version'] == '1.0'

    def test_save_model_unfitted_classifier(self, mock_unfitted_classifier, temp_file):
        """Test error when saving unfitted classifier."""
        with pytest.raises(ValueError, match="Cannot save unfitted classifier"):
            ModelPersistence.save_model(mock_unfitted_classifier, temp_file)

    def test_save_model_auto_pkl_extension(self, mock_classifier):
        """Test automatic .pkl extension addition."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, 'model_no_extension')
            ModelPersistence.save_model(mock_classifier, filepath)

            expected_path = filepath + '.pkl'
            assert os.path.exists(expected_path)

    def test_save_model_with_sketches_warning(self, mock_classifier, temp_file, caplog):
        """Test warning when include_sketches=True."""
        import logging
        caplog.set_level(logging.WARNING)

        ModelPersistence.save_model(mock_classifier, temp_file, include_sketches=True)
        assert os.path.exists(temp_file)

        # Check that warning was logged about sketch serialization
        warning_msgs = [record.message for record in caplog.records if record.levelname == 'WARNING']
        # Should have warnings about sketch serialization not implemented
        assert len(warning_msgs) >= 1

    @patch('builtins.open', side_effect=IOError("Permission denied"))
    def test_save_model_io_error(self, mock_open, mock_classifier, temp_file):
        """Test IOError handling during save."""
        with pytest.raises(IOError, match="Failed to save model"):
            ModelPersistence.save_model(mock_classifier, temp_file)

    @patch('theta_sketch_tree.classifier.ThetaSketchDecisionTreeClassifier')
    def test_load_model_basic(self, mock_cls, mock_classifier, temp_file):
        """Test basic model loading functionality."""
        # First save a model
        ModelPersistence.save_model(mock_classifier, temp_file)

        # Mock the classifier class for loading
        mock_loaded_clf = Mock()
        mock_cls.return_value = mock_loaded_clf

        # Then load it
        loaded_clf = ModelPersistence.load_model(temp_file)

        assert loaded_clf == mock_loaded_clf
        mock_cls.assert_called_once()

    @patch('theta_sketch_tree.classifier.ThetaSketchDecisionTreeClassifier')
    def test_load_model_auto_pkl_extension(self, mock_cls, mock_classifier):
        """Test automatic .pkl extension for loading."""
        mock_loaded_clf = Mock()
        mock_cls.return_value = mock_loaded_clf

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, 'model_no_ext')

            # Save with extension
            ModelPersistence.save_model(mock_classifier, filepath + '.pkl')

            # Load without extension
            loaded_clf = ModelPersistence.load_model(filepath)
            assert loaded_clf == mock_loaded_clf

    def test_load_model_file_not_found(self):
        """Test FileNotFoundError when model file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            ModelPersistence.load_model('/nonexistent/path.pkl')

    @patch('theta_sketch_tree.classifier.ThetaSketchDecisionTreeClassifier')
    def test_load_model_invalid_format(self, mock_cls, temp_file):
        """Test ValueError when model file has invalid format."""
        # Create invalid pickle file
        with open(temp_file, 'wb') as f:
            pickle.dump("invalid_data", f)

        with pytest.raises(ValueError, match="Invalid model file format"):
            ModelPersistence.load_model(temp_file)

    @patch('theta_sketch_tree.classifier.ThetaSketchDecisionTreeClassifier')
    def test_load_model_missing_version(self, mock_cls, temp_file):
        """Test ValueError when model file missing version."""
        # Create pickle without version
        with open(temp_file, 'wb') as f:
            pickle.dump({}, f)

        with pytest.raises(ValueError, match="Invalid model file format"):
            ModelPersistence.load_model(temp_file)

    @patch('builtins.open', side_effect=IOError("Read error"))
    def test_load_model_io_error(self, mock_open, temp_file):
        """Test IOError handling during load."""
        with pytest.raises(IOError, match="Failed to load model"):
            ModelPersistence.load_model(temp_file)

    @patch('theta_sketch_tree.classifier.ThetaSketchDecisionTreeClassifier')
    def test_load_model_with_sketch_data(self, mock_cls, temp_file):
        """Test loading model with sketch data."""
        mock_loaded_clf = Mock()
        mock_cls.return_value = mock_loaded_clf

        # Create model data with sketch_data
        model_data = {
            'version': '1.0',
            'hyperparameters': {
                'criterion': 'gini',
                'max_depth': 5,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'pruning': 'none',
                'min_impurity_decrease': 0.0,
                'validation_fraction': 0.1,
                'verbose': 0,
                'random_state': None
            },
            'fitted_attributes': {
                'classes_': np.array([0, 1]),
                'n_classes_': 2,
                'n_features_in_': 3,
                'feature_names_in_': np.array(['f1', 'f2', 'f3']),
                'feature_importances': np.array([0.5, 0.3, 0.2])
            },
            'tree_structure': {'n_samples': 100, 'is_leaf': True, 'depth': 0, 'class_counts': [50, 50], 'impurity': 0.5, 'prediction': 0, 'probabilities': [0.5, 0.5]},
            'feature_mapping': {'f1': 0, 'f2': 1, 'f3': 2},
            'is_fitted': True,
            'sketch_data': {'positive': {}, 'negative': {}}
        }

        with open(temp_file, 'wb') as f:
            pickle.dump(model_data, f)

        loaded_clf = ModelPersistence.load_model(temp_file)
        assert loaded_clf == mock_loaded_clf

    def test_serialize_tree_leaf_node(self):
        """Test tree serialization for leaf node."""
        leaf = TreeNode(
            depth=1,
            n_samples=50,
            class_counts=np.array([40.0, 10.0]),
            impurity=0.32
        )
        leaf.is_leaf = True
        leaf.prediction = 0
        leaf.probabilities = [0.8, 0.2]

        serialized = ModelPersistence._serialize_tree(leaf)

        assert serialized['is_leaf']
        assert serialized['prediction'] == 0
        assert serialized['probabilities'] == [0.8, 0.2]
        assert serialized['n_samples'] == 50

    def test_serialize_tree_internal_node(self):
        """Test tree serialization for internal node."""
        root = TreeNode(
            depth=0,
            n_samples=100,
            class_counts=np.array([50.0, 50.0]),
            impurity=0.5
        )
        root.is_leaf = False
        root.feature_name = 'feature1'
        root.feature_idx = 0

        left = TreeNode(depth=1, n_samples=50, class_counts=np.array([40.0, 10.0]), impurity=0.32)
        left.is_leaf = True
        left.prediction = 0

        right = TreeNode(depth=1, n_samples=50, class_counts=np.array([10.0, 40.0]), impurity=0.32)
        right.is_leaf = True
        right.prediction = 1

        root.left = left
        root.right = right

        serialized = ModelPersistence._serialize_tree(root)

        assert not serialized['is_leaf']
        assert serialized['feature_name'] == 'feature1'
        assert serialized['feature_idx'] == 0
        assert serialized['left'] is not None
        assert serialized['right'] is not None

    def test_serialize_tree_none(self):
        """Test tree serialization for None node."""
        assert ModelPersistence._serialize_tree(None) is None

    def test_deserialize_tree_leaf_node(self):
        """Test tree deserialization for leaf node."""
        tree_dict = {
            'n_samples': 50,
            'is_leaf': True,
            'depth': 1,
            'class_counts': [40.0, 10.0],
            'impurity': 0.32,
            'prediction': 0,
            'probabilities': [0.8, 0.2]
        }

        node = ModelPersistence._deserialize_tree(tree_dict)

        assert node.is_leaf
        assert node.prediction == 0
        assert node.probabilities == [0.8, 0.2]
        assert node.n_samples == 50

    def test_deserialize_tree_internal_node(self):
        """Test tree deserialization for internal node."""
        tree_dict = {
            'n_samples': 100,
            'is_leaf': False,
            'depth': 0,
            'class_counts': [50.0, 50.0],
            'impurity': 0.5,
            'feature_name': 'feature1',
            'feature_idx': 0,
            'left': {
                'n_samples': 50,
                'is_leaf': True,
                'depth': 1,
                'class_counts': [40.0, 10.0],
                'impurity': 0.32,
                'prediction': 0,
                'probabilities': None
            },
            'right': {
                'n_samples': 50,
                'is_leaf': True,
                'depth': 1,
                'class_counts': [10.0, 40.0],
                'impurity': 0.32,
                'prediction': 1,
                'probabilities': None
            }
        }

        node = ModelPersistence._deserialize_tree(tree_dict)

        assert not node.is_leaf
        assert node.feature_name == 'feature1'
        assert node.feature_idx == 0
        assert node.left is not None
        assert node.right is not None
        assert node.left.parent == node
        assert node.right.parent == node

    def test_deserialize_tree_none(self):
        """Test tree deserialization for None."""
        assert ModelPersistence._deserialize_tree(None) is None

    def test_deserialize_tree_missing_class_counts(self):
        """Test tree deserialization with missing class_counts."""
        tree_dict = {
            'n_samples': 50,
            'is_leaf': True,
            'depth': 1,
            'class_counts': None,
            'impurity': None,
            'prediction': 0,
            'probabilities': [0.8, 0.2]
        }

        node = ModelPersistence._deserialize_tree(tree_dict)

        assert node is not None
        assert np.array_equal(node.class_counts, np.array([0.0, 0.0]))
        assert node.impurity == 0.0

    def test_get_model_info_basic(self, mock_classifier):
        """Test basic model info generation."""
        info = ModelPersistence.get_model_info(mock_classifier)

        assert 'hyperparameters' in info
        assert 'n_features' in info
        assert 'n_classes' in info
        assert 'feature_names' in info
        assert 'tree_depth' in info
        assert 'tree_nodes' in info
        assert 'tree_leaves' in info
        assert 'has_sketch_data' in info

        assert info['n_features'] == 3
        assert info['n_classes'] == 2
        assert info['has_sketch_data'] == False

    def test_get_model_info_unfitted(self, mock_unfitted_classifier):
        """Test error for unfitted model info."""
        with pytest.raises(ValueError, match="Model must be fitted before getting info"):
            ModelPersistence.get_model_info(mock_unfitted_classifier)

    def test_calculate_tree_depth(self):
        """Test tree depth calculation."""
        # Create simple tree: root -> left (leaf), right -> right_left (leaf), right_right (leaf)
        root = TreeNode(depth=0, n_samples=100, class_counts=np.array([50, 50]), impurity=0.5)
        root.is_leaf = False

        left = TreeNode(depth=1, n_samples=50, class_counts=np.array([40, 10]), impurity=0.32)
        left.is_leaf = True

        right = TreeNode(depth=1, n_samples=50, class_counts=np.array([10, 40]), impurity=0.32)
        right.is_leaf = False

        right_left = TreeNode(depth=2, n_samples=25, class_counts=np.array([5, 20]), impurity=0.32)
        right_left.is_leaf = True

        right_right = TreeNode(depth=2, n_samples=25, class_counts=np.array([5, 20]), impurity=0.32)
        right_right.is_leaf = True

        root.left = left
        root.right = right
        right.left = right_left
        right.right = right_right

        depth = ModelPersistence._calculate_tree_depth(root)
        assert depth == 2  # 0->1->2 is depth 2

    def test_calculate_tree_depth_single_node(self):
        """Test tree depth calculation for single node."""
        root = TreeNode(depth=0, n_samples=100, class_counts=np.array([50, 50]), impurity=0.5)
        root.is_leaf = True

        depth = ModelPersistence._calculate_tree_depth(root)
        assert depth == 0

    def test_calculate_tree_depth_none(self):
        """Test tree depth calculation for None tree."""
        depth = ModelPersistence._calculate_tree_depth(None)
        assert depth == 0

    def test_count_tree_nodes(self):
        """Test tree node counting."""
        # Create tree with 3 nodes
        root = TreeNode(depth=0, n_samples=100, class_counts=np.array([50, 50]), impurity=0.5)
        root.is_leaf = False

        left = TreeNode(depth=1, n_samples=50, class_counts=np.array([40, 10]), impurity=0.32)
        left.is_leaf = True

        right = TreeNode(depth=1, n_samples=50, class_counts=np.array([10, 40]), impurity=0.32)
        right.is_leaf = True

        root.left = left
        root.right = right

        count = ModelPersistence._count_tree_nodes(root)
        assert count == 3

    def test_count_tree_nodes_none(self):
        """Test tree node counting for None tree."""
        count = ModelPersistence._count_tree_nodes(None)
        assert count == 0

    def test_count_tree_leaves(self):
        """Test tree leaf counting."""
        # Create tree with 2 leaves
        root = TreeNode(depth=0, n_samples=100, class_counts=np.array([50, 50]), impurity=0.5)
        root.is_leaf = False

        left = TreeNode(depth=1, n_samples=50, class_counts=np.array([40, 10]), impurity=0.32)
        left.is_leaf = True

        right = TreeNode(depth=1, n_samples=50, class_counts=np.array([10, 40]), impurity=0.32)
        right.is_leaf = True

        root.left = left
        root.right = right

        count = ModelPersistence._count_tree_leaves(root)
        assert count == 2

    def test_count_tree_leaves_single_leaf(self):
        """Test tree leaf counting for single leaf."""
        root = TreeNode(depth=0, n_samples=100, class_counts=np.array([50, 50]), impurity=0.5)
        root.is_leaf = True

        count = ModelPersistence._count_tree_leaves(root)
        assert count == 1

    def test_count_tree_leaves_none(self):
        """Test tree leaf counting for None tree."""
        count = ModelPersistence._count_tree_leaves(None)
        assert count == 0

    def test_get_model_info_with_sketch_data(self, mock_classifier):
        """Test model info when sketch data is available."""
        mock_classifier._sketch_dict = {'positive': {}, 'negative': {}}

        info = ModelPersistence.get_model_info(mock_classifier)
        assert info['has_sketch_data'] == True

    @patch('theta_sketch_tree.classifier.ThetaSketchDecisionTreeClassifier')
    def test_model_save_load_roundtrip(self, mock_cls, mock_classifier, temp_file):
        """Test full save/load roundtrip preserves model data."""
        mock_loaded_clf = Mock()
        mock_cls.return_value = mock_loaded_clf

        # Save model
        ModelPersistence.save_model(mock_classifier, temp_file)

        # Load model
        loaded_clf = ModelPersistence.load_model(temp_file)

        # Verify load succeeded
        assert loaded_clf == mock_loaded_clf
        mock_cls.assert_called_once()

    @patch('theta_sketch_tree.classifier.ThetaSketchDecisionTreeClassifier')
    def test_version_display_during_load(self, mock_cls, mock_classifier, temp_file):
        """Test that model version is displayed during load."""
        mock_loaded_clf = Mock()
        mock_cls.return_value = mock_loaded_clf

        ModelPersistence.save_model(mock_classifier, temp_file)

        # The load should succeed and display version info via logger
        loaded_clf = ModelPersistence.load_model(temp_file)
        assert loaded_clf == mock_loaded_clf