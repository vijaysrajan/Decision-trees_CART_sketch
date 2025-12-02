"""
Comprehensive tests for classifier_utils module.

Tests utility functions and convenience methods.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from theta_sketch_tree.classifier_utils import (
    ClassifierUtils,
    fit_from_csv,
    create_classifier_with_defaults
)


@pytest.fixture
def mock_fitted_classifier():
    """Create a mock fitted classifier."""
    classifier = Mock()
    classifier._is_fitted = True
    classifier.feature_names_in_ = np.array(['feature1', 'feature2', 'feature3'])
    classifier._feature_importances = np.array([0.5, 0.3, 0.2])
    classifier._check_is_fitted = Mock()
    return classifier


@pytest.fixture
def mock_sketch_data():
    """Create mock sketch data."""
    return {
        'positive': {'total': 'mock_sketch', 'feature1': ('mock_sketch1', 'mock_sketch2')},
        'negative': {'total': 'mock_sketch', 'feature1': ('mock_sketch3', 'mock_sketch4')}
    }


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return {
        'targets': {'positive': 'target_yes', 'negative': 'target_no'},
        'hyperparameters': {'criterion': 'gini', 'max_depth': 5},
        'feature_mapping': {'feature1': 0, 'feature2': 1}
    }


@pytest.fixture
def temp_csv_file():
    """Create temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write("identifier,sketch_present,sketch_absent\n")
        f.write("total,sketch1,sketch2\n")
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_config_file():
    """Create temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
        f.write("targets:\n  positive: target_yes\n  negative: target_no\n")
        f.write("hyperparameters:\n  criterion: gini\n  max_depth: 5\n")
        f.write("feature_mapping:\n  feature1: 0\n")
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestClassifierUtils:
    """Test suite for ClassifierUtils class."""

    @patch('theta_sketch_tree.load_sketches')
    @patch('theta_sketch_tree.load_config')
    def test_fit_from_csv_dual_csv_mode(self, mock_load_config, mock_load_sketches, temp_csv_file, mock_config, mock_sketch_data):
        """Test fit_from_csv with dual CSV mode."""
        # Setup mocks
        mock_load_config.return_value = mock_config
        mock_load_sketches.return_value = mock_sketch_data

        # Create mock classifier class
        mock_classifier_class = Mock()
        mock_classifier = Mock()
        mock_classifier_class.return_value = mock_classifier

        # Call method
        result = ClassifierUtils.fit_from_csv(
            positive_csv=temp_csv_file,
            negative_csv=temp_csv_file,
            config_path="config.yaml",
            classifier_class=mock_classifier_class
        )

        # Verify calls
        mock_load_sketches.assert_called_once_with(temp_csv_file, temp_csv_file)
        mock_load_config.assert_called_once_with("config.yaml")
        mock_classifier_class.assert_called_once_with(**mock_config["hyperparameters"])
        mock_classifier.fit.assert_called_once_with(mock_sketch_data, mock_config["feature_mapping"])
        assert result == mock_classifier

    @patch('theta_sketch_tree.load_sketches')
    @patch('theta_sketch_tree.load_config')
    def test_fit_from_csv_single_csv_mode(self, mock_load_config, mock_load_sketches, temp_csv_file, mock_config, mock_sketch_data):
        """Test fit_from_csv with single CSV mode."""
        # Setup mocks
        mock_load_config.return_value = mock_config
        mock_load_sketches.return_value = mock_sketch_data

        # Create mock classifier class
        mock_classifier_class = Mock()
        mock_classifier = Mock()
        mock_classifier_class.return_value = mock_classifier

        # Call method with csv_path
        result = ClassifierUtils.fit_from_csv(
            positive_csv="ignored",
            negative_csv="ignored",
            config_path="config.yaml",
            csv_path=temp_csv_file,
            classifier_class=mock_classifier_class
        )

        # Verify single CSV mode was used
        mock_load_sketches.assert_called_once_with(
            csv_path=temp_csv_file,
            target_positive=mock_config["targets"]["positive"],
            target_negative=mock_config["targets"]["negative"]
        )

    @patch('theta_sketch_tree.load_sketches')
    @patch('theta_sketch_tree.load_config')
    @patch('theta_sketch_tree.classifier.ThetaSketchDecisionTreeClassifier')
    def test_fit_from_csv_default_classifier_class(self, mock_default_class, mock_load_config, mock_load_sketches, temp_csv_file, mock_config, mock_sketch_data):
        """Test fit_from_csv with default classifier class."""
        mock_load_config.return_value = mock_config
        mock_load_sketches.return_value = mock_sketch_data

        mock_classifier = Mock()
        mock_default_class.return_value = mock_classifier

        result = ClassifierUtils.fit_from_csv(
            positive_csv=temp_csv_file,
            negative_csv=temp_csv_file,
            config_path="config.yaml"
        )

        mock_default_class.assert_called_once_with(**mock_config["hyperparameters"])

    @patch('theta_sketch_tree.classifier.ThetaSketchDecisionTreeClassifier')
    def test_create_classifier_with_defaults(self, mock_classifier_class):
        """Test creating classifier with defaults."""
        mock_classifier = Mock()
        mock_classifier_class.return_value = mock_classifier

        result = ClassifierUtils.create_classifier_with_defaults()

        expected_defaults = {
            "criterion": "gini",
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "pruning": "none",
            "verbose": 0
        }
        mock_classifier_class.assert_called_once_with(**expected_defaults)
        assert result == mock_classifier

    @patch('theta_sketch_tree.classifier.ThetaSketchDecisionTreeClassifier')
    def test_create_classifier_with_custom_params(self, mock_classifier_class):
        """Test creating classifier with custom parameters."""
        mock_classifier = Mock()
        mock_classifier_class.return_value = mock_classifier

        custom_params = {"criterion": "entropy", "max_depth": 15, "new_param": "value"}
        result = ClassifierUtils.create_classifier_with_defaults(**custom_params)

        expected_params = {
            "criterion": "entropy",  # Overridden
            "max_depth": 15,  # Overridden
            "min_samples_split": 2,  # Default
            "min_samples_leaf": 1,  # Default
            "pruning": "none",  # Default
            "verbose": 0,  # Default
            "new_param": "value"  # New param
        }
        mock_classifier_class.assert_called_once_with(**expected_params)

    def test_get_feature_importance_dict(self, mock_fitted_classifier):
        """Test getting feature importance dictionary."""
        result = ClassifierUtils.get_feature_importance_dict(mock_fitted_classifier)

        expected = {
            'feature1': 0.5,
            'feature2': 0.3,
            'feature3': 0.2
        }
        assert result == expected
        mock_fitted_classifier._check_is_fitted.assert_called_once()

    def test_get_top_features_default_k(self, mock_fitted_classifier):
        """Test getting top features with default k."""
        result = ClassifierUtils.get_top_features(mock_fitted_classifier)

        expected = [
            ('feature1', 0.5),
            ('feature2', 0.3),
            ('feature3', 0.2)
        ]
        assert result == expected
        mock_fitted_classifier._check_is_fitted.assert_called_once()

    def test_get_top_features_custom_k(self, mock_fitted_classifier):
        """Test getting top features with custom k."""
        result = ClassifierUtils.get_top_features(mock_fitted_classifier, top_k=2)

        expected = [
            ('feature1', 0.5),
            ('feature2', 0.3)
        ]
        assert result == expected

    def test_get_top_features_k_larger_than_features(self, mock_fitted_classifier):
        """Test getting top features when k > number of features."""
        result = ClassifierUtils.get_top_features(mock_fitted_classifier, top_k=10)

        # Should return all features even though k=10 > 3 features
        expected = [
            ('feature1', 0.5),
            ('feature2', 0.3),
            ('feature3', 0.2)
        ]
        assert result == expected

    @patch('theta_sketch_tree.model_persistence.ModelPersistence')
    def test_save_model(self, mock_persistence, mock_fitted_classifier):
        """Test save_model delegates to ModelPersistence."""
        ClassifierUtils.save_model(mock_fitted_classifier, "test.pkl", True)

        mock_persistence.save_model.assert_called_once_with(mock_fitted_classifier, "test.pkl", True)

    @patch('theta_sketch_tree.model_persistence.ModelPersistence')
    def test_save_model_default_include_sketches(self, mock_persistence, mock_fitted_classifier):
        """Test save_model with default include_sketches parameter."""
        ClassifierUtils.save_model(mock_fitted_classifier, "test.pkl")

        mock_persistence.save_model.assert_called_once_with(mock_fitted_classifier, "test.pkl", False)

    @patch('theta_sketch_tree.model_persistence.ModelPersistence')
    def test_load_model(self, mock_persistence):
        """Test load_model delegates to ModelPersistence."""
        mock_classifier = Mock()
        mock_persistence.load_model.return_value = mock_classifier

        result = ClassifierUtils.load_model("test.pkl")

        mock_persistence.load_model.assert_called_once_with("test.pkl")
        assert result == mock_classifier

    @patch('theta_sketch_tree.model_persistence.ModelPersistence')
    def test_get_model_info(self, mock_persistence, mock_fitted_classifier):
        """Test get_model_info delegates to ModelPersistence."""
        mock_info = {'tree_depth': 3}
        mock_persistence.get_model_info.return_value = mock_info

        result = ClassifierUtils.get_model_info(mock_fitted_classifier)

        mock_persistence.get_model_info.assert_called_once_with(mock_fitted_classifier)
        assert result == mock_info


class TestConvenienceFunctions:
    """Test convenience functions for backward compatibility."""

    @patch('theta_sketch_tree.classifier_utils.ClassifierUtils.fit_from_csv')
    def test_fit_from_csv_function(self, mock_method):
        """Test fit_from_csv convenience function."""
        mock_classifier = Mock()
        mock_method.return_value = mock_classifier

        result = fit_from_csv("pos.csv", "neg.csv", "config.yaml", "single.csv")

        mock_method.assert_called_once_with("pos.csv", "neg.csv", "config.yaml", "single.csv")
        assert result == mock_classifier

    @patch('theta_sketch_tree.classifier_utils.ClassifierUtils.create_classifier_with_defaults')
    def test_create_classifier_with_defaults_function(self, mock_method):
        """Test create_classifier_with_defaults convenience function."""
        mock_classifier = Mock()
        mock_method.return_value = mock_classifier

        result = create_classifier_with_defaults(criterion="entropy", max_depth=15)

        mock_method.assert_called_once_with(criterion="entropy", max_depth=15)
        assert result == mock_classifier


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_get_feature_importance_dict_empty_features(self):
        """Test feature importance dict with empty features."""
        classifier = Mock()
        classifier._check_is_fitted = Mock()
        classifier.feature_names_in_ = np.array([])
        classifier._feature_importances = np.array([])

        result = ClassifierUtils.get_feature_importance_dict(classifier)

        assert result == {}

    def test_get_top_features_zero_k(self, mock_fitted_classifier):
        """Test getting top features with k=0."""
        result = ClassifierUtils.get_top_features(mock_fitted_classifier, top_k=0)

        assert result == []

    def test_get_top_features_with_tied_importances(self):
        """Test getting top features with tied importance scores."""
        classifier = Mock()
        classifier._check_is_fitted = Mock()
        classifier.feature_names_in_ = np.array(['feature1', 'feature2', 'feature3'])
        classifier._feature_importances = np.array([0.3, 0.3, 0.2])  # Tied scores

        result = ClassifierUtils.get_top_features(classifier, top_k=2)

        # Should return first 2 in order (stable sort)
        assert len(result) == 2
        assert result[0][1] == 0.3  # First importance
        assert result[1][1] == 0.3  # Second importance

    @patch('theta_sketch_tree.load_sketches')
    @patch('theta_sketch_tree.load_config')
    @patch('theta_sketch_tree.classifier.ThetaSketchDecisionTreeClassifier')
    def test_fit_from_csv_config_loaded_twice_single_mode(self, mock_cls, mock_load_config, mock_load_sketches):
        """Test that config is loaded twice in single CSV mode."""
        mock_config = {
            'targets': {'positive': 'yes', 'negative': 'no'},
            'hyperparameters': {'criterion': 'gini'},
            'feature_mapping': {'f1': 0}
        }
        mock_load_config.return_value = mock_config
        mock_load_sketches.return_value = {'positive': {}, 'negative': {}}

        mock_classifier = Mock()
        mock_cls.return_value = mock_classifier

        ClassifierUtils.fit_from_csv(
            positive_csv="pos.csv",
            negative_csv="neg.csv",
            config_path="config.yaml",
            csv_path="single.csv"
        )

        # Config should be loaded twice in single CSV mode
        assert mock_load_config.call_count == 2

    def test_get_feature_importance_with_negative_values(self):
        """Test feature importance handling with negative values."""
        classifier = Mock()
        classifier._check_is_fitted = Mock()
        classifier.feature_names_in_ = np.array(['feature1', 'feature2'])
        classifier._feature_importances = np.array([-0.1, 0.5])  # Negative importance

        result = ClassifierUtils.get_feature_importance_dict(classifier)

        expected = {'feature1': -0.1, 'feature2': 0.5}
        assert result == expected

    def test_get_top_features_with_negative_importances(self):
        """Test top features with negative importances."""
        classifier = Mock()
        classifier._check_is_fitted = Mock()
        classifier.feature_names_in_ = np.array(['feature1', 'feature2', 'feature3'])
        classifier._feature_importances = np.array([-0.1, 0.5, -0.2])

        result = ClassifierUtils.get_top_features(classifier, top_k=3)

        # Should be sorted by importance descending
        expected = [('feature2', 0.5), ('feature1', -0.1), ('feature3', -0.2)]
        assert result == expected


class TestIntegration:
    """Test integration scenarios."""

    @patch('theta_sketch_tree.load_sketches')
    @patch('theta_sketch_tree.load_config')
    @patch('theta_sketch_tree.classifier.ThetaSketchDecisionTreeClassifier')
    def test_full_workflow_integration(self, mock_cls, mock_load_config, mock_load_sketches):
        """Test full workflow from CSV to fitted model."""
        # Setup mocks
        mock_config = {
            'targets': {'positive': 'yes', 'negative': 'no'},
            'hyperparameters': {'criterion': 'gini', 'max_depth': 5},
            'feature_mapping': {'feature1': 0}
        }
        mock_sketch_data = {'positive': {}, 'negative': {}}

        mock_load_config.return_value = mock_config
        mock_load_sketches.return_value = mock_sketch_data

        mock_classifier = Mock()
        mock_cls.return_value = mock_classifier

        # Test the full workflow
        result = ClassifierUtils.fit_from_csv(
            positive_csv="pos.csv",
            negative_csv="neg.csv",
            config_path="config.yaml"
        )

        # Verify complete workflow
        mock_load_sketches.assert_called_once()
        mock_load_config.assert_called_once()
        mock_cls.assert_called_once_with(**mock_config['hyperparameters'])
        mock_classifier.fit.assert_called_once_with(mock_sketch_data, mock_config['feature_mapping'])
        assert result == mock_classifier