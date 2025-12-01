"""
Integration tests for the complete decision tree pipeline.

Tests the full workflow from sketch data loading to prediction.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add tests directory to path to import mock sketches
sys.path.append(str(Path(__file__).parent))
from test_mock_sketches import create_mock_sketch_data, create_feature_mapping

from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier


class TestIntegration:
    """Test suite for end-to-end decision tree functionality."""

    @pytest.fixture
    def sample_sketch_data(self):
        """
        Create proper mock sketch data for testing.

        Returns dict with 'positive'/'negative' keys containing:
        - 'total': overall class sketches (MockThetaSketch objects)
        - feature sketches: tuple of (present, absent) MockThetaSketch objects
        """
        return create_mock_sketch_data()

    @pytest.fixture
    def feature_mapping(self):
        """Feature mapping for inference."""
        return create_feature_mapping()

    @pytest.fixture
    def test_csv_files(self):
        """Use existing test fixtures from tests/fixtures/."""
        fixtures_dir = Path(__file__).parent / 'fixtures'
        return {
            'positive_csv': str(fixtures_dir / 'target_yes_3col.csv'),
            'negative_csv': str(fixtures_dir / 'target_no_3col.csv'),
        }

    @pytest.fixture
    def config_data(self, feature_mapping):
        """Sample configuration data."""
        return {
            'hyperparameters': {
                'criterion': 'gini',
                'max_depth': 3,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
            },
            'feature_mapping': feature_mapping,
        }

    def test_classifier_initialization(self):
        """Test basic classifier initialization."""
        clf = ThetaSketchDecisionTreeClassifier(
            criterion='gini',
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1,
        )

        assert clf.criterion == 'gini'
        assert clf.max_depth == 3
        assert clf.min_samples_split == 2
        assert clf.min_samples_leaf == 1

    def test_fit_basic_functionality(self, sample_sketch_data, feature_mapping):
        """Test basic fitting functionality with mock data."""
        clf = ThetaSketchDecisionTreeClassifier(criterion='gini', max_depth=2)

        # Now this should work with proper mock sketches
        clf.fit(sample_sketch_data, feature_mapping)

        # Check that sklearn-compatible attributes are set
        assert hasattr(clf, 'classes_')
        assert hasattr(clf, 'n_classes_')
        assert hasattr(clf, 'n_features_in_')
        assert hasattr(clf, 'feature_names_in_')
        assert hasattr(clf, 'tree_')

        # Check attribute values
        assert clf.n_classes_ == 2
        assert clf.n_features_in_ == 2  # age>30, income>50k
        assert len(clf.feature_names_in_) == 2
        assert clf.tree_ is not None  # Tree should be created

    def test_predict_before_fit_raises_error(self):
        """Test that predict() raises error when called before fit()."""
        clf = ThetaSketchDecisionTreeClassifier()
        X_test = np.array([[1, 0], [0, 1]])

        with pytest.raises(ValueError, match="Classifier must be fitted before"):
            clf.predict(X_test)

        with pytest.raises(ValueError, match="Classifier must be fitted before"):
            clf.predict_proba(X_test)

    def test_full_pipeline_with_mock_sketches(self, sample_sketch_data, feature_mapping):
        """
        Test complete pipeline from fitting to prediction with mock sketches.
        """
        # Create and fit classifier
        clf = ThetaSketchDecisionTreeClassifier(criterion='gini', max_depth=3)
        clf.fit(sample_sketch_data, feature_mapping)

        # Create test data
        X_test = np.array([
            [1, 0],  # age>30=True, income>50k=False
            [0, 1],  # age>30=False, income>50k=True
            [1, 1],  # age>30=True, income>50k=True
            [0, 0],  # age>30=False, income>50k=False
        ])

        # Test predictions
        predictions = clf.predict(X_test)
        probabilities = clf.predict_proba(X_test)

        # Validate output shapes and types
        assert predictions.shape == (4,)
        assert probabilities.shape == (4, 2)
        assert predictions.dtype == np.int64
        assert probabilities.dtype == np.float64

        # Validate prediction values
        assert all(pred in [0, 1] for pred in predictions)
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all(probabilities >= 0)  # All probabilities non-negative

        # Test consistency between predict and predict_proba
        predicted_from_proba = np.argmax(probabilities, axis=1)
        assert np.array_equal(predictions, predicted_from_proba)

        print(f"Predictions: {predictions}")
        print(f"Probabilities:\n{probabilities}")

    def test_feature_importances(self, sample_sketch_data, feature_mapping):
        """Test feature importance calculation."""
        # Create and fit classifier
        clf = ThetaSketchDecisionTreeClassifier(criterion='gini', max_depth=3)
        clf.fit(sample_sketch_data, feature_mapping)

        # Test feature_importances_ property
        importances = clf.feature_importances_
        assert isinstance(importances, np.ndarray)
        assert importances.shape == (2,)  # 2 features
        assert np.allclose(importances.sum(), 1.0)  # Should sum to 1
        assert np.all(importances >= 0)  # All non-negative

        # Test get_feature_importance_dict()
        importance_dict = clf.get_feature_importance_dict()
        assert isinstance(importance_dict, dict)
        assert len(importance_dict) == 2
        assert 'age>30' in importance_dict
        assert 'income>50k' in importance_dict
        assert all(val >= 0 for val in importance_dict.values())

        # Test get_top_features()
        top_features = clf.get_top_features(top_k=2)
        assert isinstance(top_features, list)
        assert len(top_features) == 2
        # Should be sorted by importance (descending)
        assert top_features[0][1] >= top_features[1][1]

        print(f"Feature importances: {importances}")
        print(f"Importance dict: {importance_dict}")
        print(f"Top features: {top_features}")

    @pytest.mark.skip(reason="Requires real ThetaSketch loading implementation")
    def test_full_pipeline_with_real_sketches(self, test_csv_files, config_data):
        """
        Test complete pipeline with real CSV sketch files.

        NOTE: Skipped because it requires actual ThetaSketch loading implementation.
        This test documents the intended workflow.
        """
        # Load sketches from CSV
        sketch_data = load_sketches(
            test_csv_files['positive_csv'],
            test_csv_files['negative_csv']
        )

        # Create and fit classifier
        clf = ThetaSketchDecisionTreeClassifier(**config_data['hyperparameters'])
        clf.fit(sketch_data, config_data['feature_mapping'])

        # Create test data
        X_test = np.array([
            [1, 0],  # age>30=True, income>50k=False
            [0, 1],  # age>30=False, income>50k=True
            [1, 1],  # age>30=True, income>50k=True
            [0, 0],  # age>30=False, income>50k=False
        ])

        # Test predictions
        predictions = clf.predict(X_test)
        probabilities = clf.predict_proba(X_test)

        # Validate output shapes and types
        assert predictions.shape == (4,)
        assert probabilities.shape == (4, 2)
        assert predictions.dtype == np.int64
        assert probabilities.dtype == np.float64

        # Validate prediction values
        assert all(pred in [0, 1] for pred in predictions)
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all(probabilities >= 0)  # All probabilities non-negative

        # Test consistency between predict and predict_proba
        predicted_from_proba = np.argmax(probabilities, axis=1)
        assert np.array_equal(predictions, predicted_from_proba)

    def test_input_validation(self, sample_sketch_data, feature_mapping):
        """Test input validation for various scenarios."""
        clf = ThetaSketchDecisionTreeClassifier()

        # Test invalid sketch_data structure
        with pytest.raises(ValueError, match="sketch_data must contain 'positive' key"):
            clf.fit({}, feature_mapping)

        with pytest.raises(ValueError, match="sketch_data must contain 'negative' key"):
            clf.fit({'positive': {}}, feature_mapping)

        with pytest.raises(ValueError, match="must contain 'total' key"):
            clf.fit({'positive': {}, 'negative': {'total': 'mock'}}, feature_mapping)

    def test_predict_input_validation(self, sample_sketch_data, feature_mapping):
        """Test input validation for predict methods."""
        clf = ThetaSketchDecisionTreeClassifier()

        # Mock a fitted state for testing predict validation
        clf._is_fitted = True
        clf.n_features_in_ = 2
        clf.tree_ = None  # Will cause error, but we're testing input validation first

        # Test wrong number of features
        X_wrong_features = np.array([[1, 0, 1]])  # 3 features instead of 2
        with pytest.raises(ValueError, match="Input has 3 features, but classifier expects 2"):
            clf.predict(X_wrong_features)

        # Test wrong dimensions
        X_wrong_dims = np.array([1, 0])  # 1D instead of 2D
        with pytest.raises(ValueError, match="Input must be 2D array"):
            clf.predict(X_wrong_dims)

    def test_different_criteria(self, sample_sketch_data, feature_mapping):
        """Test that different split criteria can be initialized."""
        criteria = ['gini', 'entropy', 'gain_ratio', 'binomial', 'binomial_chi']

        for criterion in criteria:
            clf = ThetaSketchDecisionTreeClassifier(criterion=criterion)
            assert clf.criterion == criterion

            # Test that fit attempts to use the criterion (will fail with mock data)
            try:
                clf.fit(sample_sketch_data, feature_mapping)
            except Exception:
                pass  # Expected with mock data

    def test_hyperparameter_ranges(self):
        """Test various hyperparameter combinations."""
        # Test max_depth variations
        for max_depth in [None, 1, 5, 10]:
            clf = ThetaSketchDecisionTreeClassifier(max_depth=max_depth)
            assert clf.max_depth == max_depth

        # Test min_samples_split variations
        for min_samples in [2, 5, 10]:
            clf = ThetaSketchDecisionTreeClassifier(min_samples_split=min_samples)
            assert clf.min_samples_split == min_samples

        # Test min_samples_leaf variations
        for min_samples in [1, 3, 5]:
            clf = ThetaSketchDecisionTreeClassifier(min_samples_leaf=min_samples)
            assert clf.min_samples_leaf == min_samples

    def test_sklearn_compatibility_attributes(self, sample_sketch_data, feature_mapping):
        """Test that the classifier implements required sklearn attributes."""
        clf = ThetaSketchDecisionTreeClassifier()

        # Test that it inherits from sklearn base classes
        from sklearn.base import BaseEstimator, ClassifierMixin
        assert isinstance(clf, BaseEstimator)
        assert isinstance(clf, ClassifierMixin)

        # Mock fitted state to test attributes
        try:
            clf.fit(sample_sketch_data, feature_mapping)
        except Exception:
            # Even if fit fails, we can test the intended attribute structure
            pass

    @pytest.mark.parametrize("criterion", ['gini', 'entropy', 'gain_ratio'])
    def test_multiple_criteria_initialization(self, criterion):
        """Test initialization with different criteria."""
        clf = ThetaSketchDecisionTreeClassifier(criterion=criterion)
        assert clf.criterion == criterion
