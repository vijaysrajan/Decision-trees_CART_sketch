"""
Comprehensive tests for split_finder module.

Tests SplitFinder class and related functionality.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from theta_sketch_tree.split_finder import SplitFinder, SplitResult, find_best_split
from theta_sketch_tree.criteria import GiniCriterion
from tests.test_mock_sketches import MockThetaSketch


@pytest.fixture
def mock_criterion():
    """Create a mock criterion for testing."""
    criterion = Mock()
    criterion.evaluate_split.return_value = 0.1  # Lower is better
    return criterion


@pytest.fixture
def basic_split_finder(mock_criterion):
    """Create basic split finder for testing."""
    return SplitFinder(criterion=mock_criterion, min_samples_leaf=1, verbose=0)


@pytest.fixture
def verbose_split_finder(mock_criterion):
    """Create verbose split finder for testing."""
    return SplitFinder(criterion=mock_criterion, min_samples_leaf=1, verbose=3)


@pytest.fixture
def simple_sketch_dict():
    """Create simple sketch dictionary for testing."""
    return {
        'positive': {
            'feature1': (MockThetaSketch(50), MockThetaSketch(50)),  # present, absent
            'feature2': (MockThetaSketch(30), MockThetaSketch(70)),
        },
        'negative': {
            'feature1': (MockThetaSketch(40), MockThetaSketch(60)),
            'feature2': (MockThetaSketch(20), MockThetaSketch(80)),
        }
    }


@pytest.fixture
def empty_sketch_dict():
    """Create empty sketch dictionary."""
    return {
        'positive': {},
        'negative': {}
    }


class TestSplitFinder:
    """Test suite for SplitFinder class."""

    def test_initialization(self):
        """Test SplitFinder initialization."""
        criterion = Mock()
        finder = SplitFinder(criterion=criterion, min_samples_leaf=5, verbose=2)

        assert finder.criterion == criterion
        assert finder.min_samples_leaf == 5
        assert finder.verbose == 2
        assert finder.logger is not None

    def test_find_best_split_basic(self, basic_split_finder, simple_sketch_dict):
        """Test basic split finding functionality."""
        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(100)
        parent_counts = np.array([100, 100])

        result = basic_split_finder.find_best_split(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            parent_class_counts=parent_counts,
            parent_impurity=0.5,
            sketch_dict=simple_sketch_dict,
            feature_names=['feature1', 'feature2'],
            already_used=set()
        )

        assert result is not None
        assert isinstance(result, SplitResult)
        assert result.feature_name in ['feature1', 'feature2']

    def test_find_best_split_no_available_features(self, basic_split_finder, simple_sketch_dict):
        """Test when no features are available (already used)."""
        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(100)
        parent_counts = np.array([100, 100])

        result = basic_split_finder.find_best_split(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            parent_class_counts=parent_counts,
            parent_impurity=0.5,
            sketch_dict=simple_sketch_dict,
            feature_names=['feature1', 'feature2'],
            already_used={'feature1', 'feature2'}  # All features already used
        )

        assert result is None

    def test_find_best_split_verbose_no_features(self, verbose_split_finder, empty_sketch_dict):
        """Test verbose output when no features available."""
        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(100)
        parent_counts = np.array([100, 100])

        result = verbose_split_finder.find_best_split(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            parent_class_counts=parent_counts,
            parent_impurity=0.5,
            sketch_dict=empty_sketch_dict,
            feature_names=[],
            already_used=set()
        )

        # Should return None when no features available
        assert result is None

    def test_find_best_split_verbose_new_best(self, verbose_split_finder, simple_sketch_dict, caplog):
        """Test verbose output for new best split."""
        import logging
        caplog.set_level(logging.DEBUG)

        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(100)
        parent_counts = np.array([100, 100])

        # Mock criterion to return decreasing scores
        verbose_split_finder.criterion.evaluate_split.side_effect = [0.3, 0.1]  # Second is better

        result = verbose_split_finder.find_best_split(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            parent_class_counts=parent_counts,
            parent_impurity=0.5,
            sketch_dict=simple_sketch_dict,
            feature_names=['feature1', 'feature2'],
            already_used=set()
        )

        assert result is not None
        # Check that we found a result (logging may vary)
        assert result.score == 0.1  # Should have the better score

    def test_get_available_features(self, basic_split_finder):
        """Test feature availability filtering."""
        sketch_dict = {
            'positive': {'feature1': 'mock', 'feature2': 'mock'},
            'negative': {'feature1': 'mock', 'feature2': 'mock', 'feature3': 'mock'}
        }

        # Test with no features used
        available = basic_split_finder._get_available_features(
            feature_names=['feature1', 'feature2', 'feature3', 'feature4'],
            already_used=set(),
            sketch_dict=sketch_dict
        )
        # Only feature1 and feature2 are in both positive and negative
        assert set(available) == {'feature1', 'feature2'}

        # Test with some features already used
        available = basic_split_finder._get_available_features(
            feature_names=['feature1', 'feature2'],
            already_used={'feature1'},
            sketch_dict=sketch_dict
        )
        assert available == ['feature2']

        # Test when feature not in sketch dict
        available = basic_split_finder._get_available_features(
            feature_names=['nonexistent_feature'],
            already_used=set(),
            sketch_dict=sketch_dict
        )
        assert available == []

    def test_evaluate_feature_split_invalid_structure(self, basic_split_finder):
        """Test handling of invalid sketch structure."""
        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(100)
        parent_counts = np.array([100, 100])

        # Create sketch dict with invalid structure (missing indices)
        invalid_sketch_dict = {
            'positive': {
                'feature1': ['only_one_element']  # Should have 2 elements
            },
            'negative': {
                'feature1': (MockThetaSketch(40), MockThetaSketch(60))
            }
        }

        result = basic_split_finder._evaluate_feature_split(
            feature_name='feature1',
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            parent_class_counts=parent_counts,
            parent_impurity=0.5,
            sketch_dict=invalid_sketch_dict
        )

        assert result is None

    def test_evaluate_feature_split_empty_child(self, basic_split_finder):
        """Test handling of empty child nodes."""
        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(100)
        parent_counts = np.array([100, 100])

        # Create sketches that will result in empty children
        empty_present = MockThetaSketch(0)  # Empty sketch
        empty_absent = MockThetaSketch(0)   # Empty sketch

        sketch_dict = {
            'positive': {
                'feature1': (empty_present, empty_absent)
            },
            'negative': {
                'feature1': (empty_present, empty_absent)
            }
        }

        result = basic_split_finder._evaluate_feature_split(
            feature_name='feature1',
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            parent_class_counts=parent_counts,
            parent_impurity=0.5,
            sketch_dict=sketch_dict
        )

        assert result is None

    def test_evaluate_feature_split_min_samples_violation(self):
        """Test handling of min_samples_leaf constraint violation."""
        # Create split finder with high min_samples_leaf
        high_min_finder = SplitFinder(
            criterion=Mock(),
            min_samples_leaf=50,  # High threshold
            verbose=0
        )

        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(100)
        parent_counts = np.array([100, 100])

        # Create sketches that will result in small children
        small_present = MockThetaSketch(10)  # Small sketch - violates min_samples_leaf
        large_absent = MockThetaSketch(90)

        sketch_dict = {
            'positive': {
                'feature1': (small_present, large_absent)
            },
            'negative': {
                'feature1': (small_present, large_absent)
            }
        }

        result = high_min_finder._evaluate_feature_split(
            feature_name='feature1',
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            parent_class_counts=parent_counts,
            parent_impurity=0.5,
            sketch_dict=sketch_dict
        )

        assert result is None

    def test_validate_min_samples_valid(self, basic_split_finder):
        """Test min samples validation with valid children."""
        left_counts = np.array([10, 15])
        right_counts = np.array([20, 25])

        valid = basic_split_finder._validate_min_samples(left_counts, right_counts)
        assert valid == True

    def test_validate_min_samples_invalid_left(self, basic_split_finder):
        """Test min samples validation with invalid left child."""
        # Create finder with min_samples_leaf=10
        strict_finder = SplitFinder(criterion=Mock(), min_samples_leaf=10)

        left_counts = np.array([2, 3])  # Total = 5, less than min_samples_leaf=10
        right_counts = np.array([20, 25])

        valid = strict_finder._validate_min_samples(left_counts, right_counts)
        assert valid == False

    def test_validate_min_samples_invalid_right(self, basic_split_finder):
        """Test min samples validation with invalid right child."""
        strict_finder = SplitFinder(criterion=Mock(), min_samples_leaf=10)

        left_counts = np.array([10, 15])
        right_counts = np.array([2, 3])  # Total = 5, less than min_samples_leaf=10

        valid = strict_finder._validate_min_samples(left_counts, right_counts)
        assert valid == False

    def test_validate_min_samples_boundary(self, basic_split_finder):
        """Test min samples validation at boundary."""
        boundary_finder = SplitFinder(criterion=Mock(), min_samples_leaf=5)

        left_counts = np.array([2, 3])  # Total = 5, exactly equal to min_samples_leaf
        right_counts = np.array([3, 2])  # Total = 5, exactly equal to min_samples_leaf

        valid = boundary_finder._validate_min_samples(left_counts, right_counts)
        assert valid == True

    def test_split_result_creation(self, basic_split_finder, simple_sketch_dict):
        """Test SplitResult creation with valid split."""
        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(100)
        parent_counts = np.array([100, 100])

        # Mock criterion to return valid score
        basic_split_finder.criterion.evaluate_split.return_value = 0.2

        result = basic_split_finder._evaluate_feature_split(
            feature_name='feature1',
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            parent_class_counts=parent_counts,
            parent_impurity=0.5,
            sketch_dict=simple_sketch_dict
        )

        assert result is not None
        assert result.feature_name == 'feature1'
        assert result.score == 0.2
        assert result.left_sketch_pos is not None
        assert result.left_sketch_neg is not None
        assert result.right_sketch_pos is not None
        assert result.right_sketch_neg is not None
        assert len(result.left_counts) == 2
        assert len(result.right_counts) == 2


class TestConvenienceFunction:
    """Test the convenience function."""

    def test_find_best_split_function(self, simple_sketch_dict):
        """Test find_best_split convenience function."""
        criterion = GiniCriterion()
        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(100)
        parent_counts = np.array([100, 100])

        result = find_best_split(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            parent_class_counts=parent_counts,
            parent_impurity=0.5,
            sketch_dict=simple_sketch_dict,
            feature_names=['feature1', 'feature2'],
            already_used=set(),
            criterion=criterion,
            min_samples_leaf=1,
            verbose=0
        )

        assert result is not None
        assert isinstance(result, SplitResult)

    def test_find_best_split_function_with_verbose(self, simple_sketch_dict, caplog):
        """Test find_best_split convenience function with verbose output."""
        criterion = GiniCriterion()
        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(100)
        parent_counts = np.array([100, 100])

        result = find_best_split(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            parent_class_counts=parent_counts,
            parent_impurity=0.5,
            sketch_dict=simple_sketch_dict,
            feature_names=['feature1', 'feature2'],
            already_used=set(),
            criterion=criterion,
            min_samples_leaf=1,
            verbose=3  # High verbosity
        )

        assert result is not None


class TestSplitResult:
    """Test SplitResult NamedTuple."""

    def test_split_result_creation(self):
        """Test creating SplitResult directly."""
        mock_sketch = MockThetaSketch(50)
        left_counts = np.array([30, 20])
        right_counts = np.array([20, 30])

        result = SplitResult(
            feature_name='test_feature',
            score=0.15,
            left_sketch_pos=mock_sketch,
            left_sketch_neg=mock_sketch,
            right_sketch_pos=mock_sketch,
            right_sketch_neg=mock_sketch,
            left_counts=left_counts,
            right_counts=right_counts
        )

        assert result.feature_name == 'test_feature'
        assert result.score == 0.15
        assert result.left_sketch_pos == mock_sketch
        assert np.array_equal(result.left_counts, left_counts)
        assert np.array_equal(result.right_counts, right_counts)

    def test_split_result_immutable(self):
        """Test that SplitResult is immutable (NamedTuple property)."""
        mock_sketch = MockThetaSketch(50)
        result = SplitResult(
            feature_name='test_feature',
            score=0.15,
            left_sketch_pos=mock_sketch,
            left_sketch_neg=mock_sketch,
            right_sketch_pos=mock_sketch,
            right_sketch_neg=mock_sketch,
            left_counts=np.array([30, 20]),
            right_counts=np.array([20, 30])
        )

        # Should not be able to modify fields (NamedTuple is immutable)
        with pytest.raises(AttributeError):
            result.feature_name = 'new_name'


class TestIntegration:
    """Integration tests for split finding."""

    def test_multiple_features_best_selection(self, simple_sketch_dict):
        """Test that split finder selects the best feature among multiple options."""
        # Create criterion that returns different scores for different features
        mock_criterion = Mock()

        def criterion_side_effect(parent, left, right, parent_impurity=None):
            # Return different scores based on call order
            # This is a bit artificial but tests the selection logic
            if not hasattr(criterion_side_effect, 'call_count'):
                criterion_side_effect.call_count = 0
            criterion_side_effect.call_count += 1

            if criterion_side_effect.call_count == 1:
                return 0.4  # Worse score for first feature
            else:
                return 0.1  # Better score for second feature

        mock_criterion.evaluate_split.side_effect = criterion_side_effect

        finder = SplitFinder(criterion=mock_criterion, min_samples_leaf=1, verbose=0)
        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(100)
        parent_counts = np.array([100, 100])

        result = finder.find_best_split(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            parent_class_counts=parent_counts,
            parent_impurity=0.5,
            sketch_dict=simple_sketch_dict,
            feature_names=['feature1', 'feature2'],
            already_used=set()
        )

        assert result is not None
        # Should select the feature with better (lower) score
        assert result.score == 0.1

    def test_edge_case_single_feature(self, basic_split_finder):
        """Test split finding with only one available feature."""
        single_feature_dict = {
            'positive': {
                'only_feature': (MockThetaSketch(50), MockThetaSketch(50))
            },
            'negative': {
                'only_feature': (MockThetaSketch(40), MockThetaSketch(60))
            }
        }

        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(100)
        parent_counts = np.array([100, 100])

        result = basic_split_finder.find_best_split(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            parent_class_counts=parent_counts,
            parent_impurity=0.5,
            sketch_dict=single_feature_dict,
            feature_names=['only_feature'],
            already_used=set()
        )

        assert result is not None
        assert result.feature_name == 'only_feature'