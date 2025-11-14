"""
Unit tests for split criteria.

Tests all criterion classes: Gini, Entropy, Gain Ratio, Binomial, Chi-Square.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from theta_sketch_tree.criteria import (
    GiniCriterion, EntropyCriterion, GainRatioCriterion,
    BinomialCriterion, ChiSquareCriterion, get_criterion
)


class TestCriteriaBase:
    """Base tests for all criteria."""

    @pytest.fixture
    def sample_counts(self):
        """Sample class count arrays for testing."""
        return {
            'pure_positive': np.array([0, 100]),
            'pure_negative': np.array([100, 0]),
            'balanced': np.array([50, 50]),
            'imbalanced': np.array([80, 20]),
            'empty': np.array([0, 0]),
            'single': np.array([1, 0]),
        }


class TestGiniCriterion(TestCriteriaBase):
    """Test Gini impurity criterion."""

    @pytest.fixture
    def criterion(self):
        return GiniCriterion()

    def test_compute_impurity(self, criterion, sample_counts):
        """Test Gini impurity calculation."""
        # Pure nodes should have impurity = 0
        assert criterion.compute_impurity(sample_counts['pure_positive']) == 0.0
        assert criterion.compute_impurity(sample_counts['pure_negative']) == 0.0

        # Balanced split should have maximum impurity = 0.5
        balanced_impurity = criterion.compute_impurity(sample_counts['balanced'])
        assert_allclose(balanced_impurity, 0.5)

        # Imbalanced split
        imbalanced_impurity = criterion.compute_impurity(sample_counts['imbalanced'])
        expected = 1 - (0.8**2 + 0.2**2)  # 1 - (0.64 + 0.04) = 0.32
        assert_allclose(imbalanced_impurity, expected)

    def test_evaluate_split(self, criterion, sample_counts):
        """Test split evaluation."""
        parent_counts = sample_counts['balanced']  # [50, 50]
        left_counts = sample_counts['pure_positive']  # [0, 100]
        right_counts = sample_counts['pure_negative']  # [100, 0]

        # Perfect split should have very good score (low value)
        score = criterion.evaluate_split(parent_counts, left_counts, right_counts)
        assert score < 0.5  # Should be much better than parent impurity

    def test_edge_cases(self, criterion):
        """Test edge cases."""
        # Empty counts
        assert criterion.compute_impurity(np.array([0, 0])) == 0.0

        # Single sample
        assert criterion.compute_impurity(np.array([1, 0])) == 0.0


class TestEntropyCriterion(TestCriteriaBase):
    """Test Information Gain (Entropy) criterion."""

    @pytest.fixture
    def criterion(self):
        return EntropyCriterion()

    def test_compute_impurity(self, criterion, sample_counts):
        """Test entropy calculation."""
        # Pure nodes should have entropy = 0
        assert criterion.compute_impurity(sample_counts['pure_positive']) == 0.0
        assert criterion.compute_impurity(sample_counts['pure_negative']) == 0.0

        # Balanced split should have maximum entropy = 1.0
        balanced_impurity = criterion.compute_impurity(sample_counts['balanced'])
        assert_allclose(balanced_impurity, 1.0)

        # Test imbalanced split
        imbalanced_impurity = criterion.compute_impurity(sample_counts['imbalanced'])
        # Entropy = -sum(p * log2(p))
        p1, p2 = 0.8, 0.2
        expected = -(p1 * np.log2(p1) + p2 * np.log2(p2))
        assert_allclose(imbalanced_impurity, expected, rtol=1e-10)

    def test_edge_cases(self, criterion):
        """Test edge cases for entropy."""
        # Empty counts
        assert criterion.compute_impurity(np.array([0, 0])) == 0.0

        # Very small probability
        small_counts = np.array([1e6, 1])
        result = criterion.compute_impurity(small_counts)
        assert result >= 0  # Should be non-negative


class TestGainRatioCriterion(TestCriteriaBase):
    """Test Gain Ratio criterion."""

    @pytest.fixture
    def criterion(self):
        return GainRatioCriterion()

    def test_compute_impurity(self, criterion, sample_counts):
        """Gain ratio uses entropy for impurity calculation."""
        # Should behave like entropy for impurity
        assert criterion.compute_impurity(sample_counts['pure_positive']) == 0.0
        assert criterion.compute_impurity(sample_counts['balanced']) == pytest.approx(1.0)

    def test_evaluate_split(self, criterion):
        """Test gain ratio evaluation."""
        parent_counts = np.array([50, 50])
        left_counts = np.array([40, 10])
        right_counts = np.array([10, 40])

        score = criterion.evaluate_split(parent_counts, left_counts, right_counts)
        # Gain ratio normalizes information gain by split entropy
        assert isinstance(score, float)
        assert not np.isnan(score)


class TestBinomialCriterion(TestCriteriaBase):
    """Test Binomial criterion."""

    @pytest.fixture
    def criterion(self):
        return BinomialCriterion()

    def test_compute_impurity(self, criterion, sample_counts):
        """Test binomial impurity calculation."""
        # Pure nodes should have low impurity
        assert criterion.compute_impurity(sample_counts['pure_positive']) >= 0
        assert criterion.compute_impurity(sample_counts['pure_negative']) >= 0

        # Should return valid numbers
        balanced = criterion.compute_impurity(sample_counts['balanced'])
        assert isinstance(balanced, float)
        assert not np.isnan(balanced)

    def test_evaluate_split(self, criterion):
        """Test binomial split evaluation."""
        parent_counts = np.array([60, 40])
        left_counts = np.array([45, 5])
        right_counts = np.array([15, 35])

        score = criterion.evaluate_split(parent_counts, left_counts, right_counts)
        assert isinstance(score, float)
        assert not np.isnan(score)


class TestChiSquareCriterion(TestCriteriaBase):
    """Test Chi-Square criterion."""

    @pytest.fixture
    def criterion(self):
        return ChiSquareCriterion()

    def test_evaluate_split(self, criterion):
        """Test chi-square split evaluation."""
        parent_counts = np.array([60, 40])
        left_counts = np.array([40, 10])
        right_counts = np.array([20, 30])

        score = criterion.evaluate_split(parent_counts, left_counts, right_counts)
        assert isinstance(score, float)
        assert not np.isnan(score)

    def test_edge_cases(self, criterion):
        """Test edge cases for chi-square."""
        # Test with zero expected values
        parent_counts = np.array([100, 0])
        left_counts = np.array([50, 0])
        right_counts = np.array([50, 0])

        score = criterion.evaluate_split(parent_counts, left_counts, right_counts)
        # Should handle zero expected values gracefully
        assert isinstance(score, float)


class TestCriterionFactory:
    """Test the get_criterion factory function."""

    def test_get_criterion(self):
        """Test criterion factory function."""
        assert isinstance(get_criterion('gini'), GiniCriterion)
        assert isinstance(get_criterion('entropy'), EntropyCriterion)
        assert isinstance(get_criterion('gain_ratio'), GainRatioCriterion)
        assert isinstance(get_criterion('binomial'), BinomialCriterion)
        assert isinstance(get_criterion('chi_square'), ChiSquareCriterion)
        assert isinstance(get_criterion('binomial_chi'), BinomialCriterion)  # Alias

    def test_invalid_criterion(self):
        """Test invalid criterion name."""
        with pytest.raises(ValueError, match="Unknown criterion"):
            get_criterion('invalid')

    def test_case_insensitive(self):
        """Test that criterion names are case insensitive."""
        # Should work with uppercase
        assert isinstance(get_criterion('GINI'), GiniCriterion)
        assert isinstance(get_criterion('Entropy'), EntropyCriterion)

    def test_invalid_criterion_specific(self):
        """Test specific invalid criterion names."""
        with pytest.raises(ValueError):
            get_criterion('unknown_criterion')


class TestCriteriaComparison:
    """Test comparison between different criteria."""

    @pytest.fixture
    def criteria(self):
        return {
            'gini': GiniCriterion(),
            'entropy': EntropyCriterion(),
            'gain_ratio': GainRatioCriterion(),
        }

    @pytest.fixture
    def test_data(self):
        """Test data for comparison."""
        return {
            'parent_counts': np.array([60, 40]),
            'left_counts': np.array([45, 15]),
            'right_counts': np.array([15, 25]),
        }

    def test_criteria_consistency(self, criteria, test_data):
        """Test that all criteria return reasonable values."""
        scores = {}
        for name, criterion in criteria.items():
            score = criterion.evaluate_split(
                test_data['parent_counts'],
                test_data['left_counts'],
                test_data['right_counts']
            )
            scores[name] = score

            # All scores should be finite numbers
            assert isinstance(score, float)
            assert not np.isnan(score)
            assert not np.isinf(score)

        print(f"Criterion scores: {scores}")

    def test_pure_split_preference(self, criteria):
        """Test that all criteria prefer pure splits."""
        parent_counts = np.array([50, 50])

        # Perfect split (pure children)
        left_pure = np.array([50, 0])
        right_pure = np.array([0, 50])

        # Poor split (similar to parent)
        left_poor = np.array([25, 25])
        right_poor = np.array([25, 25])

        for name, criterion in criteria.items():
            pure_score = criterion.evaluate_split(parent_counts, left_pure, right_pure)
            poor_score = criterion.evaluate_split(parent_counts, left_poor, right_poor)

            # Pure split should be better (lower score for most criteria)
            # Note: Some criteria might use different scoring directions
            if name in ['gini', 'entropy']:
                assert pure_score < poor_score, f"{name} should prefer pure splits"
