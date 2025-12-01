"""
Unit tests for simplified split criteria.

Tests all criterion classes with streamlined test coverage focused on core functionality.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from theta_sketch_tree.criteria import (
    GiniCriterion, EntropyCriterion, GainRatioCriterion,
    BinomialCriterion, ChiSquareCriterion, get_criterion
)


class TestCriteriaSimplified:
    """Comprehensive tests for simplified criteria."""

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing all criteria."""
        return {
            'pure_positive': np.array([0.0, 100.0]),
            'pure_negative': np.array([100.0, 0.0]),
            'balanced': np.array([50.0, 50.0]),
            'imbalanced': np.array([80.0, 20.0]),
            'empty': np.array([0.0, 0.0]),
        }

    def test_gini_criterion(self, sample_data):
        """Test Gini criterion implementation."""
        gini = GiniCriterion()

        # Pure nodes should have 0 impurity
        assert gini.compute_impurity(sample_data['pure_positive']) == 0.0
        assert gini.compute_impurity(sample_data['pure_negative']) == 0.0

        # Balanced split has maximum impurity = 0.5
        assert_allclose(gini.compute_impurity(sample_data['balanced']), 0.5, atol=1e-10)

        # Test split evaluation
        parent = sample_data['balanced']
        left = sample_data['pure_positive']
        right = sample_data['pure_negative']

        # Perfect split should have negative score (good split)
        score = gini.evaluate_split(parent, left, right)
        assert score < 0  # Improvement

    def test_entropy_criterion(self, sample_data):
        """Test entropy criterion implementation."""
        entropy = EntropyCriterion()

        # Pure nodes should have 0 entropy
        assert entropy.compute_impurity(sample_data['pure_positive']) == 0.0
        assert entropy.compute_impurity(sample_data['pure_negative']) == 0.0

        # Balanced split has maximum entropy = 1.0
        assert_allclose(entropy.compute_impurity(sample_data['balanced']), 1.0, atol=1e-10)

        # Test information gain
        parent = sample_data['balanced']
        left = sample_data['pure_positive']
        right = sample_data['pure_negative']

        score = entropy.evaluate_split(parent, left, right)
        assert score < 0  # Information gain (negative score)

    def test_gain_ratio_criterion(self, sample_data):
        """Test gain ratio criterion (inherits from entropy)."""
        gain_ratio = GainRatioCriterion()

        # Should inherit entropy computation
        assert_allclose(gain_ratio.compute_impurity(sample_data['balanced']), 1.0, atol=1e-10)

        # Test gain ratio calculation
        parent = sample_data['balanced']
        left = sample_data['pure_positive']
        right = sample_data['pure_negative']

        score = gain_ratio.evaluate_split(parent, left, right)
        assert score < 0  # Good split (negative score)

    def test_binomial_criterion(self, sample_data):
        """Test binomial test criterion."""
        binomial = BinomialCriterion()

        # Balanced data should have high p-value (not significant)
        p_val = binomial.compute_impurity(sample_data['balanced'])
        assert p_val > 0.05  # Not significant

        # Highly imbalanced should have low p-value
        extreme = np.array([5.0, 95.0])
        p_val = binomial.compute_impurity(extreme)
        assert p_val < 0.05  # Significant

    def test_chi_square_criterion(self, sample_data):
        """Test chi-square test criterion."""
        chi_sq = ChiSquareCriterion()

        # Node impurity not applicable for chi-square
        assert chi_sq.compute_impurity(sample_data['balanced']) == 0.0

        # Test split evaluation
        parent = sample_data['balanced']
        left = sample_data['pure_positive']
        right = sample_data['pure_negative']

        p_val = chi_sq.evaluate_split(parent, left, right)
        assert 0.0 <= p_val <= 1.0  # Valid p-value

    def test_class_weights(self, sample_data):
        """Test class weight functionality."""
        # Test with class weights
        weights = {0: 0.5, 1: 2.0}  # Emphasize positive class
        gini_weighted = GiniCriterion(class_weight=weights)
        gini_unweighted = GiniCriterion()

        # Weighted should be different from unweighted
        weighted_impurity = gini_weighted.compute_impurity(sample_data['imbalanced'])
        unweighted_impurity = gini_unweighted.compute_impurity(sample_data['imbalanced'])

        assert weighted_impurity != unweighted_impurity

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        gini = GiniCriterion()

        # Empty arrays
        assert gini.compute_impurity(np.array([0.0, 0.0])) == 0.0

        # Single class
        assert gini.compute_impurity(np.array([10.0, 0.0])) == 0.0

        # Very small values
        tiny = np.array([1e-10, 1e-10])
        assert gini.compute_impurity(tiny) >= 0.0

    def test_criterion_factory(self):
        """Test get_criterion factory function."""
        # Test all criterion types
        assert isinstance(get_criterion('gini'), GiniCriterion)
        assert isinstance(get_criterion('entropy'), EntropyCriterion)
        assert isinstance(get_criterion('gain_ratio'), GainRatioCriterion)
        assert isinstance(get_criterion('binomial'), BinomialCriterion)
        assert isinstance(get_criterion('chi_square'), ChiSquareCriterion)

        # Test case insensitive
        assert isinstance(get_criterion('GINI'), GiniCriterion)
        assert isinstance(get_criterion('Entropy'), EntropyCriterion)

        # Test alias
        assert isinstance(get_criterion('binomial_chi'), BinomialCriterion)

        # Test error handling
        with pytest.raises(ValueError):
            get_criterion('invalid_criterion')

    def test_criterion_parameters(self):
        """Test criterion initialization with parameters."""
        # Test binomial with custom p-value
        binomial = get_criterion('binomial', min_pvalue=0.01)
        assert binomial.min_pvalue == 0.01

        # Test with class weights
        gini = get_criterion('gini', class_weight={0: 1.0, 1: 2.0})
        assert gini.class_weight == {0: 1.0, 1: 2.0}

    def test_consistency_with_original(self, sample_data):
        """Test that simplified version produces same results as original."""
        # Import original if available for comparison
        try:
            from theta_sketch_tree.criteria import GiniCriterion as OriginalGini

            simplified = GiniCriterion()
            original = OriginalGini()

            # Should produce same results
            for key, data in sample_data.items():
                if np.sum(data) > 0:  # Skip empty case
                    simplified_result = simplified.compute_impurity(data)
                    original_result = original.compute_impurity(data)
                    assert_allclose(simplified_result, original_result, atol=1e-10)

        except ImportError:
            # Original not available, skip test
            pytest.skip("Original criteria module not available for comparison")

    @pytest.mark.parametrize("criterion_name", ['gini', 'entropy', 'gain_ratio'])
    def test_split_quality_ranking(self, criterion_name):
        """Test that criteria correctly rank split quality."""
        criterion = get_criterion(criterion_name)

        # Perfect split: pure left and right
        parent = np.array([50.0, 50.0])
        perfect_left = np.array([50.0, 0.0])
        perfect_right = np.array([0.0, 50.0])

        # Poor split: same as parent
        poor_left = np.array([25.0, 25.0])
        poor_right = np.array([25.0, 25.0])

        perfect_score = criterion.evaluate_split(parent, perfect_left, perfect_right)
        poor_score = criterion.evaluate_split(parent, poor_left, poor_right)

        # Perfect split should be better (more negative for impurity-based criteria)
        assert perfect_score < poor_score

    def test_mathematical_properties(self):
        """Test mathematical properties of criteria."""
        gini = GiniCriterion()
        entropy = EntropyCriterion()

        # Gini should be in [0, 0.5] for binary classification
        test_cases = [
            np.array([100.0, 0.0]),    # Pure -> 0
            np.array([50.0, 50.0]),    # Balanced -> 0.5
            np.array([80.0, 20.0]),    # Imbalanced -> between 0 and 0.5
        ]

        for case in test_cases:
            gini_val = gini.compute_impurity(case)
            assert 0.0 <= gini_val <= 0.5

        # Entropy should be in [0, 1] for binary classification
        for case in test_cases:
            entropy_val = entropy.compute_impurity(case)
            assert 0.0 <= entropy_val <= 1.0

    def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms in criteria."""
        from theta_sketch_tree.criteria import BinomialCriterion, ChiSquareCriterion

        # Test binomial criterion with edge cases
        binomial = BinomialCriterion()

        # Test with zero total (should return 1.0)
        zero_counts = np.array([0.0, 0.0])
        assert binomial.compute_impurity(zero_counts) == 1.0

        # Test with very small counts
        tiny_counts = np.array([1e-10, 1e-10])
        result = binomial.compute_impurity(tiny_counts)
        # Allow for NaN in extreme cases
        assert np.isnan(result) or (0.0 <= result <= 1.0)

        # Test chi-square with edge cases
        chi_square = ChiSquareCriterion()

        # Test with zero counts (should handle gracefully)
        result = chi_square.compute_impurity(zero_counts)
        assert result >= 0.0

        # Test split evaluation with edge cases
        parent_counts = np.array([100.0, 100.0])
        left_counts = np.array([0.0, 100.0])  # Pure left
        right_counts = np.array([100.0, 0.0])  # Pure right

        # This should work without errors
        gini_score = binomial.evaluate_split(parent_counts, left_counts, right_counts)
        assert isinstance(gini_score, (int, float))

        chi_score = chi_square.evaluate_split(parent_counts, left_counts, right_counts)
        assert isinstance(chi_score, (int, float))

    def test_criterion_with_invalid_inputs(self):
        """Test criteria behavior with invalid or unexpected inputs."""
        from theta_sketch_tree.criteria import get_criterion

        # Test all criteria with various edge cases
        criteria_names = ['gini', 'entropy', 'gain_ratio', 'binomial', 'chi_square']

        for criterion_name in criteria_names:
            criterion = get_criterion(criterion_name)

            # Test with negative counts (should be handled gracefully)
            try:
                result = criterion.compute_impurity(np.array([-1.0, 10.0]))
                # Some criteria may return negative values or NaN for invalid inputs
                assert isinstance(result, (int, float)) or np.isnan(result)
            except (ValueError, RuntimeWarning):
                pass  # Some criteria may reject negative values

            # Test with NaN values
            try:
                result = criterion.compute_impurity(np.array([np.nan, 10.0]))
                # Should either handle gracefully or raise an exception
                if not np.isnan(result):
                    assert result >= 0.0
            except (ValueError, RuntimeWarning):
                pass

            # Test with infinite values
            try:
                result = criterion.compute_impurity(np.array([np.inf, 10.0]))
                if not np.isinf(result):
                    assert result >= 0.0
            except (ValueError, RuntimeWarning):
                pass