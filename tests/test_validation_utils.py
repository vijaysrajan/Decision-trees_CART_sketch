"""
Test validation utilities.

This module tests the validation functions and error handling.
"""

import pytest
import numpy as np
from theta_sketch_tree.validation_utils import (
    ValidationError,
    ParameterValidator,
    SketchDataValidator,
    ArrayValidator,
    TreeValidator,
    validate_and_convert_input
)


class TestValidationError:
    """Test the ValidationError exception class."""

    def test_validation_error_creation(self):
        """Test ValidationError can be created and raised."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Test error message")
        assert "Test error message" in str(exc_info.value)


class TestParameterValidator:
    """Test parameter validation methods."""

    def test_validate_criterion_valid(self):
        """Test validation of valid criteria."""
        valid_criteria = ['gini', 'entropy', 'gain_ratio', 'binomial', 'chi_square']
        for criterion in valid_criteria:
            # Should not raise
            ParameterValidator.validate_criterion(criterion)

    def test_validate_criterion_invalid(self):
        """Test validation of invalid criteria."""
        with pytest.raises(ValidationError, match="Invalid criterion"):
            ParameterValidator.validate_criterion('invalid_criterion')

    def test_validate_pruning_method_valid(self):
        """Test validation of valid pruning methods."""
        valid_methods = ['none', 'cost_complexity', 'reduced_error', 'min_impurity', 'validation']
        for method in valid_methods:
            # Should not raise
            ParameterValidator.validate_pruning_method(method)

    def test_validate_pruning_method_invalid(self):
        """Test validation of invalid pruning methods."""
        with pytest.raises(ValidationError, match="Invalid pruning method"):
            ParameterValidator.validate_pruning_method('invalid_method')

    def test_validate_positive_integer_valid(self):
        """Test validation of valid positive integers."""
        ParameterValidator.validate_positive_integer(1, "test_param")
        ParameterValidator.validate_positive_integer(10, "test_param")
        ParameterValidator.validate_positive_integer(5, "test_param", minimum=2)

    def test_validate_positive_integer_invalid(self):
        """Test validation of invalid positive integers."""
        with pytest.raises(ValidationError):
            ParameterValidator.validate_positive_integer(0, "test_param")

        with pytest.raises(ValidationError):
            ParameterValidator.validate_positive_integer(-1, "test_param")

        with pytest.raises(ValidationError):
            ParameterValidator.validate_positive_integer(1.5, "test_param")

        with pytest.raises(ValidationError):
            ParameterValidator.validate_positive_integer(2, "test_param", minimum=5)

    def test_validate_positive_float_valid(self):
        """Test validation of valid positive floats."""
        ParameterValidator.validate_positive_float(0.0, "test_param")
        ParameterValidator.validate_positive_float(1.5, "test_param")
        ParameterValidator.validate_positive_float(10, "test_param")  # int should work
        ParameterValidator.validate_positive_float(5.0, "test_param", minimum=2.0)

    def test_validate_positive_float_invalid(self):
        """Test validation of invalid positive floats."""
        with pytest.raises(ValidationError):
            ParameterValidator.validate_positive_float(-1.0, "test_param")

        with pytest.raises(ValidationError):
            ParameterValidator.validate_positive_float("1.0", "test_param")

        with pytest.raises(ValidationError):
            ParameterValidator.validate_positive_float(1.0, "test_param", minimum=2.0)

    def test_validate_fraction_valid(self):
        """Test validation of valid fractions."""
        ParameterValidator.validate_fraction(0.0, "test_param")
        ParameterValidator.validate_fraction(0.5, "test_param")
        ParameterValidator.validate_fraction(1.0, "test_param")
        ParameterValidator.validate_fraction(1, "test_param")  # int should work

    def test_validate_fraction_invalid(self):
        """Test validation of invalid fractions."""
        with pytest.raises(ValidationError):
            ParameterValidator.validate_fraction(-0.1, "test_param")

        with pytest.raises(ValidationError):
            ParameterValidator.validate_fraction(1.1, "test_param")

        with pytest.raises(ValidationError):
            ParameterValidator.validate_fraction("0.5", "test_param")

    def test_validate_verbose_level_valid(self):
        """Test validation of valid verbose levels."""
        ParameterValidator.validate_verbose_level(0)
        ParameterValidator.validate_verbose_level(1)
        ParameterValidator.validate_verbose_level(3)

    def test_validate_verbose_level_invalid(self):
        """Test validation of invalid verbose levels."""
        with pytest.raises(ValidationError):
            ParameterValidator.validate_verbose_level(-1)

        with pytest.raises(ValidationError):
            ParameterValidator.validate_verbose_level(1.5)

        with pytest.raises(ValidationError):
            ParameterValidator.validate_verbose_level("1")


class TestSketchDataValidator:
    """Test sketch data validation methods."""

    def test_validate_sketch_data_structure_valid(self):
        """Test validation of valid sketch data structure."""
        valid_data = {
            'positive': {'total': 'mock_sketch'},
            'negative': {'total': 'mock_sketch'}
        }
        # Should not raise
        SketchDataValidator.validate_sketch_data(valid_data)

    def test_validate_sketch_data_structure_missing_positive(self):
        """Test validation fails when positive key is missing."""
        invalid_data = {'negative': {'total': 'mock_sketch'}}
        with pytest.raises(ValidationError, match="sketch_data must contain 'positive' key"):
            SketchDataValidator.validate_sketch_data_structure(invalid_data)

    def test_validate_sketch_data_structure_missing_negative(self):
        """Test validation fails when negative key is missing."""
        invalid_data = {'positive': {'total': 'mock_sketch'}}
        with pytest.raises(ValidationError, match="sketch_data must contain 'negative' key"):
            SketchDataValidator.validate_sketch_data_structure(invalid_data)

    def test_validate_sketch_data_structure_missing_total(self):
        """Test validation fails when total key is missing."""
        invalid_data = {
            'positive': {},
            'negative': {'total': 'mock_sketch'}
        }
        with pytest.raises(ValidationError, match="must contain 'total' key"):
            SketchDataValidator.validate_sketch_data_structure(invalid_data)

    def test_validate_feature_mapping_valid(self):
        """Test validation of valid feature mapping."""
        valid_mapping = {'feature_1': 0, 'feature_2': 1, 'feature_3': 2}
        # Should not raise
        SketchDataValidator.validate_feature_mapping(valid_mapping)

    def test_validate_feature_mapping_invalid(self):
        """Test validation of invalid feature mapping."""
        with pytest.raises(ValidationError, match="feature_mapping must be a dictionary"):
            SketchDataValidator.validate_feature_mapping([])

        with pytest.raises(ValidationError, match="feature_mapping cannot be empty"):
            SketchDataValidator.validate_feature_mapping({})

        # Test non-string keys
        with pytest.raises(ValidationError, match="All feature names must be strings"):
            SketchDataValidator.validate_feature_mapping({1: 0, 'feature_2': 1})

        # Test non-integer values
        with pytest.raises(ValidationError, match="All feature indices must be integers"):
            SketchDataValidator.validate_feature_mapping({'feature_1': '0', 'feature_2': 1})


class TestArrayValidator:
    """Test array validation methods."""

    def test_validate_array_2d_valid(self):
        """Test validation of valid 2D arrays."""
        valid_arrays = [
            np.array([[1, 2], [3, 4]]),
            np.array([[1.0, 2.0]]),
            [[1, 2], [3, 4]]  # list should work
        ]
        for arr in valid_arrays:
            result = ArrayValidator.validate_array_2d(arr, "test_array")
            assert result.ndim == 2

    def test_validate_array_2d_invalid(self):
        """Test validation of invalid arrays."""
        with pytest.raises(ValidationError, match="Input must be 2D array"):
            ArrayValidator.validate_array_2d(np.array([1, 2, 3]), "test_array")

        with pytest.raises(ValidationError, match="Input must be 2D array"):
            ArrayValidator.validate_array_2d(np.array([[[1, 2]]]), "test_array")

    def test_validate_feature_count_valid(self):
        """Test validation of valid feature counts."""
        # Should not raise
        ArrayValidator.validate_feature_count(5, 5, "test_array")

    def test_validate_feature_count_invalid(self):
        """Test validation of invalid feature counts."""
        with pytest.raises(ValidationError, match="Input has 3 features, but classifier expects 5"):
            ArrayValidator.validate_feature_count(3, 5, "test_array")


class TestTreeValidator:
    """Test tree validation methods."""

    def test_validate_tree_structure_none(self):
        """Test validation fails for None tree."""
        with pytest.raises(ValidationError, match="Tree structure cannot be None"):
            TreeValidator.validate_tree_structure(None)

    def test_validate_fitted_state_not_fitted(self):
        """Test validation fails for unfitted classifier."""
        class MockClassifier:
            _is_fitted = False

        classifier = MockClassifier()
        with pytest.raises(ValidationError, match="Classifier must be fitted"):
            TreeValidator.validate_fitted_state(classifier)

    def test_validate_fitted_state_fitted(self):
        """Test validation passes for fitted classifier."""
        class MockClassifier:
            _is_fitted = True

        classifier = MockClassifier()
        # Should not raise
        TreeValidator.validate_fitted_state(classifier)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    def test_parameter_validator_edge_cases(self):
        """Test parameter validator with edge cases."""
        # Test None values
        with pytest.raises(ValidationError):
            ParameterValidator.validate_positive_integer(None, "test")

        with pytest.raises(ValidationError):
            ParameterValidator.validate_positive_float(None, "test")

        # Test boundary values
        ParameterValidator.validate_fraction(0.0, "test")
        ParameterValidator.validate_fraction(1.0, "test")

        # Test minimum values
        ParameterValidator.validate_positive_integer(0, "test", minimum=0)
        ParameterValidator.validate_positive_float(0.0, "test", minimum=0.0)

    def test_array_validator_edge_cases(self):
        """Test array validator with edge cases."""
        # Test empty array
        empty_array = np.array([]).reshape(0, 2)
        result = ArrayValidator.validate_array_2d(empty_array, "test")
        assert result.shape == (0, 2)

        # Test single row
        single_row = np.array([[1, 2]])
        result = ArrayValidator.validate_array_2d(single_row, "test")
        assert result.shape == (1, 2)

    def test_sketch_validator_edge_cases(self):
        """Test sketch validator with edge cases."""
        # Test with additional keys (should be allowed)
        extra_keys_data = {
            'positive': {'total': 'mock', 'extra_key': 'value'},
            'negative': {'total': 'mock', 'extra_key': 'value'},
            'metadata': 'allowed'
        }
        # Should not raise
        SketchDataValidator.validate_sketch_data_structure(extra_keys_data)

        # Test feature mapping with non-sequential indices (should be allowed)
        non_sequential = {'feat_a': 5, 'feat_b': 1, 'feat_c': 10}
        # Should not raise
        SketchDataValidator.validate_feature_mapping(non_sequential)