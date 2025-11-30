"""
Centralized validation utilities for theta sketch tree.

Provides consistent input validation, error handling, and type checking
across all components.
"""

from typing import Dict, Any, Optional, Union, Tuple, List
import numpy as np
from numpy.typing import NDArray


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class ParameterValidator:
    """
    Centralized parameter validation for theta sketch tree components.
    """

    @staticmethod
    def validate_criterion(criterion: str) -> None:
        """Validate split criterion parameter."""
        valid_criteria = {'gini', 'entropy', 'gain_ratio', 'binomial', 'chi_square'}
        if criterion not in valid_criteria:
            raise ValidationError(
                f"Invalid criterion '{criterion}'. Must be one of: {valid_criteria}"
            )

    @staticmethod
    def validate_pruning_method(method: str) -> None:
        """Validate pruning method parameter."""
        valid_methods = {'none', 'cost_complexity', 'reduced_error', 'min_impurity', 'validation'}
        if method not in valid_methods:
            raise ValidationError(
                f"Invalid pruning method '{method}'. Must be one of: {valid_methods}"
            )

    @staticmethod
    def validate_positive_integer(value: int, name: str, minimum: int = 1) -> None:
        """Validate positive integer parameter."""
        if not isinstance(value, int) or value < minimum:
            raise ValidationError(f"{name} must be an integer >= {minimum}, got {value}")

    @staticmethod
    def validate_positive_float(value: float, name: str, minimum: float = 0.0) -> None:
        """Validate positive float parameter."""
        if not isinstance(value, (int, float)) or value < minimum:
            raise ValidationError(f"{name} must be a number >= {minimum}, got {value}")

    @staticmethod
    def validate_fraction(value: float, name: str) -> None:
        """Validate fraction parameter (0.0 to 1.0)."""
        if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
            raise ValidationError(f"{name} must be a fraction between 0.0 and 1.0, got {value}")

    @staticmethod
    def validate_verbose_level(verbose: int) -> None:
        """Validate verbosity level."""
        if not isinstance(verbose, int) or verbose < 0:
            raise ValidationError(f"verbose must be a non-negative integer, got {verbose}")


class SketchDataValidator:
    """
    Validation utilities for sketch data structures.
    """

    @staticmethod
    def validate_sketch_data(sketch_data: Dict[str, Dict[str, Any]]) -> None:
        """
        Validate sketch data structure.

        Parameters
        ----------
        sketch_data : dict
            Dictionary with 'positive' and 'negative' keys

        Raises
        ------
        ValidationError
            If structure is invalid
        """
        if not isinstance(sketch_data, dict):
            raise ValidationError("sketch_data must be a dictionary")

        if 'positive' not in sketch_data:
            raise ValidationError("sketch_data must contain 'positive' key")
        if 'negative' not in sketch_data:
            raise ValidationError("sketch_data must contain 'negative' key")

        for class_name in ['positive', 'negative']:
            class_data = sketch_data[class_name]
            if not isinstance(class_data, dict):
                raise ValidationError(f"sketch_data['{class_name}'] must be a dictionary")

            if 'total' not in class_data:
                raise ValidationError(f"sketch_data['{class_name}'] must contain 'total' key")

        # Check for features only after all required keys are validated
        all_feature_keys = []
        for class_name in ['positive', 'negative']:
            class_data = sketch_data[class_name]
            feature_keys = [k for k in class_data.keys() if k != 'total']
            all_feature_keys.extend(feature_keys)

        if not all_feature_keys:
            raise ValidationError("No features found in sketch_data")

    @staticmethod
    def validate_feature_mapping(feature_mapping: Dict[str, int]) -> None:
        """
        Validate feature mapping structure.

        Parameters
        ----------
        feature_mapping : dict
            Maps feature names to column indices

        Raises
        ------
        ValidationError
            If mapping is invalid
        """
        if not isinstance(feature_mapping, dict):
            raise ValidationError("feature_mapping must be a dictionary")

        if not feature_mapping:
            raise ValidationError("feature_mapping cannot be empty")

        for feature_name, column_idx in feature_mapping.items():
            if not isinstance(feature_name, str):
                raise ValidationError(f"Feature names must be strings, got {type(feature_name)}")
            if not isinstance(column_idx, int) or column_idx < 0:
                raise ValidationError(f"Column indices must be non-negative integers, got {column_idx}")


class ArrayValidator:
    """
    Validation utilities for numpy arrays and input data.
    """

    @staticmethod
    def validate_input_array(X: NDArray, expected_features: Optional[int] = None) -> NDArray:
        """
        Validate input array for prediction.

        Parameters
        ----------
        X : array-like
            Input data to validate
        expected_features : int, optional
            Expected number of features

        Returns
        -------
        X : ndarray
            Validated input array

        Raises
        ------
        ValidationError
            If array is invalid
        """
        try:
            X = np.asarray(X)
        except Exception as e:
            raise ValidationError(f"Cannot convert input to array: {e}")

        if X.ndim != 2:
            raise ValidationError(f"Input must be 2D array, got shape {X.shape}")

        # Allow empty arrays for predictions (should return empty results)
        if X.size == 0 and X.ndim == 2:
            return X

        if expected_features is not None and X.shape[1] != expected_features:
            raise ValidationError(
                f"Input has {X.shape[1]} features, but classifier expects {expected_features}"
            )

        return X

    @staticmethod
    def validate_binary_array(X: NDArray) -> NDArray:
        """
        Validate that array contains only binary values (0/1).

        Parameters
        ----------
        X : ndarray
            Input array to validate

        Returns
        -------
        X : ndarray
            Validated binary array

        Raises
        ------
        ValidationError
            If array contains non-binary values
        """
        unique_values = np.unique(X)
        valid_values = {0, 1, 0.0, 1.0}

        if not all(val in valid_values for val in unique_values):
            raise ValidationError(
                f"Input array must contain only binary values (0/1), "
                f"found values: {unique_values}"
            )

        return X


class TreeValidator:
    """
    Validation utilities for tree structures and components.
    """

    @staticmethod
    def validate_tree_hyperparameters(
        max_depth: Optional[int],
        min_samples_split: int,
        min_samples_leaf: int,
        min_impurity_decrease: float,
        validation_fraction: float
    ) -> None:
        """
        Validate tree hyperparameters.

        Parameters
        ----------
        max_depth : int or None
            Maximum tree depth
        min_samples_split : int
            Minimum samples to split
        min_samples_leaf : int
            Minimum samples in leaf
        min_impurity_decrease : float
            Minimum impurity decrease
        validation_fraction : float
            Validation fraction for pruning

        Raises
        ------
        ValidationError
            If any parameter is invalid
        """
        if max_depth is not None:
            ParameterValidator.validate_positive_integer(max_depth, "max_depth", minimum=1)

        ParameterValidator.validate_positive_integer(min_samples_split, "min_samples_split", minimum=2)
        ParameterValidator.validate_positive_integer(min_samples_leaf, "min_samples_leaf", minimum=1)

        # Note: sklearn allows min_samples_leaf >= min_samples_split
        # This is handled at tree building time, not during validation

        ParameterValidator.validate_positive_float(min_impurity_decrease, "min_impurity_decrease")
        ParameterValidator.validate_fraction(validation_fraction, "validation_fraction")

    @staticmethod
    def validate_fitted_state(is_fitted: bool, operation: str) -> None:
        """
        Validate that classifier is fitted before operation.

        Parameters
        ----------
        is_fitted : bool
            Whether classifier is fitted
        operation : str
            Name of operation being attempted

        Raises
        ------
        ValidationError
            If classifier is not fitted
        """
        if not is_fitted:
            raise ValidationError(f"Classifier must be fitted before {operation}. Call fit() first.")


def validate_and_convert_input(X, expected_features: int) -> NDArray:
    """
    Convenience function to validate and convert input for predictions.

    Parameters
    ----------
    X : array-like
        Input data
    expected_features : int
        Expected number of features

    Returns
    -------
    X : ndarray
        Validated input array

    Raises
    ------
    ValidationError
        If input is invalid
    """
    X = ArrayValidator.validate_input_array(X, expected_features)
    return ArrayValidator.validate_binary_array(X)