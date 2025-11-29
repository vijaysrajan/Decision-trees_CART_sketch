"""
Utility functions for ThetaSketchDecisionTreeClassifier.

This module provides convenience methods and helper functions
to reduce the main classifier class complexity.
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .classifier import ThetaSketchDecisionTreeClassifier


class ClassifierUtils:
    """
    Utility class containing convenience methods for the classifier.
    """

    @staticmethod
    def fit_from_csv(
        positive_csv: str,
        negative_csv: str,
        config_path: str,
        csv_path: Optional[str] = None,
        classifier_class=None
    ) -> "ThetaSketchDecisionTreeClassifier":
        """
        Convenience method: load sketches from CSV and fit model in one call.

        Parameters
        ----------
        positive_csv : str
            Path to CSV with positive class sketches
        negative_csv : str
            Path to CSV with negative class sketches
        config_path : str
            Path to configuration file
        csv_path : str, optional
            Path to single CSV (alternative to positive/negative CSVs)
        classifier_class : class, optional
            Classifier class to instantiate (for dependency injection)

        Returns
        -------
        classifier : ThetaSketchDecisionTreeClassifier
            Fitted classifier instance

        Examples
        --------
        >>> clf = ClassifierUtils.fit_from_csv(
        ...     "positive.csv", "negative.csv", "config.yaml"
        ... )
        """
        # Import here to avoid circular dependencies
        if classifier_class is None:
            from .classifier import ThetaSketchDecisionTreeClassifier
            classifier_class = ThetaSketchDecisionTreeClassifier

        from . import load_sketches, load_config

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
        clf = classifier_class(**config["hyperparameters"])

        # Fit model
        clf.fit(sketch_data, config["feature_mapping"])

        return clf

    @staticmethod
    def create_classifier_with_defaults(**kwargs) -> "ThetaSketchDecisionTreeClassifier":
        """
        Create classifier with sensible defaults.

        Parameters
        ----------
        **kwargs : dict
            Override default parameters

        Returns
        -------
        classifier : ThetaSketchDecisionTreeClassifier
            Classifier instance with defaults applied
        """
        # Import here to avoid circular dependencies
        from .classifier import ThetaSketchDecisionTreeClassifier

        # Default parameters
        defaults = {
            "criterion": "gini",
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "pruning": "none",
            "verbose": 0
        }

        # Override with user parameters
        defaults.update(kwargs)

        return ThetaSketchDecisionTreeClassifier(**defaults)


# Convenience functions for backward compatibility
def fit_from_csv(
    positive_csv: str,
    negative_csv: str,
    config_path: str,
    csv_path: Optional[str] = None,
) -> "ThetaSketchDecisionTreeClassifier":
    """
    Convenience function to load sketches from CSV and fit model in one call.

    Parameters
    ----------
    positive_csv : str
        Path to CSV with positive class sketches
    negative_csv : str
        Path to CSV with negative class sketches
    config_path : str
        Path to configuration file
    csv_path : str, optional
        Path to single CSV (alternative to positive/negative CSVs)

    Returns
    -------
    classifier : ThetaSketchDecisionTreeClassifier
        Fitted classifier instance
    """
    return ClassifierUtils.fit_from_csv(
        positive_csv, negative_csv, config_path, csv_path
    )


def create_classifier_with_defaults(**kwargs) -> "ThetaSketchDecisionTreeClassifier":
    """
    Create classifier with sensible defaults.

    Parameters
    ----------
    **kwargs : dict
        Override default parameters

    Returns
    -------
    classifier : ThetaSketchDecisionTreeClassifier
        Classifier instance with defaults applied
    """
    return ClassifierUtils.create_classifier_with_defaults(**kwargs)