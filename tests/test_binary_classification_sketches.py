"""
Binary Classification Dataset Theta Sketch Generator and Integration Test.

Creates theta sketches from binary classification datasets for decision tree training.
Designed to work with any CSV dataset with a 'class' column containing two values.
"""

import numpy as np
import pandas as pd
import pytest
import json
from typing import Dict, Any, Tuple

# Global configuration parameters
DEFAULT_LG_K = 16           # Log-base-2 of nominal entries (65,536 entries)
DEFAULT_MIN_SAMPLES_SPLIT = 2   # Minimum samples required to split a node
DEFAULT_MIN_SAMPLES_LEAF = 1   # Minimum samples required in a leaf node
DEFAULT_MAX_DEPTH = 5     # Maximum tree depth (None for unlimited depth)
DEFAULT_CRITERION = "gini"  # Split criterion: 'gini', 'entropy', 'gain_ratio', 'binomial', 'binomial_chi'
DEFAULT_TREE_BUILDER = "intersection"  # Tree builder mode: "intersection" or "ratio_based"
                                      # - "intersection": Original algorithm with sketch intersections (may have sample conservation issues)
                                      # - "ratio_based": New algorithm using global ratios (guarantees sample conservation)
DEFAULT_VERBOSE = 0         # Verbosity level: 0=silent, 1=basic info, 2=detailed debug

from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier

try:
    from datasketches import update_theta_sketch, theta_intersection, theta_union
    DATASKETCHES_AVAILABLE = True
except ImportError:
    DATASKETCHES_AVAILABLE = False
    raise ImportError("Apache DataSketches library required. Install with: pip install datasketches")


class ThetaSketchWrapper:
    """
    Wrapper for Apache DataSketches that adds intersection method for tree builder compatibility.
    """

    def __init__(self, sketch):
        self._sketch = sketch

    def get_estimate(self):
        """Return estimated cardinality."""
        return self._sketch.get_estimate()

    def update(self, item):
        """Add item to sketch."""
        return self._sketch.update(item)

    def intersection(self, other):
        """Compute intersection with another sketch using Apache DataSketches API."""
        if isinstance(other, ThetaSketchWrapper):
            other_sketch = other._sketch
        else:
            other_sketch = other

        intersector = theta_intersection()
        intersector.update(self._sketch)
        intersector.update(other_sketch)
        result = intersector.get_result()

        return ThetaSketchWrapper(result)

    def __getattr__(self, name):
        """Delegate other methods to the underlying sketch."""
        return getattr(self._sketch, name)


def load_dataset() -> pd.DataFrame:
    """
    Load binary classification dataset from file.

    Searches for common dataset file formats and automatically handles
    different column naming conventions. Works with any CSV that has
    a binary target variable.
    """
    import os

    tests_dir = os.path.dirname(__file__)  # Directory containing this test file
    root_dir = os.path.dirname(tests_dir)  # Parent directory (project root)

    # Check for mushroom dataset files in order of preference
    filenames = ['agaricus-lepiota.csv', 'mushrooms.csv', 'mushroom.csv', 'agaricus-lepiota.data']
    search_paths = [
        os.path.join(tests_dir, 'resources'),  # tests/resources/
        tests_dir,                             # tests/
        root_dir,                              # project root/
        '.'                                    # current directory
    ]

    for search_path in search_paths:
        for filename in filenames:
            filepath = os.path.join(search_path, filename)
            if os.path.exists(filepath):
                print(f"Loading mushroom dataset from: {filepath}")

                # Try reading with different formats
                try:
                    df = pd.read_csv(filepath)
                    print(f"Dataset shape: {df.shape}")
                    print(f"Columns: {list(df.columns)}")

                    # Handle different column naming conventions
                    if 'class' not in df.columns and 'target' not in df.columns:
                        # Assume first column is class (standard UCI format)
                        columns = ['class'] + [f'feature_{i}' for i in range(1, len(df.columns))]
                        df.columns = columns
                        print("Assumed first column is class (no header detected)")
                    elif 'target' in df.columns:
                        # Rename target to class for consistency
                        df = df.rename(columns={'target': 'class'})
                        print("Renamed 'target' column to 'class'")

                    # Handle UCI encoding (p=poisonous, e=edible)
                    class_values = set(df['class'].unique())
                    print(f"Class values found: {sorted(class_values)}")

                    if 'p' in class_values and 'e' in class_values:
                        df['class'] = df['class'].map({'p': 'poisonous', 'e': 'edible'})
                        print("Mapped p->poisonous, e->edible")
                    elif len(class_values) == 2:
                        # Map any two values to poisonous/edible
                        sorted_values = sorted(class_values)
                        mapping = {sorted_values[0]: 'edible', sorted_values[1]: 'poisonous'}
                        df['class'] = df['class'].map(mapping)
                        print(f"Mapped {mapping}")
                    else:
                        raise ValueError(f"Expected 2 class values, found {len(class_values)}: {class_values}")

                    print(f"Final class distribution: {df['class'].value_counts().to_dict()}")
                    return df

                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    continue

    # If we get here, no file was found
    raise FileNotFoundError(
        f"No mushroom dataset found in any of these locations:\n" +
        "\n".join([f"  {os.path.join(path, fname)}" for path in search_paths for fname in filenames])
    )


def create_binary_classification_sketches(df: pd.DataFrame, lg_k: int = DEFAULT_LG_K) -> Dict[str, Dict[str, Any]]:
    """
    Create theta sketches from any binary classification dataset using Apache DataSketches.

    Parameters
    ----------
    df : DataFrame
        Dataset with 'class' column containing exactly two unique values
    lg_k : int, default=DEFAULT_LG_K
        Log-base-2 of nominal entries (k = 2^lg_k)

    Returns
    -------
    sketch_data : dict
        Format: {'positive': {...}, 'negative': {...}}
        Each contains 'total' and feature sketches using Apache DataSketches
    """
    print(f"Creating theta sketches with lg_k={lg_k} (size={2**lg_k}) using Apache DataSketches")

    # Identify the two class values
    class_values = sorted(df['class'].unique())
    if len(class_values) != 2:
        raise ValueError(f"Expected exactly 2 class values, found {len(class_values)}: {class_values}")

    # Assign first (alphabetically) to negative, second to positive
    negative_class, positive_class = class_values
    positive_df = df[df['class'] == positive_class].copy()
    negative_df = df[df['class'] == negative_class].copy()

    print(f"Total dataset: {len(df)} samples")
    print(f"Positive class '{positive_class}': {len(positive_df)} samples ({len(positive_df)/len(df)*100:.1f}%)")
    print(f"Negative class '{negative_class}': {len(negative_df)} samples ({len(negative_df)/len(df)*100:.1f}%)")

    sketch_data = {
        'positive': {},  # positive_class
        'negative': {}   # negative_class
    }

    # Create total sketches for each class using Apache DataSketches with wrapper
    pos_total = ThetaSketchWrapper(update_theta_sketch(lg_k))
    neg_total = ThetaSketchWrapper(update_theta_sketch(lg_k))

    print("Adding all samples to total sketches...")
    # Add ALL samples to total sketches (using row index as unique ID)
    for idx in positive_df.index:
        pos_total.update(f"sample_{idx}")

    for idx in negative_df.index:
        neg_total.update(f"sample_{idx}")

    sketch_data['positive']['total'] = pos_total
    sketch_data['negative']['total'] = neg_total

    # Create feature sketches
    feature_columns = [col for col in df.columns if col != 'class']

    print(f"Processing {len(feature_columns)} features...")
    for feature in feature_columns:
        print(f"  Feature: {feature}")

        # Get unique values for this feature
        unique_values = df[feature].unique()

        # For each unique value, create binary features (value_present/absent)
        for value in unique_values:
            feature_name = f"{feature}={value}"

            # POSITIVE class sketches
            pos_present = ThetaSketchWrapper(update_theta_sketch(lg_k))  # Has this feature value
            pos_absent = ThetaSketchWrapper(update_theta_sketch(lg_k))   # Doesn't have this feature value

            for idx, row in positive_df.iterrows():
                sample_id = f"sample_{idx}"
                if row[feature] == value:
                    pos_present.update(sample_id)
                else:
                    pos_absent.update(sample_id)

            # NEGATIVE class sketches
            neg_present = ThetaSketchWrapper(update_theta_sketch(lg_k))
            neg_absent = ThetaSketchWrapper(update_theta_sketch(lg_k))

            for idx, row in negative_df.iterrows():
                sample_id = f"sample_{idx}"
                if row[feature] == value:
                    neg_present.update(sample_id)
                else:
                    neg_absent.update(sample_id)

            # Store as tuples (present, absent)
            sketch_data['positive'][feature_name] = (pos_present, pos_absent)
            sketch_data['negative'][feature_name] = (neg_present, neg_absent)

    # Print summary
    n_features = len([k for k in sketch_data['positive'].keys() if k != 'total'])
    print(f"Created {n_features} binary features from {len(feature_columns)} original features")

    # Validate sketch estimates
    pos_estimate = pos_total.get_estimate()
    neg_estimate = neg_total.get_estimate()
    print(f"Sketch estimates vs actual counts:")
    print(f"  Positive ('{positive_class}'): {pos_estimate:.0f} estimate, {len(positive_df)} actual")
    print(f"  Negative ('{negative_class}'): {neg_estimate:.0f} estimate, {len(negative_df)} actual")
    print(f"Note: Sketch estimates may differ from actual due to Theta Sketch approximation")

    return sketch_data


# Backward compatibility aliases for existing code
def load_mushroom_dataset() -> pd.DataFrame:
    """Backward compatibility wrapper for load_dataset."""
    return load_dataset()


def create_mushroom_sketches(df: pd.DataFrame, lg_k: int = DEFAULT_LG_K) -> Dict[str, Dict[str, Any]]:
    """Backward compatibility wrapper for create_binary_classification_sketches."""
    return create_binary_classification_sketches(df, lg_k)


def create_binary_classification_feature_mapping(sketch_data: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """
    Create feature mapping for binary classification features.

    Maps feature names to column indices for prediction matrix.
    """
    feature_names = [k for k in sketch_data['positive'].keys() if k != 'total']
    feature_mapping = {name: idx for idx, name in enumerate(feature_names)}

    print(f"Created feature mapping for {len(feature_mapping)} features")
    return feature_mapping


def create_mushroom_feature_mapping(sketch_data: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """Backward compatibility wrapper for create_binary_classification_feature_mapping."""
    return create_binary_classification_feature_mapping(sketch_data)


def tree_to_json(node, max_depth: int = 10, current_depth: int = 0) -> Dict:
    """
    Convert decision tree to JSON format for easy visualization.

    Parameters
    ----------
    node : TreeNode
        Root node of the tree
    max_depth : int
        Maximum depth to traverse (to limit output size)
    current_depth : int
        Current depth in recursion

    Returns
    -------
    dict
        JSON representation of the tree
    """
    if current_depth >= max_depth:
        return {
            "type": "truncated",
            "depth": current_depth,
            "message": f"Truncated at depth {max_depth}"
        }

    tree_dict = {
        "depth": current_depth,
        "n_samples": node.n_samples,
        "class_counts": node.class_counts.tolist() if hasattr(node, 'class_counts') else None,
        "impurity": round(node.impurity, 4) if hasattr(node, 'impurity') else None,
        "is_leaf": node.is_leaf
    }

    if node.is_leaf:
        tree_dict.update({
            "type": "leaf",
            "prediction": node.prediction,
            "probabilities": [round(p, 4) for p in node.probabilities] if hasattr(node, 'probabilities') else None
        })
    else:
        tree_dict.update({
            "type": "split",
            "feature_name": node.feature_name,
            "feature_idx": node.feature_idx,
            "split_condition": f"{node.feature_name} == 1",
            "left_condition": f"{node.feature_name} == 1 (TRUE)",
            "right_condition": f"{node.feature_name} == 0 (FALSE/NOT)",
            "left": tree_to_json(node.left, max_depth, current_depth + 1),
            "right": tree_to_json(node.right, max_depth, current_depth + 1)
        })

    return tree_dict


def print_tree_json(tree_root, max_depth: int = 7):
    """Print tree in JSON format with nice formatting."""
    tree_json = tree_to_json(tree_root, max_depth)
    print("\n=== Decision Tree Structure (JSON) ===")
    print(json.dumps(tree_json, indent=2))
    return tree_json


# Integration Tests
class TestMushroomIntegration:
    """Integration tests using realistic mushroom theta sketches."""

    @pytest.fixture(scope='class')
    def mushroom_data(self):
        """Load mushroom dataset (cached for entire test class)."""
        return load_mushroom_dataset()

    @pytest.fixture(scope='class')
    def mushroom_sketches(self, mushroom_data):
        """Create mushroom sketches using real Apache DataSketches (cached for entire test class)."""
        # Use ALL data - no sampling
        return create_mushroom_sketches(mushroom_data, lg_k=DEFAULT_LG_K)  # Use global DEFAULT_LG_K for 131,072 nominal values

    @pytest.fixture
    def mushroom_feature_mapping(self, mushroom_sketches):
        """Create feature mapping for mushroom features."""
        return create_mushroom_feature_mapping(mushroom_sketches)

    def test_mushroom_sketch_structure(self, mushroom_sketches):
        """Test that mushroom sketches have correct structure."""
        # Check top-level keys
        assert 'positive' in mushroom_sketches
        assert 'negative' in mushroom_sketches

        # Check total sketches exist
        assert 'total' in mushroom_sketches['positive']
        assert 'total' in mushroom_sketches['negative']

        # Check that we have features
        pos_features = [k for k in mushroom_sketches['positive'].keys() if k != 'total']
        neg_features = [k for k in mushroom_sketches['negative'].keys() if k != 'total']

        assert len(pos_features) > 0
        assert len(pos_features) == len(neg_features)
        assert set(pos_features) == set(neg_features)

        # Check feature sketches are tuples (present, absent)
        for feature in pos_features[:3]:  # Check first 3 features
            assert isinstance(mushroom_sketches['positive'][feature], tuple)
            assert len(mushroom_sketches['positive'][feature]) == 2
            assert isinstance(mushroom_sketches['negative'][feature], tuple)
            assert len(mushroom_sketches['negative'][feature]) == 2

    def test_mushroom_sketch_estimates(self, mushroom_sketches):
        """Test that sketch estimates are reasonable."""
        pos_total = mushroom_sketches['positive']['total'].get_estimate()
        neg_total = mushroom_sketches['negative']['total'].get_estimate()

        # Should have positive estimates
        assert pos_total > 0
        assert neg_total > 0

        # Should be reasonable for 1000 sample dataset (lenient bounds for HLL estimation)
        assert 1 <= pos_total <= 8000  # Allow for full dataset size
        assert 1 <= neg_total <= 8000

        print(f"Positive estimate: {pos_total:.0f}, Negative estimate: {neg_total:.0f}")

    def test_mushroom_decision_tree_fitting(self, mushroom_sketches, mushroom_feature_mapping):
        """Test that decision tree can fit on mushroom sketches."""
        # Create classifier
        clf = ThetaSketchDecisionTreeClassifier(
            criterion=DEFAULT_CRITERION,
            max_depth=DEFAULT_MAX_DEPTH,
            min_samples_split=DEFAULT_MIN_SAMPLES_SPLIT,
            tree_builder=DEFAULT_TREE_BUILDER,
            verbose=1  # Override default for testing output
        )

        # Should fit without errors
        clf.fit(mushroom_sketches, mushroom_feature_mapping)

        # Check fitted attributes
        assert clf._is_fitted
        assert clf.tree_ is not None
        assert len(clf.feature_names_in_) == len(mushroom_feature_mapping)

        # Check feature importances
        importances = clf.feature_importances_
        assert len(importances) == len(mushroom_feature_mapping)
        assert np.all(importances >= 0)
        assert np.isclose(importances.sum(), 1.0)

        print(f"Tree built with {len(mushroom_feature_mapping)} features")
        print(f"Top 3 important features:")
        top_features = clf.get_top_features(3)
        for feature, importance in top_features:
            print(f"  {feature}: {importance:.3f}")

    def test_mushroom_prediction(self, mushroom_sketches, mushroom_feature_mapping):
        """Test prediction on mushroom data."""
        # Fit classifier
        clf = ThetaSketchDecisionTreeClassifier(
            criterion='entropy',  # Test with entropy instead of default
            max_depth=DEFAULT_MAX_DEPTH,
            min_samples_split=DEFAULT_MIN_SAMPLES_SPLIT,
            tree_builder=DEFAULT_TREE_BUILDER,
            verbose=DEFAULT_VERBOSE
        )
        clf.fit(mushroom_sketches, mushroom_feature_mapping)

        # Create test data (binary feature matrix)
        n_features = len(mushroom_feature_mapping)
        n_samples = 50

        # Generate random binary features
        np.random.seed(123)
        X_test = np.random.randint(0, 2, size=(n_samples, n_features))

        # Test prediction methods
        predictions = clf.predict(X_test)
        probabilities = clf.predict_proba(X_test)

        # Validate outputs
        assert len(predictions) == n_samples
        assert all(pred in [0, 1] for pred in predictions)

        assert probabilities.shape == (n_samples, 2)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

        # Consistency check
        predicted_from_proba = np.argmax(probabilities, axis=1)
        assert np.array_equal(predictions, predicted_from_proba)

        print(f"Predictions: {np.bincount(predictions)}")
        print(f"Mean probabilities: [{probabilities[:, 0].mean():.3f}, {probabilities[:, 1].mean():.3f}]")

    def test_mushroom_different_criteria(self, mushroom_sketches, mushroom_feature_mapping):
        """Test that all criteria work with mushroom data."""
        criteria = ['gini', 'entropy', 'gain_ratio']

        X_test = np.random.randint(0, 2, size=(20, len(mushroom_feature_mapping)))

        for criterion in criteria:
            clf = ThetaSketchDecisionTreeClassifier(
                criterion=criterion,
                max_depth=DEFAULT_MAX_DEPTH,
                tree_builder=DEFAULT_TREE_BUILDER,
                verbose=DEFAULT_VERBOSE
            )

            # Should fit and predict without errors
            clf.fit(mushroom_sketches, mushroom_feature_mapping)
            predictions = clf.predict(X_test)

            assert len(predictions) == 20
            assert all(pred in [0, 1] for pred in predictions)

            print(f"Criterion {criterion}: {np.bincount(predictions)}")

    def test_mushroom_tree_structure(self, mushroom_sketches, mushroom_feature_mapping):
        """Test tree structure and print in JSON format with global min_samples parameters."""
        clf = ThetaSketchDecisionTreeClassifier(
            criterion=DEFAULT_CRITERION,
            max_depth=DEFAULT_MAX_DEPTH,
            min_samples_split=DEFAULT_MIN_SAMPLES_SPLIT,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
            tree_builder=DEFAULT_TREE_BUILDER,
            verbose=1  # Override default for testing output
        )

        clf.fit(mushroom_sketches, mushroom_feature_mapping)

        # Print tree structure in JSON format
        tree_json = print_tree_json(clf.tree_, max_depth=10)

        # Also print summary
        print(f"\n=== Tree Summary ===")
        print(f"Tree depth: {clf.tree_.depth if hasattr(clf.tree_, 'depth') else 'N/A'}")
        print(f"Number of features: {len(mushroom_feature_mapping)}")

        # Show top features
        print("\nTop 10 important features:")
        for feature, importance in clf.get_top_features(10):
            print(f"  {feature}: {importance:.4f}")

        # Basic validation
        assert clf._is_fitted
        assert tree_json['depth'] == 0  # Root should be at depth 0
        assert tree_json['n_samples'] > 0


if __name__ == "__main__":
    """Run mushroom sketch creation as standalone script."""
    print("=== Creating Mushroom Theta Sketches ===")

    # Load dataset
    df = load_mushroom_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"Classes: {df['class'].value_counts().to_dict()}")

    # Create sketches
    sketch_data = create_mushroom_sketches(df.sample(n=2000, random_state=42), lg_k=DEFAULT_LG_K)
    feature_mapping = create_mushroom_feature_mapping(sketch_data)

    # Test classifier
    clf = ThetaSketchDecisionTreeClassifier(
        criterion=DEFAULT_CRITERION,
        max_depth=DEFAULT_MAX_DEPTH,
        tree_builder=DEFAULT_TREE_BUILDER,
        verbose=1  # Override default for demo output
    )
    clf.fit(sketch_data, feature_mapping)

    print(f"\n=== Results ===")
    print(f"Tree depth: {clf.tree_.depth if hasattr(clf.tree_, 'depth') else 'N/A'}")
    print(f"Number of features: {len(feature_mapping)}")

    # Show top features
    print("\nTop 10 important features:")
    for feature, importance in clf.get_top_features(10):
        print(f"  {feature}: {importance:.4f}")
