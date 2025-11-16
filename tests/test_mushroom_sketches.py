"""
Mushroom Dataset Theta Sketch Generator and Integration Test.

Creates realistic theta sketches from UCI Mushroom dataset for poisonous classification.
Uses lg_k=17 (sketch size 2^17 = 131,072) with proper sketch structure.
"""

import numpy as np
import pandas as pd
import pytest
import json
from typing import Dict, Any

# Global configuration parameters
DEFAULT_LG_K = 16           # Log-base-2 of nominal entries (at least 131,072 entries for high precision)
DEFAULT_MIN_SAMPLES_SPLIT = 6   # Minimum samples required to split a node
DEFAULT_MIN_SAMPLES_LEAF = 3    # Minimum samples required in a leaf node
DEFAULT_TREE_BUILDER = "intersection"  # Tree builder mode: "intersection" or "ratio_based"
                                      # - "intersection": Original algorithm with sketch intersections (may have sample conservation issues)
                                      # - "ratio_based": New algorithm using global ratios (guarantees sample conservation)

from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier


try:
    from datasketches import update_theta_sketch, theta_intersection, theta_union
    DATASKETCHES_AVAILABLE = True
except ImportError:
    DATASKETCHES_AVAILABLE = False
    print("Warning: datasketches library not available, using mock implementation")


class RealThetaSketch:
    """
    Proper Theta Sketch implementation using Apache DataSketches library.

    Uses high-precision theta sketches (lg_k=17) for better accuracy while
    maintaining probabilistic properties for large-scale data processing.
    """

    def __init__(self, lg_k: int = DEFAULT_LG_K):
        """
        Initialize with log-base-2 of nominal entries.

        Parameters
        ----------
        lg_k : int
            Log-base-2 of nominal entries (k = 2^lg_k)
            Default from DEFAULT_LG_K=17 gives 131,072 nominal entries for high precision
        """
        if not DATASKETCHES_AVAILABLE:
            raise ImportError("datasketches library required for RealThetaSketch")

        self.lg_k = lg_k
        self.sketch = update_theta_sketch(lg_k)

    def add(self, item: str) -> None:
        """Add an item to the theta sketch."""
        self.sketch.update(item)

    def get_estimate(self) -> float:
        """Get cardinality estimate from theta sketch."""
        return max(0.0, self.sketch.get_estimate())

    def intersection(self, other: 'RealThetaSketch') -> 'RealThetaSketch':
        """Proper theta sketch intersection using DataSketches library."""
        if DATASKETCHES_AVAILABLE:
            # Use real theta sketch intersection
            intersector = theta_intersection()
            intersector.update(self.sketch)
            intersector.update(other.sketch)
            intersection_sketch = intersector.get_result()

            result = RealThetaSketch(self.lg_k)
            result.sketch = intersection_sketch
            return result
        else:
            # Fallback for when DataSketches is not available
            result = RealThetaSketch(self.lg_k)
            # This is still an approximation, but better than the arbitrary 0.75
            intersection_est = min(self.get_estimate(), other.get_estimate()) * 0.8
            for i in range(int(intersection_est)):
                result.sketch.update(f"intersection_{hash((id(self), id(other)))}_{i}")
            return result


    def union(self, other: 'RealThetaSketch') -> 'RealThetaSketch':
        """Compute union using proper theta sketch operations."""
        # Use Apache DataSketches union operation
        union_sketch = theta_union(self.lg_k)
        union_sketch.update(self.sketch)
        union_sketch.update(other.sketch)
        union_result = union_sketch.get_result()

        # Create new sketch with union estimate
        result = RealThetaSketch(self.lg_k)
        union_est = union_result.get_estimate()

        # Add synthetic items to approximate the union
        for i in range(int(union_est)):
            result.sketch.update(f"union_{hash((id(self), id(other)))}_{i}")

        return result

    def get_theta(self) -> float:
        """Get the current theta value."""
        return self.sketch.get_theta()

    def is_estimation_mode(self) -> bool:
        """Check if sketch is in estimation mode."""
        return self.sketch.is_estimation_mode()


# Use real Theta Sketches if available, fallback to mock for testing
ThetaSketch = RealThetaSketch if DATASKETCHES_AVAILABLE else None

if not DATASKETCHES_AVAILABLE:
    # Fallback mock implementation with higher precision simulation
    class MockThetaSketch:
        def __init__(self, lg_k: int = DEFAULT_LG_K):
            self.lg_k = lg_k
            self.items = set()
            # Simulate theta sketch precision with hash sampling
            self.max_items = 2 ** lg_k  # Nominal capacity
            self.theta = 1.0

        def add(self, item: str):
            # Simulate theta sketch behavior with hash-based sampling
            item_hash = hash(item) % (2**32)  # 32-bit hash simulation
            if len(self.items) < self.max_items:
                self.items.add(item_hash)
            else:
                # Simulate theta adjustment when sketch is full
                self.theta = min(self.theta, item_hash / (2**32))
                if item_hash / (2**32) < self.theta:
                    self.items.add(item_hash)
                    # Remove items above new theta
                    self.items = {h for h in self.items if h / (2**32) < self.theta}

        def get_estimate(self) -> float:
            if self.theta == 1.0:
                return max(0.0, float(len(self.items)))
            else:
                return max(0.0, float(len(self.items)) / self.theta)

        def intersection(self, other):
            result = MockThetaSketch(self.lg_k)
            # Simulate intersection with common theta
            common_theta = max(self.theta, other.theta)
            intersection_items = self.items.intersection(other.items)
            result.items = intersection_items
            result.theta = common_theta
            return result

        def union(self, other):
            result = MockThetaSketch(self.lg_k)
            # Simulate union with common theta
            common_theta = max(self.theta, other.theta)
            union_items = self.items.union(other.items)
            result.items = union_items
            result.theta = common_theta
            return result

    ThetaSketch = MockThetaSketch


def load_mushroom_dataset() -> pd.DataFrame:
    """
    Load UCI Mushroom dataset from file.

    Always reads from a file - no synthetic data generation.
    """
    import os

    tests_dir = os.path.dirname(__file__)  # Directory containing this test file
    root_dir = os.path.dirname(tests_dir)  # Parent directory (project root)

    # Check for mushroom dataset files in order of preference
    filenames = ['mushrooms.csv', 'mushroom.csv', 'agaricus-lepiota.data', 'agaricus-lepiota.csv']
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


def create_mushroom_sketches(df: pd.DataFrame, lg_k: int = DEFAULT_LG_K) -> Dict[str, Dict[str, Any]]:
    """
    Create real theta sketches from mushroom dataset using Apache DataSketches.

    Parameters
    ----------
    df : DataFrame
        Mushroom dataset with 'class' column
    lg_k : int, default=DEFAULT_LG_K
        Log-base-2 of nominal entries (k = 2^lg_k). Default from DEFAULT_LG_K for 131,072 nominal entries.

    Returns
    -------
    sketch_data : dict
        Format: {'positive': {...}, 'negative': {...}}
        Each contains 'total' and feature sketches using real Theta Sketches
    """
    sketch_type = "Real Apache DataSketches" if DATASKETCHES_AVAILABLE else "Mock (for testing)"
    print(f"Creating theta sketches with lg_k={lg_k} (size={2**lg_k}) using {sketch_type}")

    # Separate by class
    poisonous_df = df[df['class'] == 'poisonous'].copy()
    edible_df = df[df['class'] == 'edible'].copy()

    print(f"Total dataset: {len(df)} samples")
    print(f"Poisonous samples: {len(poisonous_df)} ({len(poisonous_df)/len(df)*100:.1f}%)")
    print(f"Edible samples: {len(edible_df)} ({len(edible_df)/len(df)*100:.1f}%)")

    sketch_data = {
        'positive': {},  # poisonous
        'negative': {}   # edible
    }

    # Create total sketches for each class using real Theta Sketches
    pos_total = ThetaSketch(lg_k)
    neg_total = ThetaSketch(lg_k)

    print("Adding all samples to total sketches...")
    # Add ALL samples to total sketches (using row index as unique ID)
    for idx in poisonous_df.index:
        pos_total.add(f"sample_{idx}")

    for idx in edible_df.index:
        neg_total.add(f"sample_{idx}")

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
            pos_present = ThetaSketch(lg_k)  # Has this feature value
            pos_absent = ThetaSketch(lg_k)   # Doesn't have this feature value

            for idx, row in poisonous_df.iterrows():
                sample_id = f"sample_{idx}"
                if row[feature] == value:
                    pos_present.add(sample_id)
                else:
                    pos_absent.add(sample_id)

            # NEGATIVE class sketches
            neg_present = ThetaSketch(lg_k)
            neg_absent = ThetaSketch(lg_k)

            for idx, row in edible_df.iterrows():
                sample_id = f"sample_{idx}"
                if row[feature] == value:
                    neg_present.add(sample_id)
                else:
                    neg_absent.add(sample_id)

            # Store as tuples (present, absent)
            sketch_data['positive'][feature_name] = (pos_present, pos_absent)
            sketch_data['negative'][feature_name] = (neg_present, neg_absent)

    # Print summary
    n_features = len([k for k in sketch_data['positive'].keys() if k != 'total'])
    print(f"Created {n_features} binary features from {len(feature_columns)} original features")

    # Validate sketch estimates - THIS EXPLAINS THE SAMPLE COUNT DISCREPANCY
    pos_estimate = pos_total.get_estimate()
    neg_estimate = neg_total.get_estimate()
    print(f"Sketch estimates vs actual counts:")
    print(f"  Positive (poisonous): {pos_estimate:.0f} estimate, {len(poisonous_df)} actual")
    print(f"  Negative (edible): {neg_estimate:.0f} estimate, {len(edible_df)} actual")
    print(f"Note: Sketch estimates may differ from actual due to Theta Sketch approximation")

    return sketch_data


def create_mushroom_feature_mapping(sketch_data: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """
    Create feature mapping for mushroom features.

    Maps feature names to column indices for prediction matrix.
    """
    feature_names = [k for k in sketch_data['positive'].keys() if k != 'total']
    feature_mapping = {name: idx for idx, name in enumerate(feature_names)}

    print(f"Created feature mapping for {len(feature_mapping)} features")
    return feature_mapping


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


def print_tree_json(tree_root, max_depth: int = 5):
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
            criterion='gini',
            max_depth=10,  # Shallow tree for testing
            min_samples_split=DEFAULT_MIN_SAMPLES_SPLIT,
            tree_builder=DEFAULT_TREE_BUILDER,
            verbose=1
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
            criterion='entropy',
            max_depth=10,
            min_samples_split=DEFAULT_MIN_SAMPLES_SPLIT,
            tree_builder=DEFAULT_TREE_BUILDER
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
                max_depth=10,
                tree_builder=DEFAULT_TREE_BUILDER,
                verbose=0
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
            criterion='gini',
            max_depth=10,  # Allow deeper tree
            min_samples_split=DEFAULT_MIN_SAMPLES_SPLIT,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
            tree_builder=DEFAULT_TREE_BUILDER,
            verbose=1
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
        criterion='gini',
        max_depth=10,
        tree_builder=DEFAULT_TREE_BUILDER,
        verbose=1
    )
    clf.fit(sketch_data, feature_mapping)

    print(f"\n=== Results ===")
    print(f"Tree depth: {clf.tree_.depth if hasattr(clf.tree_, 'depth') else 'N/A'}")
    print(f"Number of features: {len(feature_mapping)}")

    # Show top features
    print("\nTop 10 important features:")
    for feature, importance in clf.get_top_features(10):
        print(f"  {feature}: {importance:.4f}")
