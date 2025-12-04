"""
Enhanced mushroom dataset validation tests.

These tests address the inadequacies in the current regression tests by adding:
1. Recursive tree structure comparison
2. Mathematical correctness validation
3. Prediction accuracy validation
4. Sketch operation validation
5. Cross-validation against ground truth
"""

import pytest
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path

from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier
from theta_sketch_tree.criteria import GiniCriterion, EntropyCriterion, ChiSquareCriterion
from tests.test_binary_classification_sketches import (
    load_mushroom_dataset,
    create_mushroom_sketches,
    create_mushroom_feature_mapping
)


class TestEnhancedMushroomValidation:
    """Enhanced validation tests for mushroom dataset."""

    @pytest.fixture
    def mushroom_data(self):
        """Load mushroom dataset and sketches."""
        df = load_mushroom_dataset()
        # Use pre-computed sketches for consistency with other tests
        from tools.sketch_generation.create_mushroom_sketch_files import load_sketches_from_csv
        import json
        positive_file = "tests/fixtures/mushroom_positive_sketches_lg_k_11.csv"
        negative_file = "tests/fixtures/mushroom_negative_sketches_lg_k_11.csv"
        sketches = load_sketches_from_csv(positive_file, negative_file, lg_k=11)
        with open("tests/fixtures/mushroom_feature_mapping.json", 'r') as f:
            feature_mapping = json.load(f)
        return df, sketches, feature_mapping

    @pytest.fixture
    def baseline_outputs(self):
        """Load baseline outputs for comparison."""
        try:
            with open("tests/integration/mushroom/baselines/mushroom_baseline_outputs_lg_k_11.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            pytest.skip("Baseline outputs not found. Run generate_mushroom_baselines.py first.")

    def compare_tree_structures_recursively(self, current_node: Dict, baseline_node: Dict, path: str = "root") -> None:
        """
        Recursively compare two tree structures for exact match.

        This addresses the critical gap in test_core_criteria_regression which only
        compares top-level metadata but ignores the actual tree structure.
        """
        # Compare node metadata
        assert current_node["is_leaf"] == baseline_node["is_leaf"], f"Leaf status mismatch at {path}"
        assert current_node["depth"] == baseline_node["depth"], f"Depth mismatch at {path}"

        # Compare sample counts (allow small float differences)
        current_samples = current_node["n_samples"]
        baseline_samples = baseline_node["n_samples"]
        assert abs(current_samples - baseline_samples) < 1.0, f"Sample count mismatch at {path}: {current_samples} vs {baseline_samples}"

        # Compare class counts
        current_counts = current_node["class_counts"]
        baseline_counts = baseline_node["class_counts"]
        assert len(current_counts) == len(baseline_counts), f"Class count length mismatch at {path}"
        for i, (curr, base) in enumerate(zip(current_counts, baseline_counts)):
            assert abs(curr - base) < 1.0, f"Class count {i} mismatch at {path}: {curr} vs {base}"

        # Compare impurity (allow small numerical differences)
        current_impurity = current_node["impurity"]
        baseline_impurity = baseline_node["impurity"]
        assert abs(current_impurity - baseline_impurity) < 1e-6, f"Impurity mismatch at {path}: {current_impurity} vs {baseline_impurity}"

        # If it's a split node, compare split details and recurse
        if not current_node["is_leaf"]:
            assert current_node["feature_idx"] == baseline_node["feature_idx"], f"Feature index mismatch at {path}"
            assert current_node["feature_name"] == baseline_node["feature_name"], f"Feature name mismatch at {path}"

            # Recursively compare children
            assert "left" in current_node and "left" in baseline_node, f"Missing left child at {path}"
            assert "right" in current_node and "right" in baseline_node, f"Missing right child at {path}"

            self.compare_tree_structures_recursively(current_node["left"], baseline_node["left"], f"{path}/left")
            self.compare_tree_structures_recursively(current_node["right"], baseline_node["right"], f"{path}/right")

    def test_recursive_tree_structure_comparison(self, mushroom_data, baseline_outputs):
        """Test complete recursive tree structure matching."""
        df, sketches, feature_mapping = mushroom_data

        # Test with default_gini configuration
        if "default_gini" not in baseline_outputs:
            pytest.skip("No default_gini baseline")

        baseline = baseline_outputs["default_gini"]
        if "error" in baseline:
            pytest.skip(f"Baseline had error: {baseline['error']}")

        # Train classifier
        clf = ThetaSketchDecisionTreeClassifier(criterion="gini", max_depth=5, verbose=0)
        clf.fit(sketches, feature_mapping)

        # Serialize current tree
        from tests.test_binary_classification_sketches import tree_to_json
        current_tree = tree_to_json(clf.tree_, max_depth=10)

        # Perform recursive comparison
        baseline_tree = baseline["tree_structure"]
        self.compare_tree_structures_recursively(current_tree, baseline_tree)

    def test_split_criteria_mathematical_correctness(self, mushroom_data):
        """Validate split criteria calculations against manual computation."""
        df, sketches, feature_mapping = mushroom_data

        # Test Gini criterion with known values
        gini_criterion = GiniCriterion()

        # Known test case
        left_counts = np.array([100.0, 50.0])    # [negative, positive]
        right_counts = np.array([50.0, 100.0])
        parent_counts = left_counts + right_counts

        # Manual Gini calculation
        def manual_gini(counts):
            if counts.sum() == 0:
                return 0.0
            proportions = counts / counts.sum()
            return 1.0 - np.sum(proportions ** 2)

        left_gini = manual_gini(left_counts)
        right_gini = manual_gini(right_counts)
        parent_gini = manual_gini(parent_counts)

        # Weighted average
        total_samples = parent_counts.sum()
        left_weight = left_counts.sum() / total_samples
        right_weight = right_counts.sum() / total_samples
        expected_weighted_gini = left_weight * left_gini + right_weight * right_gini
        expected_gain = parent_gini - expected_weighted_gini

        # Test against our implementation - note that our criterion returns negative impurity decrease
        actual_score = gini_criterion.evaluate_split(parent_counts, left_counts, right_counts)
        expected_score = expected_weighted_gini - parent_gini  # Negative of the gain

        # Should match within floating point precision
        assert abs(expected_score - actual_score) < 1e-10, f"Gini calculation mismatch: expected {expected_score}, got {actual_score}"

    def test_entropy_criterion_correctness(self, mushroom_data):
        """Validate entropy criterion against manual calculation."""
        df, sketches, feature_mapping = mushroom_data

        entropy_criterion = EntropyCriterion()

        # Test case
        left_counts = np.array([80.0, 20.0])
        right_counts = np.array([30.0, 70.0])
        parent_counts = left_counts + right_counts

        # Manual entropy calculation
        def manual_entropy(counts):
            if counts.sum() == 0:
                return 0.0
            proportions = counts / counts.sum()
            # Avoid log(0) by filtering out zeros
            proportions = proportions[proportions > 0]
            return -np.sum(proportions * np.log2(proportions))

        left_entropy = manual_entropy(left_counts)
        right_entropy = manual_entropy(right_counts)
        parent_entropy = manual_entropy(parent_counts)

        # Weighted average
        total_samples = parent_counts.sum()
        left_weight = left_counts.sum() / total_samples
        right_weight = right_counts.sum() / total_samples
        expected_weighted_entropy = left_weight * left_entropy + right_weight * right_entropy
        expected_gain = parent_entropy - expected_weighted_entropy

        # Test against our implementation - note that our criterion returns negative impurity decrease
        actual_score = entropy_criterion.evaluate_split(parent_counts, left_counts, right_counts)
        expected_score = expected_weighted_entropy - parent_entropy  # Negative of the gain

        assert abs(expected_score - actual_score) < 1e-10, f"Entropy calculation mismatch: expected {expected_score}, got {actual_score}"

    def test_prediction_accuracy_validation(self, mushroom_data):
        """Validate prediction accuracy against known mushroom dataset characteristics."""
        df, sketches, feature_mapping = mushroom_data

        # Train classifier
        clf = ThetaSketchDecisionTreeClassifier(criterion="gini", max_depth=5, verbose=0)
        clf.fit(sketches, feature_mapping)

        # Create test data from original dataframe
        X_test = self._convert_mushroom_to_binary_matrix(df.head(1000), feature_mapping)
        y_test = (df.head(1000)['class'] == 'poisonous').astype(int).values

        # Make predictions
        predictions = clf.predict(X_test)
        probabilities = clf.predict_proba(X_test)

        # Validate prediction properties
        assert len(predictions) == len(y_test), "Prediction count mismatch"
        assert all(p in [0, 1] for p in predictions), "Invalid prediction values"
        assert probabilities.shape == (len(y_test), 2), "Probability shape mismatch"
        assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities don't sum to 1"

        # Mushroom dataset should achieve reasonably high accuracy (it's highly separable)
        accuracy = np.mean(predictions == y_test)
        assert accuracy > 0.85, f"Accuracy too low: {accuracy:.3f} (expected > 0.85)"

        # Top feature should be odor-related (domain knowledge)
        top_features = clf.get_top_features(5)
        top_feature_names = [name for name, _ in top_features]
        odor_features = [name for name in top_feature_names if 'odor' in name.lower()]
        assert len(odor_features) > 0, f"Expected odor feature in top 5, got: {top_feature_names}"

    def test_sketch_operation_validation(self, mushroom_data):
        """Validate sketch intersection counts against raw data counts."""
        df, sketches, feature_mapping = mushroom_data

        # Select a specific feature to validate
        test_feature = "odor=n"  # odor=none

        if test_feature not in sketches['positive']:
            pytest.skip(f"Feature {test_feature} not found in sketches")

        # Count manually from raw data: positive class AND odor=none
        positive_with_feature = len(df[(df['class'] == 'poisonous') & (df['odor'] == 'n')])
        positive_without_feature = len(df[(df['class'] == 'poisonous') & (df['odor'] != 'n')])

        print(f"Manual counts: with_feature={positive_with_feature}, without_feature={positive_without_feature}")

        # Count via sketch intersection
        sketch_with_feature = sketches['positive'][test_feature][0].get_estimate()  # present tuple
        sketch_without_feature = sketches['positive'][test_feature][1].get_estimate()  # absent tuple

        print(f"Sketch counts: with_feature={sketch_with_feature}, without_feature={sketch_without_feature}")

        # Validate counts are reasonably close (within theta sketch error bounds)
        def validate_sketch_accuracy(manual_count, sketch_count, description):
            if manual_count == 0:
                assert sketch_count < 50, f"{description}: Expected ~0, got {sketch_count}"
            else:
                relative_error = abs(manual_count - sketch_count) / manual_count
                assert relative_error < 0.25, f"{description}: Manual={manual_count}, Sketch={sketch_count}, Error={relative_error:.3f}"

        validate_sketch_accuracy(positive_with_feature, sketch_with_feature, "Positive with feature")
        validate_sketch_accuracy(positive_without_feature, sketch_without_feature, "Positive without feature")

    def test_edge_case_robustness(self, mushroom_data):
        """Test algorithm robustness with edge cases."""
        df, sketches, feature_mapping = mushroom_data

        # Test 1: Single sample (should not crash)
        n_features = len(feature_mapping)
        single_sample = np.random.randint(0, 2, size=(1, n_features))  # Correct number of features
        clf = ThetaSketchDecisionTreeClassifier(criterion="gini", max_depth=5)
        clf.fit(sketches, feature_mapping)

        predictions = clf.predict(single_sample)
        assert len(predictions) == 1
        assert predictions[0] in [0, 1]

        # Test 2: Empty predictions array
        empty_array = np.array([]).reshape(0, n_features)
        empty_predictions = clf.predict(empty_array)
        assert len(empty_predictions) == 0

        # Test 3: Large batch (performance check) - reduced size
        large_batch = np.random.randint(0, 2, size=(1000, n_features))
        large_predictions = clf.predict(large_batch)
        assert len(large_predictions) == 1000
        assert all(p in [0, 1] for p in large_predictions[:100])  # Check first 100

    def test_model_persistence_correctness(self, mushroom_data):
        """Validate that model save/load preserves prediction behavior."""
        df, sketches, feature_mapping = mushroom_data

        # Train original model
        clf_original = ThetaSketchDecisionTreeClassifier(criterion="gini", max_depth=5)
        clf_original.fit(sketches, feature_mapping)

        # Create test data
        X_test = self._convert_mushroom_to_binary_matrix(df.head(100), feature_mapping)
        original_predictions = clf_original.predict(X_test)
        original_probabilities = clf_original.predict_proba(X_test)

        # Save and load model
        import tempfile
        import os
        from theta_sketch_tree.model_persistence import ModelPersistence

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            ModelPersistence.save_model(clf_original, tmp.name)
            clf_loaded = ModelPersistence.load_model(tmp.name)
            os.unlink(tmp.name)

        # Compare predictions
        loaded_predictions = clf_loaded.predict(X_test)
        loaded_probabilities = clf_loaded.predict_proba(X_test)

        np.testing.assert_array_equal(original_predictions, loaded_predictions, "Predictions changed after save/load")
        np.testing.assert_array_almost_equal(original_probabilities, loaded_probabilities, decimal=10, err_msg="Probabilities changed after save/load")

    def _convert_mushroom_to_binary_matrix(self, df: pd.DataFrame, feature_mapping: Dict[str, int]) -> np.ndarray:
        """Convert mushroom dataframe to binary feature matrix for prediction."""
        n_samples = len(df)
        n_features = len(feature_mapping)
        X = np.zeros((n_samples, n_features), dtype=int)

        for feature_name, feature_idx in feature_mapping.items():
            if '=' in feature_name:
                column, value = feature_name.split('=', 1)
                if column in df.columns:
                    # Set to 1 where the condition is true
                    X[:, feature_idx] = (df[column] == value).astype(int)

        return X