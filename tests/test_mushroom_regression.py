"""
Regression tests for mushroom dataset to prevent output changes during refactoring.

This test module validates that the decision tree outputs remain consistent
with the baseline reference outputs generated before code cleanup.
"""

import json
import pytest
import numpy as np

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration
from numpy.testing import assert_allclose
from tests.test_binary_classification_sketches import (
    load_mushroom_dataset,
    create_mushroom_sketches,
    create_mushroom_feature_mapping,
    tree_to_json
)
from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier


class TestMushroomRegression:
    """Regression tests using baseline reference outputs."""

    @pytest.fixture(scope="class")
    def mushroom_data(self):
        """Load mushroom data once for all tests."""
        df = load_mushroom_dataset()
        sketches = create_mushroom_sketches(df)
        feature_mapping = create_mushroom_feature_mapping(sketches)
        return df, sketches, feature_mapping

    @pytest.fixture(scope="class")
    def baseline_outputs(self):
        """Load baseline reference outputs."""
        try:
            with open("mushroom_baseline_outputs.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            pytest.skip("Baseline outputs not found. Run generate_mushroom_baselines.py first.")

    def _compare_tree_structures_recursively(self, current_node, baseline_node, path, log_file=None):
        """
        Recursively compare two tree structures for exact match.

        This addresses the critical flaw where test_core_criteria_regression only
        compared top-level metadata but ignored the actual tree structure.

        Args:
            current_node: Current tree node to compare
            baseline_node: Baseline tree node to compare against
            path: Current path in the tree (for error reporting)
            log_file: Optional file handle to write detailed logs
        """
        # Log basic node info
        node_info = f"🌳 Comparing node at {path}:"
        node_details = f"   Type: {'LEAF' if current_node['is_leaf'] else 'SPLIT'}, Depth: {current_node['depth']}, Samples: {current_node['n_samples']:.0f}"

        print(node_info)
        print(node_details)
        if log_file:
            log_file.write(f"{node_info}\n")
            log_file.write(f"{node_details}\n")

        # Compare node metadata
        try:
            assert current_node["is_leaf"] == baseline_node["is_leaf"], f"Leaf status mismatch at {path}"
            assert current_node["depth"] == baseline_node["depth"], f"Depth mismatch at {path}"
            print("   ✅ Node metadata matches")
            if log_file:
                log_file.write("   ✅ Node metadata matches\n")
        except AssertionError as e:
            error_msg = f"   ❌ METADATA ERROR: {str(e)}"
            print(error_msg)
            if log_file:
                log_file.write(f"{error_msg}\n")
            raise

        # Compare sample counts (allow small float differences)
        current_samples = current_node["n_samples"]
        baseline_samples = baseline_node["n_samples"]
        sample_diff = abs(current_samples - baseline_samples)
        sample_info = f"   📊 Sample counts: current={current_samples:.0f}, baseline={baseline_samples:.0f}, diff={sample_diff:.6f}"
        print(sample_info)
        if log_file:
            log_file.write(f"{sample_info}\n")

        try:
            assert sample_diff < 1.0, f"Sample count mismatch at {path}: {current_samples} vs {baseline_samples}"
            print("   ✅ Sample counts match")
            if log_file:
                log_file.write("   ✅ Sample counts match\n")
        except AssertionError as e:
            error_msg = f"   ❌ SAMPLE COUNT ERROR: {str(e)}"
            print(error_msg)
            if log_file:
                log_file.write(f"{error_msg}\n")
            raise

        # Compare class counts
        current_counts = current_node["class_counts"]
        baseline_counts = baseline_node["class_counts"]
        class_info = f"   📈 Class counts: current={current_counts}, baseline={baseline_counts}"
        print(class_info)
        if log_file:
            log_file.write(f"{class_info}\n")

        try:
            assert len(current_counts) == len(baseline_counts), f"Class count length mismatch at {path}"
            for i, (curr, base) in enumerate(zip(current_counts, baseline_counts)):
                assert abs(curr - base) < 1.0, f"Class count {i} mismatch at {path}: {curr} vs {base}"
            print("   ✅ Class counts match")
            if log_file:
                log_file.write("   ✅ Class counts match\n")
        except AssertionError as e:
            error_msg = f"   ❌ CLASS COUNT ERROR: {str(e)}"
            print(error_msg)
            if log_file:
                log_file.write(f"{error_msg}\n")
            raise

        # Compare impurity (allow small numerical differences)
        current_impurity = current_node["impurity"]
        baseline_impurity = baseline_node["impurity"]
        impurity_diff = abs(current_impurity - baseline_impurity)
        impurity_info = f"   🎯 Impurity: current={current_impurity:.8f}, baseline={baseline_impurity:.8f}, diff={impurity_diff:.2e}"
        print(impurity_info)
        if log_file:
            log_file.write(f"{impurity_info}\n")

        try:
            assert impurity_diff < 1e-6, f"Impurity mismatch at {path}: {current_impurity} vs {baseline_impurity}"
            print("   ✅ Impurity matches")
            if log_file:
                log_file.write("   ✅ Impurity matches\n")
        except AssertionError as e:
            error_msg = f"   ❌ IMPURITY ERROR: {str(e)}"
            print(error_msg)
            if log_file:
                log_file.write(f"{error_msg}\n")
            raise

        # If it's a split node, compare split details and recurse to children
        if not current_node["is_leaf"]:
            # Compare split feature details
            split_info = f"   🔀 Split feature: {current_node['feature_name']} (idx={current_node['feature_idx']})"
            split_condition = f"   📋 Condition: {current_node.get('split_condition', 'N/A')}"
            print(split_info)
            print(split_condition)
            if log_file:
                log_file.write(f"{split_info}\n")
                log_file.write(f"{split_condition}\n")

            try:
                assert current_node["feature_idx"] == baseline_node["feature_idx"], f"Feature index mismatch at {path}"
                assert current_node["feature_name"] == baseline_node["feature_name"], f"Feature name mismatch at {path}"
                print("   ✅ Split feature matches")
                if log_file:
                    log_file.write("   ✅ Split feature matches\n")
            except AssertionError as e:
                error_msg = f"   ❌ SPLIT FEATURE ERROR: {str(e)}"
                print(error_msg)
                if log_file:
                    log_file.write(f"{error_msg}\n")
                raise

            # Check for children presence
            try:
                assert "left" in current_node and "left" in baseline_node, f"Missing left child at {path}"
                assert "right" in current_node and "right" in baseline_node, f"Missing right child at {path}"
                print("   ✅ Both children present")
                if log_file:
                    log_file.write("   ✅ Both children present\n")
            except AssertionError as e:
                error_msg = f"   ❌ CHILDREN ERROR: {str(e)}"
                print(error_msg)
                if log_file:
                    log_file.write(f"{error_msg}\n")
                raise

            # Add separator before recursing
            separator = f"\n{'─' * 80}"
            print(separator)
            if log_file:
                log_file.write(f"{separator}\n")

            # Recursively compare children
            self._compare_tree_structures_recursively(current_node["left"], baseline_node["left"], f"{path}/left", log_file)
            self._compare_tree_structures_recursively(current_node["right"], baseline_node["right"], f"{path}/right", log_file)
        else:
            # Leaf node details
            prediction_info = f"   🍃 Leaf prediction: {current_node.get('prediction', 'N/A')}"
            probabilities = current_node.get('probabilities', [])
            prob_info = f"   📊 Probabilities: {probabilities}"
            print(prediction_info)
            print(prob_info)
            if log_file:
                log_file.write(f"{prediction_info}\n")
                log_file.write(f"{prob_info}\n")

        # Add spacing after each node
        print()
        if log_file:
            log_file.write("\n")

    def serialize_current_output(self, clf, description):
        """Serialize current classifier output for comparison."""
        try:
            tree_json = tree_to_json(clf.tree_, max_depth=10)

            # Feature importances
            feature_importances = clf.feature_importances_.tolist() if hasattr(clf, 'feature_importances_') else []

            # Top features
            try:
                top_features = [(name, float(imp)) for name, imp in clf.get_top_features(10)]
            except:
                top_features = []

            return {
                "description": description,
                "tree_structure": tree_json,
                "tree_depth": tree_json.get('depth', 0) if tree_json else 0,
                "n_samples": tree_json.get('n_samples', 0) if tree_json else 0,
                "feature_importances": feature_importances,
                "top_features": top_features,
                "n_features": clf.n_features_in_ if hasattr(clf, 'n_features_in_') else 0,
                "classes": clf.classes_.tolist() if hasattr(clf, 'classes_') else []
            }
        except Exception as e:
            return {"error": f"Serialization failed: {str(e)}"}

    @pytest.mark.parametrize("config_name", [
        "default_gini",
        "entropy_shallow",
        "gain_ratio_medium",
        "binomial_deep",
        "chi_square_default"
    ])
    def test_core_criteria_regression(self, mushroom_data, baseline_outputs, config_name):
        """Test that core criteria produce consistent outputs."""
        df, sketches, feature_mapping = mushroom_data

        if config_name not in baseline_outputs:
            pytest.skip(f"No baseline for {config_name}")

        baseline = baseline_outputs[config_name]
        if "error" in baseline:
            pytest.skip(f"Baseline had error: {baseline['error']}")

        # Recreate the configuration
        config_map = {
            "default_gini": {"criterion": "gini", "max_depth": 5},
            "entropy_shallow": {"criterion": "entropy", "max_depth": 3},
            "gain_ratio_medium": {"criterion": "gain_ratio", "max_depth": 7},
            "binomial_deep": {"criterion": "binomial", "max_depth": 10},
            "chi_square_default": {"criterion": "chi_square", "max_depth": 5}
        }

        config = config_map[config_name]

        # Train classifier
        clf = ThetaSketchDecisionTreeClassifier(verbose=0, **config)
        clf.fit(sketches, feature_mapping)

        # Get current output
        current = self.serialize_current_output(clf, baseline["description"])

        # RECORD MODE: Print complete JSON output for manual inspection
        print(f"\n{'='*60}")
        print(f"CURRENT OUTPUT FOR {config_name.upper()}:")
        print(f"{'='*60}")
        print(json.dumps(current, indent=2, default=str))
        print(f"{'='*60}")

        print(f"\n{'='*60}")
        print(f"BASELINE OUTPUT FOR {config_name.upper()}:")
        print(f"{'='*60}")
        print(json.dumps(baseline, indent=2, default=str))
        print(f"{'='*60}\n")

        # Compare key metrics
        assert current["n_features"] == baseline["n_features"], f"Feature count mismatch in {config_name}"
        assert current["classes"] == baseline["classes"], f"Classes mismatch in {config_name}"
        assert current["tree_depth"] == baseline["tree_depth"], f"Tree depth mismatch in {config_name}"

        # Compare sample counts (allow small differences due to floating point)
        assert abs(current["n_samples"] - baseline["n_samples"]) < 1, f"Sample count mismatch in {config_name}"

        # CRITICAL: Compare full tree structure recursively with detailed logging
        print(f"Starting recursive tree comparison for {config_name}...")

        # Create detailed log file for this configuration
        log_filename = f"tree_comparison_{config_name}.log"
        print(f"📝 Writing detailed logs to: {log_filename}")

        with open(log_filename, 'w') as log_file:
            # Write log header
            header = f"""{'='*100}
RECURSIVE TREE STRUCTURE COMPARISON LOG
Configuration: {config_name.upper()}
Timestamp: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*100}

"""
            log_file.write(header)
            print(f"🌲 Beginning detailed recursive comparison...")

            # Perform recursive comparison with file logging
            self._compare_tree_structures_recursively(
                current["tree_structure"],
                baseline["tree_structure"],
                f"{config_name}/root",
                log_file
            )

            # Write log footer
            footer = f"""
{'='*100}
RECURSIVE TREE COMPARISON COMPLETED SUCCESSFULLY ✓
Configuration: {config_name.upper()}
All nodes, splits, and impurities match within tolerance
{'='*100}
"""
            log_file.write(footer)

        print(f"✅ Recursive tree comparison completed for {config_name}")
        print(f"📄 Detailed log saved to: {log_filename}")

    def test_feature_importance_consistency(self, mushroom_data, baseline_outputs):
        """Test that feature importance rankings remain consistent."""
        df, sketches, feature_mapping = mushroom_data

        # Test with default gini configuration
        if "default_gini" not in baseline_outputs:
            pytest.skip("No default_gini baseline")

        baseline = baseline_outputs["default_gini"]
        if "error" in baseline:
            pytest.skip(f"Baseline had error: {baseline['error']}")

        # Train classifier
        clf = ThetaSketchDecisionTreeClassifier(
            criterion="gini", max_depth=5, verbose=0
        )
        clf.fit(sketches, feature_mapping)

        # Get current top features
        current_top_features = [(name, float(imp)) for name, imp in clf.get_top_features(10)]
        baseline_top_features = baseline["top_features"]

        # Check that top 3 features are the same (order and names)
        for i in range(min(3, len(current_top_features), len(baseline_top_features))):
            current_feature = current_top_features[i][0]
            baseline_feature = baseline_top_features[i][0]

            assert current_feature == baseline_feature, (
                f"Top feature #{i+1} mismatch: current='{current_feature}', "
                f"baseline='{baseline_feature}'"
            )

            # Check importance values are reasonably close (within 5%)
            current_importance = current_top_features[i][1]
            baseline_importance = baseline_top_features[i][1]

            relative_diff = abs(current_importance - baseline_importance) / baseline_importance
            assert relative_diff < 0.05, (
                f"Feature importance for '{current_feature}' changed significantly: "
                f"current={current_importance:.6f}, baseline={baseline_importance:.6f}, "
                f"relative_diff={relative_diff:.3f}"
            )

    def test_tree_structure_consistency(self, mushroom_data, baseline_outputs):
        """Test that tree structure remains consistent for key configurations."""
        df, sketches, feature_mapping = mushroom_data

        # Test shallow tree structure (easier to validate)
        config_name = "entropy_shallow"
        if config_name not in baseline_outputs:
            pytest.skip(f"No baseline for {config_name}")

        baseline = baseline_outputs[config_name]
        if "error" in baseline:
            pytest.skip(f"Baseline had error: {baseline['error']}")

        # Train classifier
        clf = ThetaSketchDecisionTreeClassifier(
            criterion="entropy", max_depth=3, verbose=0
        )
        clf.fit(sketches, feature_mapping)

        # Get tree structure
        current_tree = tree_to_json(clf.tree_, max_depth=10)
        baseline_tree = baseline["tree_structure"]

        # Compare root node properties
        assert current_tree["depth"] == baseline_tree["depth"], "Root depth mismatch"
        assert abs(current_tree["n_samples"] - baseline_tree["n_samples"]) < 1, "Root sample count mismatch"

        # Compare class counts (allow small floating point differences)
        current_counts = current_tree["class_counts"]
        baseline_counts = baseline_tree["class_counts"]

        for i, (current, baseline_val) in enumerate(zip(current_counts, baseline_counts)):
            assert abs(current - baseline_val) < 1, f"Class count {i} mismatch: {current} vs {baseline_val}"

    @pytest.mark.parametrize("config_name", [
        "gini_cost_complexity",
        "entropy_validation_pruning"
    ])
    def test_pruning_regression(self, mushroom_data, baseline_outputs, config_name):
        """Test that pruning methods produce consistent results."""
        df, sketches, feature_mapping = mushroom_data

        if config_name not in baseline_outputs:
            pytest.skip(f"No baseline for {config_name}")

        baseline = baseline_outputs[config_name]
        if "error" in baseline:
            pytest.skip(f"Baseline had error: {baseline['error']}")

        # Configuration mapping
        config_map = {
            "gini_cost_complexity": {
                "criterion": "gini", "max_depth": 8,
                "pruning": "cost_complexity"
            },
            "entropy_validation_pruning": {
                "criterion": "entropy", "max_depth": 10,
                "pruning": "validation"
            }
        }

        config = config_map[config_name]

        # Train classifier
        clf = ThetaSketchDecisionTreeClassifier(verbose=0, **config)
        clf.fit(sketches, feature_mapping)

        # Get current output
        current = self.serialize_current_output(clf, baseline["description"])

        # Pruning can have some variability, so use looser checks
        assert current["n_features"] == baseline["n_features"], f"Feature count mismatch in {config_name}"
        assert current["classes"] == baseline["classes"], f"Classes mismatch in {config_name}"

        # Tree structure may vary with pruning, but should be reasonable
        assert current["tree_depth"] <= baseline["tree_depth"] + 2, f"Tree too deep in {config_name}"
        assert current["n_samples"] == baseline["n_samples"], f"Sample count mismatch in {config_name}"

    def test_all_criteria_work(self, mushroom_data):
        """Comprehensive test that all criteria can process mushroom data without errors."""
        df, sketches, feature_mapping = mushroom_data

        criteria = ["gini", "entropy", "gain_ratio", "binomial", "chi_square"]

        for criterion in criteria:
            clf = ThetaSketchDecisionTreeClassifier(
                criterion=criterion, max_depth=3, verbose=0
            )

            # Should not raise exceptions
            clf.fit(sketches, feature_mapping)

            # Should be able to get predictions
            X_test = np.random.randint(0, 2, size=(10, len(feature_mapping)))
            predictions = clf.predict(X_test)
            probabilities = clf.predict_proba(X_test)

            # Basic sanity checks
            assert len(predictions) == 10
            assert probabilities.shape == (10, 2)
            assert all(p in [0, 1] for p in predictions)
