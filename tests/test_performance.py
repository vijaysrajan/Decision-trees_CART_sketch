"""
Performance tests for Theta Sketch Decision Tree.

This module benchmarks training and inference performance,
memory usage, and scalability characteristics.
"""

import time
import numpy as np
import pytest
from typing import Dict, Any, Tuple
from unittest.mock import Mock

from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier
from tests.test_mock_sketches import MockThetaSketch

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class PerformanceMetrics:
    """Track performance metrics during testing."""

    def __init__(self):
        self.training_time = 0.0
        self.prediction_time = 0.0
        self.memory_peak_mb = 0.0
        self.tree_depth = 0
        self.n_nodes = 0

    def __repr__(self):
        return (f"PerformanceMetrics(train={self.training_time:.3f}s, "
                f"predict={self.prediction_time:.3f}s, "
                f"memory={self.memory_peak_mb:.1f}MB, "
                f"depth={self.tree_depth}, nodes={self.n_nodes})")


class TestPerformance:
    """Performance benchmark tests."""

    def create_large_sketch_dataset(self, n_features: int, scale: str = 'medium') -> Tuple[Dict, Dict]:
        """Create synthetic sketch dataset for performance testing."""
        feature_names = [f'feature_{i}' for i in range(n_features)]

        # Scale determines sketch cardinalities
        if scale == 'small':
            base_card = 100
        elif scale == 'medium':
            base_card = 10000
        elif scale == 'large':
            base_card = 100000
        else:
            raise ValueError(f"Unknown scale: {scale}")

        # Create positive class sketches
        positive_sketches = {'total': MockThetaSketch(base_card * 2)}

        for i, feature_name in enumerate(feature_names):
            # Create varied cardinalities for realism
            present_card = base_card + (i * 1000) % base_card
            absent_card = base_card + ((i + 100) * 1000) % base_card

            positive_sketches[feature_name] = (
                MockThetaSketch(present_card),
                MockThetaSketch(absent_card)
            )

        # Create negative class sketches with different distributions
        negative_sketches = {'total': MockThetaSketch(base_card * 3)}

        for i, feature_name in enumerate(feature_names):
            present_card = base_card + ((i + 200) * 1000) % base_card
            absent_card = base_card + ((i + 300) * 1000) % base_card

            negative_sketches[feature_name] = (
                MockThetaSketch(present_card),
                MockThetaSketch(absent_card)
            )

        sketch_data = {'positive': positive_sketches, 'negative': negative_sketches}
        feature_mapping = {name: i for i, name in enumerate(feature_names)}

        return sketch_data, feature_mapping

    def measure_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if not HAS_PSUTIL:
            return 0.0  # Skip memory monitoring if psutil not available
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def count_tree_nodes(self, node) -> int:
        """Count total nodes in tree."""
        if node.is_leaf:
            return 1
        return 1 + self.count_tree_nodes(node.left) + self.count_tree_nodes(node.right)

    def measure_tree_depth(self, node) -> int:
        """Measure maximum tree depth."""
        if node.is_leaf:
            return 1
        left_depth = self.measure_tree_depth(node.left)
        right_depth = self.measure_tree_depth(node.right)
        return 1 + max(left_depth, right_depth)

    @pytest.mark.parametrize("n_features", [10, 50, 100])
    @pytest.mark.parametrize("scale", ['small', 'medium'])
    def test_training_performance(self, n_features: int, scale: str):
        """Benchmark training performance across different scales."""
        # Create dataset
        sketch_data, feature_mapping = self.create_large_sketch_dataset(n_features, scale)

        # Initialize classifier
        clf = ThetaSketchDecisionTreeClassifier(
            criterion='gini',
            max_depth=10,  # Limit depth for performance
            verbose=0
        )

        # Measure training performance
        memory_before = self.measure_memory_usage()
        start_time = time.time()

        clf.fit(sketch_data, feature_mapping)

        end_time = time.time()
        memory_after = self.measure_memory_usage()

        # Collect metrics
        metrics = PerformanceMetrics()
        metrics.training_time = end_time - start_time
        metrics.memory_peak_mb = memory_after - memory_before
        metrics.tree_depth = self.measure_tree_depth(clf.tree_)
        metrics.n_nodes = self.count_tree_nodes(clf.tree_)

        # Performance assertions
        assert metrics.training_time < 60.0, f"Training too slow: {metrics.training_time:.2f}s"
        if HAS_PSUTIL:
            assert metrics.memory_peak_mb < 500.0, f"Memory usage too high: {metrics.memory_peak_mb:.1f}MB"

        # Log results for analysis
        print(f"\nTraining Performance [{n_features} features, {scale} scale]: {metrics}")

        # Verify tree was built
        assert clf._is_fitted
        assert metrics.tree_depth > 0
        assert metrics.n_nodes > 0

    @pytest.mark.parametrize("n_samples", [100, 1000, 10000])
    def test_prediction_performance(self, n_samples: int):
        """Benchmark prediction performance with different batch sizes."""
        # Create small training dataset
        sketch_data, feature_mapping = self.create_large_sketch_dataset(20, 'small')

        # Train classifier
        clf = ThetaSketchDecisionTreeClassifier(criterion='gini', max_depth=8, verbose=0)
        clf.fit(sketch_data, feature_mapping)

        # Create prediction data
        X_test = np.random.randint(0, 2, size=(n_samples, 20))

        # Benchmark predict()
        start_time = time.time()
        predictions = clf.predict(X_test)
        predict_time = time.time() - start_time

        # Benchmark predict_proba()
        start_time = time.time()
        probabilities = clf.predict_proba(X_test)
        proba_time = time.time() - start_time

        # Performance assertions
        assert predict_time < 10.0, f"Prediction too slow: {predict_time:.2f}s for {n_samples} samples"
        assert proba_time < 10.0, f"Probability prediction too slow: {proba_time:.2f}s"

        # Verify outputs
        assert len(predictions) == n_samples
        assert probabilities.shape == (n_samples, 2)

        # Log results
        pred_rate = n_samples / predict_time
        proba_rate = n_samples / proba_time
        print(f"\nPrediction Performance [{n_samples} samples]:")
        print(f"  predict(): {predict_time:.3f}s ({pred_rate:.0f} samples/sec)")
        print(f"  predict_proba(): {proba_time:.3f}s ({proba_rate:.0f} samples/sec)")

    def test_feature_importance_performance(self):
        """Benchmark feature importance calculation performance."""
        # Create dataset with many features
        sketch_data, feature_mapping = self.create_large_sketch_dataset(100, 'medium')

        # Train classifier
        clf = ThetaSketchDecisionTreeClassifier(criterion='gini', max_depth=12, verbose=0)
        clf.fit(sketch_data, feature_mapping)

        # Benchmark feature importance access
        start_time = time.time()
        importances = clf.feature_importances_
        importance_time = time.time() - start_time

        # Benchmark top features
        start_time = time.time()
        top_features = clf.get_top_features(top_k=10)
        top_features_time = time.time() - start_time

        # Performance assertions
        assert importance_time < 1.0, f"Feature importance too slow: {importance_time:.3f}s"
        assert top_features_time < 1.0, f"Top features too slow: {top_features_time:.3f}s"

        # Verify outputs
        assert len(importances) == 100
        assert len(top_features) == 10
        assert np.allclose(importances.sum(), 1.0)

        print(f"\nFeature Importance Performance:")
        print(f"  feature_importances_: {importance_time:.3f}s")
        print(f"  get_top_features(): {top_features_time:.3f}s")

    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available for memory monitoring")
    def test_memory_efficiency(self):
        """Test memory usage patterns during tree building."""
        n_features_list = [10, 25, 50]
        memory_usage = []

        for n_features in n_features_list:
            # Create dataset
            sketch_data, feature_mapping = self.create_large_sketch_dataset(n_features, 'medium')

            # Measure memory before
            memory_before = self.measure_memory_usage()

            # Train classifier
            clf = ThetaSketchDecisionTreeClassifier(criterion='gini', max_depth=10, verbose=0)
            clf.fit(sketch_data, feature_mapping)

            # Measure memory after
            memory_after = self.measure_memory_usage()
            memory_delta = memory_after - memory_before
            memory_usage.append(memory_delta)

            print(f"Memory usage for {n_features} features: {memory_delta:.1f}MB")

        # Check that memory usage grows reasonably with feature count
        assert memory_usage[1] > memory_usage[0], "Memory should increase with more features"

        # But not exponentially
        ratio = memory_usage[-1] / memory_usage[0]
        assert ratio < 20.0, f"Memory growth too steep: {ratio:.1f}x"

    def test_scalability_limits(self):
        """Test behavior at scale boundaries."""
        # Test with maximum reasonable feature count
        large_features = 200
        sketch_data, feature_mapping = self.create_large_sketch_dataset(large_features, 'small')

        # Should complete within reasonable time
        clf = ThetaSketchDecisionTreeClassifier(
            criterion='gini',
            max_depth=8,  # Keep depth reasonable
            min_samples_split=10,  # Prevent overfitting
            verbose=0
        )

        start_time = time.time()
        clf.fit(sketch_data, feature_mapping)
        training_time = end_time = time.time() - start_time

        # Should handle large feature spaces
        assert clf._is_fitted
        assert training_time < 120.0, f"Large-scale training too slow: {training_time:.1f}s"

        # Test prediction on large batch
        large_batch = 5000
        X_test = np.random.randint(0, 2, size=(large_batch, large_features))

        start_time = time.time()
        predictions = clf.predict(X_test)
        prediction_time = time.time() - start_time

        assert len(predictions) == large_batch
        assert prediction_time < 30.0, f"Large batch prediction too slow: {prediction_time:.1f}s"

        print(f"\nScalability Test [{large_features} features, {large_batch} samples]:")
        print(f"  Training: {training_time:.2f}s")
        print(f"  Prediction: {prediction_time:.2f}s")

    @pytest.mark.parametrize("criterion", ['gini', 'entropy', 'gain_ratio', 'binomial'])
    def test_criterion_performance(self, criterion: str):
        """Compare performance across different split criteria."""
        # Create consistent dataset
        sketch_data, feature_mapping = self.create_large_sketch_dataset(30, 'medium')

        # Train with specific criterion
        clf = ThetaSketchDecisionTreeClassifier(
            criterion=criterion,
            max_depth=10,
            verbose=0
        )

        start_time = time.time()
        clf.fit(sketch_data, feature_mapping)
        training_time = time.time() - start_time

        # All criteria should complete reasonably fast
        assert training_time < 30.0, f"Criterion '{criterion}' too slow: {training_time:.2f}s"
        assert clf._is_fitted

        # Test prediction performance
        X_test = np.random.randint(0, 2, size=(1000, 30))
        start_time = time.time()
        predictions = clf.predict(X_test)
        prediction_time = time.time() - start_time

        assert prediction_time < 5.0
        assert len(predictions) == 1000

        print(f"Criterion '{criterion}': train={training_time:.2f}s, predict={prediction_time:.3f}s")


class TestPerformanceRegressions:
    """Tests to catch performance regressions."""

    def test_baseline_performance(self):
        """Establish performance baseline for regression testing."""
        # Standard test case
        from tests.test_mock_sketches import create_mock_sketch_data, create_feature_mapping
        sketch_data = create_mock_sketch_data()
        feature_mapping = create_feature_mapping()

        clf = ThetaSketchDecisionTreeClassifier(criterion='gini', verbose=0)

        # Training should be very fast for small dataset
        start_time = time.time()
        clf.fit(sketch_data, feature_mapping)
        training_time = time.time() - start_time

        # Baseline: small dataset should train in under 1 second
        assert training_time < 1.0, f"Baseline training too slow: {training_time:.3f}s"

        # Prediction should be instantaneous
        X_test = np.array([[1, 0], [0, 1], [1, 1]])
        start_time = time.time()
        predictions = clf.predict(X_test)
        prediction_time = time.time() - start_time

        assert prediction_time < 0.01, f"Baseline prediction too slow: {prediction_time:.3f}s"
        assert len(predictions) == 3

        print(f"Baseline Performance: train={training_time:.3f}s, predict={prediction_time:.4f}s")