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
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

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


class TestPruningPerformance:
    """Performance tests specifically for pruning algorithms."""

    def create_large_dataset_for_pruning(self, n_samples: int, n_features: int = 50) -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
        """Create large synthetic dataset for pruning performance testing with realistic signal."""
        feature_names = [f'feature_{i}' for i in range(n_features)]
        base_card = n_samples // 4

        # Create positive class sketches with REALISTIC discrimination
        positive_sketches = {'total': MockThetaSketch(n_samples // 2)}
        for i, feature_name in enumerate(feature_names):
            # Create features with actual discriminative power
            if i < n_features // 3:  # First third: strong discriminative features
                present_card = int(base_card * (0.8 + i * 0.1 / n_features))  # Higher for positive
                absent_card = int(base_card * (0.3 - i * 0.1 / n_features))   # Lower for positive
            elif i < 2 * n_features // 3:  # Second third: medium discriminative
                present_card = int(base_card * (0.6 + i * 0.05 / n_features))
                absent_card = int(base_card * (0.5 - i * 0.05 / n_features))
            else:  # Last third: weak/noise features
                present_card = int(base_card * (0.5 + np.random.normal(0, 0.1)))
                absent_card = int(base_card * (0.5 + np.random.normal(0, 0.1)))

            # Ensure positive cardinalities
            present_card = max(100, present_card)
            absent_card = max(100, absent_card)

            positive_sketches[feature_name] = (
                MockThetaSketch(present_card),
                MockThetaSketch(absent_card)
            )

        # Create negative class sketches with OPPOSITE patterns
        negative_sketches = {'total': MockThetaSketch(n_samples // 2)}
        for i, feature_name in enumerate(feature_names):
            if i < n_features // 3:  # Strong discriminative: opposite pattern
                present_card = int(base_card * (0.3 + i * 0.1 / n_features))  # Lower for negative
                absent_card = int(base_card * (0.8 - i * 0.1 / n_features))   # Higher for negative
            elif i < 2 * n_features // 3:  # Medium discriminative: opposite pattern
                present_card = int(base_card * (0.4 + i * 0.05 / n_features))
                absent_card = int(base_card * (0.6 - i * 0.05 / n_features))
            else:  # Noise features: similar to positive
                present_card = int(base_card * (0.5 + np.random.normal(0, 0.1)))
                absent_card = int(base_card * (0.5 + np.random.normal(0, 0.1)))

            # Ensure positive cardinalities
            present_card = max(100, present_card)
            absent_card = max(100, absent_card)

            negative_sketches[feature_name] = (
                MockThetaSketch(present_card),
                MockThetaSketch(absent_card)
            )

        sketch_data = {'positive': positive_sketches, 'negative': negative_sketches}
        feature_mapping = {name: i for i, name in enumerate(feature_names)}

        # Create validation data with realistic patterns
        np.random.seed(42)
        X_val = np.zeros((n_samples // 4, n_features))
        y_val = np.zeros(n_samples // 4)

        for i in range(n_samples // 4):
            # Create realistic feature patterns based on class
            y_val[i] = np.random.randint(0, 2)

            for j in range(n_features):
                if j < n_features // 3:  # Strong features follow class pattern
                    prob = 0.8 if y_val[i] == 1 else 0.2
                elif j < 2 * n_features // 3:  # Medium features
                    prob = 0.7 if y_val[i] == 1 else 0.3
                else:  # Noise features
                    prob = 0.5

                X_val[i, j] = 1 if np.random.random() < prob else 0

        return sketch_data, feature_mapping, X_val, y_val.astype(int)

    @pytest.mark.parametrize("n_samples", [1000, 5000, 10000])
    @pytest.mark.parametrize("pruning_method", ['cost_complexity', 'validation', 'min_impurity'])
    def test_pruning_algorithm_performance(self, n_samples: int, pruning_method: str):
        """Profile pruning algorithms on large datasets."""
        # Create large dataset (reduce features for very large n_samples)
        n_features = 50 if n_samples >= 8000 else 75
        sketch_data, feature_mapping, X_val, y_val = self.create_large_dataset_for_pruning(
            n_samples, n_features=n_features
        )

        # Configure pruning parameters
        pruning_params = {
            'cost_complexity': {'min_impurity_decrease': 0.001},
            'validation': {'validation_fraction': 0.25},
            'min_impurity': {'min_impurity_decrease': 0.001}
        }

        # Train classifier with pruning
        clf = ThetaSketchDecisionTreeClassifier(
            criterion='gini',
            max_depth=10,  # More reasonable for large datasets
            pruning=pruning_method,
            verbose=1,
            **pruning_params[pruning_method]
        )

        # Measure total training + pruning time
        memory_before = self.measure_memory_usage() if HAS_PSUTIL else 0
        start_time = time.time()

        if pruning_method == 'validation':
            clf.fit(sketch_data, feature_mapping, X_val=X_val, y_val=y_val)
        else:
            clf.fit(sketch_data, feature_mapping)

        end_time = time.time()
        memory_after = self.measure_memory_usage() if HAS_PSUTIL else 0

        total_time = end_time - start_time
        memory_delta = memory_after - memory_before

        # Collect tree metrics
        tree_nodes = self.count_tree_nodes(clf.tree_) if hasattr(clf, 'tree_') else 0
        tree_depth = self.measure_tree_depth(clf.tree_) if hasattr(clf, 'tree_') else 0

        # Performance assertions
        assert total_time < 300.0, f"Pruning too slow: {total_time:.2f}s for {n_samples} samples"
        assert clf._is_fitted, "Classifier should be fitted after pruning"

        # Log detailed performance metrics
        print(f"\nPruning Performance [{pruning_method}, {n_samples} samples]:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Memory delta: {memory_delta:.1f}MB" if HAS_PSUTIL else "  Memory: N/A")
        print(f"  Final tree: {tree_nodes} nodes, depth {tree_depth}")

        # Verify pruning effectiveness on large datasets (more reasonable threshold)
        if tree_nodes > 0:
            print(f"  Tree complexity: {tree_nodes} nodes")
            # Allow larger trees for large datasets - focus on reasonable upper bounds
            max_nodes = min(2000, n_samples // 5)  # Scale with dataset size
            assert tree_nodes < max_nodes, f"Pruning ineffective: {tree_nodes} nodes (expected < {max_nodes})"

    def test_pruning_scalability(self):
        """Test pruning scalability with increasing dataset sizes."""
        sizes = [2000, 5000, 8000]
        results = []

        for n_samples in sizes:
            sketch_data, feature_mapping, X_val, y_val = self.create_large_dataset_for_pruning(
                n_samples, n_features=50
            )

            clf = ThetaSketchDecisionTreeClassifier(
                criterion='gini',
                max_depth=12,
                pruning='cost_complexity',
                min_impurity_decrease=0.01,
                verbose=0
            )

            start_time = time.time()
            clf.fit(sketch_data, feature_mapping)
            training_time = time.time() - start_time

            tree_nodes = self.count_tree_nodes(clf.tree_)
            results.append({
                'samples': n_samples,
                'time': training_time,
                'nodes': tree_nodes
            })

            print(f"Scale test [{n_samples} samples]: {training_time:.2f}s, {tree_nodes} nodes")

        # Check that time scaling is reasonable (should be roughly linear or sub-quadratic)
        time_ratio = results[-1]['time'] / results[0]['time']
        sample_ratio = results[-1]['samples'] / results[0]['samples']

        # Time should not grow exponentially with data size
        assert time_ratio <= sample_ratio * 2, f"Poor time scaling: {time_ratio:.1f}x for {sample_ratio:.1f}x data"

        print(f"\nScalability Analysis:")
        print(f"  Sample ratio: {sample_ratio:.1f}x")
        print(f"  Time ratio: {time_ratio:.1f}x")
        print(f"  Scaling efficiency: {'Good' if time_ratio <= sample_ratio * 1.5 else 'Poor'}")

    def measure_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if not HAS_PSUTIL:
            return 0.0
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

    def test_validation_data_conversion_benchmark(self):
        """Benchmark current manual validation data conversion process."""
        # Create test dataset
        sketch_data, feature_mapping, _, _ = self.create_large_dataset_for_pruning(5000, 100)

        # Simulate DataFrame → binary conversion (current manual process)
        import pandas as pd

        # Create mock DataFrame similar to real usage
        n_samples = 2000
        categorical_data = {
            f'cat_feature_{i}': np.random.choice(['A', 'B', 'C'], n_samples)
            for i in range(20)
        }
        numerical_data = {
            f'num_feature_{i}': np.random.randn(n_samples)
            for i in range(30)
        }

        df = pd.DataFrame({**categorical_data, **numerical_data})
        df['target'] = np.random.randint(0, 2, n_samples)

        # Benchmark current conversion process
        start_time = time.time()

        # Simulate manual one-hot encoding and binning process
        X_binary = np.zeros((len(df), 100))  # Mock binary feature matrix

        # Manual conversion simulation (what users currently do)
        for i, (_, row) in enumerate(df.iterrows()):
            # Categorical features
            for j, cat_col in enumerate([col for col in df.columns if 'cat_' in col]):
                feature_idx = j * 3  # 3 categories per feature
                if row[cat_col] == 'A':
                    X_binary[i, feature_idx] = 1
                elif row[cat_col] == 'B':
                    X_binary[i, feature_idx + 1] = 1
                elif row[cat_col] == 'C':
                    X_binary[i, feature_idx + 2] = 1

            # Numerical features (binning)
            for j, num_col in enumerate([col for col in df.columns if 'num_' in col]):
                feature_idx = 60 + j  # Start after categorical features
                X_binary[i, feature_idx] = 1 if row[num_col] > 0 else 0

        conversion_time = time.time() - start_time

        # Performance baseline for optimization
        print(f"\nValidation Conversion Benchmark [{n_samples} samples, 100 features]:")
        print(f"  Current manual process: {conversion_time:.3f}s")
        print(f"  Conversion rate: {n_samples / conversion_time:.0f} samples/sec")

        # This establishes baseline for optimization
        assert conversion_time < 60.0, f"Current conversion too slow: {conversion_time:.2f}s"

        # Don't return value to avoid pytest warning

    def test_optimized_validation_conversion_performance(self):
        """Test performance improvements from optimized validation conversion."""
        # Create large test dataset
        sketch_data, feature_mapping, _, _ = self.create_large_dataset_for_pruning(3000, 50)

        # Train classifier to get feature mapping
        clf = ThetaSketchDecisionTreeClassifier(
            criterion='gini',
            max_depth=8,
            enable_cache=True,
            verbose=0
        )
        clf.fit(sketch_data, feature_mapping)

        # Create validation DataFrame
        import pandas as pd
        n_samples = 1000

        # Create binary data that matches feature_mapping (no feature engineering)
        validation_data = {}
        feature_names = list(feature_mapping.keys())
        for i, fname in enumerate(feature_names):
            validation_data[fname] = np.random.randint(0, 2, n_samples)  # Binary features
        validation_data['target'] = np.random.randint(0, 2, n_samples)

        df_val = pd.DataFrame(validation_data)

        # Test optimized conversion
        start_time = time.time()
        X_val_opt, y_val_opt = clf.convert_validation_data_optimized(df_val, 'target')
        optimized_time = time.time() - start_time

        # Test manual conversion (baseline) - simple binary conversion
        start_time = time.time()
        X_val_manual = np.zeros((len(df_val), len(feature_mapping)))
        # Simulate manual conversion process (simple binary mapping)
        for i, (_, row) in enumerate(df_val.iterrows()):
            for j, fname in enumerate(feature_names):
                X_val_manual[i, j] = 1 if row[fname] else 0
        manual_time = time.time() - start_time

        # Performance assertions
        assert optimized_time < manual_time, "Optimized conversion should be faster"
        speedup = manual_time / optimized_time if optimized_time > 0 else 1

        print(f"\nValidation Conversion Performance [{n_samples} samples]:")
        print(f"  Manual process: {manual_time:.3f}s")
        print(f"  Optimized process: {optimized_time:.3f}s")
        print(f"  Speedup: {speedup:.1f}x")

        # Test caching effectiveness
        start_time = time.time()
        X_val_cached, y_val_cached = clf.convert_validation_data_optimized(df_val, 'target')
        cached_time = time.time() - start_time

        cache_speedup = optimized_time / cached_time if cached_time > 0 else 1
        print(f"  Cached conversion: {cached_time:.3f}s")
        print(f"  Cache speedup: {cache_speedup:.1f}x")

        # Verify correctness
        assert X_val_opt.shape == X_val_cached.shape, "Cached result should match optimized"
        np.testing.assert_array_equal(X_val_opt, X_val_cached, "Cached conversion should be identical")

        # Get conversion stats
        stats = clf.get_validation_conversion_stats()
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.1f}%")
        print(f"  Total conversions: {stats['total_conversions']}")

        assert speedup > 1.2, f"Insufficient speedup: {speedup:.1f}x (expected >1.2x)"
        if cached_time > 0:
            assert cache_speedup > 2.0, f"Insufficient cache speedup: {cache_speedup:.1f}x (expected >2.0x)"

    def test_pruning_with_optimized_validation(self):
        """Test end-to-end pruning performance with optimized validation conversion."""
        # Create dataset
        sketch_data, feature_mapping, X_val, y_val = self.create_large_dataset_for_pruning(8000, 100)

        # Test validation pruning with progress bars and optimization
        clf = ThetaSketchDecisionTreeClassifier(
            criterion='gini',
            max_depth=12,
            pruning='validation',
            validation_fraction=0.25,
            enable_cache=True,
            verbose=1
        )

        print("\n🔬 Testing Optimized Validation Pruning...")
        start_time = time.time()
        clf.fit(sketch_data, feature_mapping, X_val=X_val, y_val=y_val)
        total_time = time.time() - start_time

        # Verify pruning worked
        assert clf._is_fitted, "Classifier should be fitted"
        tree_nodes = self.count_tree_nodes(clf.tree_)

        print(f"\n✅ Validation Pruning Complete:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Final tree: {tree_nodes} nodes")
        print(f"  Dataset: {len(X_val)} validation samples, {len(feature_mapping)} features")

        # Performance assertion for large dataset
        assert total_time < 300.0, f"Optimized pruning too slow: {total_time:.1f}s"
        assert tree_nodes > 0, "Tree should have been built"
        assert tree_nodes < 300, f"Tree too large after pruning: {tree_nodes} nodes"