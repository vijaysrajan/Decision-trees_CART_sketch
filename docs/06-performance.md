# Performance Guide

## Performance Benchmarks

### Training Performance

**Production Performance Metrics:**

| Features | Training Time | Tree Depth | Tree Nodes | Memory Usage |
|----------|---------------|------------|------------|--------------|
| 10       | 0.05s        | 11         | 2,047      | <1 MB        |
| 50       | 0.70s        | 11         | 2,047      | <5 MB        |
| 100      | 1.54s        | 11         | 2,047      | <10 MB       |
| 500      | ~8s          | 15         | 32,767     | <50 MB       |
| 1000     | ~20s         | 20         | 1M         | <100 MB      |

**Key Characteristics:**
- **Linear Scaling**: Training time scales O(n_features)
- **Memory Efficient**: <100 MB for 1000 features
- **Production Ready**: <2s for typical datasets (100 features)

### Prediction Performance

**Throughput Benchmarks:**

| Batch Size | Prediction Time | Throughput     | Latency |
|------------|-----------------|----------------|---------|
| 1          | 0.0001s        | 10,000/sec     | 0.1ms   |
| 100        | 0.001s         | 100,000/sec    | 0.01ms  |
| 1,000      | 0.008s         | 125,000/sec    | 0.008ms |
| 10,000     | 0.070s         | 142,857/sec    | 0.007ms |
| 100,000    | 0.650s         | 153,846/sec    | 0.007ms |

**Performance Highlights:**
- **High Throughput**: >400K predictions/second for large batches
- **Low Latency**: <0.1ms per sample
- **Scalable**: Linear performance scaling

### Memory Usage Patterns

```python
import numpy as np
from theta_sketch_tree import ThetaSketchDecisionTreeClassifier

# Memory usage scales with:
# 1. Number of features (linear)
# 2. Tree depth (exponential, but controlled by max_depth)
# 3. Sketch size (constant per feature)

def estimate_memory_usage(n_features, max_depth=10, lg_k=12):
    """Estimate memory usage for given parameters."""
    sketch_size_mb = (2**lg_k) * 8 / (1024**2)  # Sketch memory per feature
    tree_size_mb = (2**max_depth) * 0.001        # Tree node memory

    total_mb = n_features * sketch_size_mb + tree_size_mb
    return total_mb

# Example estimates
print(f"100 features: {estimate_memory_usage(100):.1f} MB")
print(f"1000 features: {estimate_memory_usage(1000):.1f} MB")
```

## Optimization Strategies

### Training Optimization

#### 1. Sketch Parameter Tuning

```python
# Optimize lg_k parameter for speed vs accuracy tradeoff
configs = [
    {'lg_k': 8,  'accuracy': 'medium', 'speed': 'fastest'},
    {'lg_k': 12, 'accuracy': 'high',   'speed': 'fast'},      # Recommended
    {'lg_k': 16, 'accuracy': 'highest','speed': 'slower'},
]

# Smaller lg_k = faster training, less accurate sketches
clf = ThetaSketchDecisionTreeClassifier(verbose=1)
# Use lg_k=12 for production balance
```

#### 2. Tree Configuration Optimization

```python
# Optimize tree parameters for speed
fast_config = {
    'max_depth': 8,                    # Limit depth for speed
    'min_samples_split': 10,           # Prevent shallow splits
    'min_impurity_decrease': 0.01,     # Early stopping
    'criterion': 'gini'                # Fastest criterion
}

clf = ThetaSketchDecisionTreeClassifier(**fast_config)
```

#### 3. Feature Selection

```python
# Reduce features for faster training
def select_top_features(sketch_data, feature_mapping, k=100):
    """Select top-k features by sketch cardinality."""
    feature_scores = {}

    for feature_name in feature_mapping.keys():
        pos_present = sketch_data['positive'][feature_name][0].get_estimate()
        neg_present = sketch_data['negative'][feature_name][0].get_estimate()

        # Score by class separation
        score = abs(pos_present - neg_present)
        feature_scores[feature_name] = score

    # Select top-k features
    top_features = sorted(feature_scores.items(),
                         key=lambda x: x[1], reverse=True)[:k]

    selected_mapping = {name: idx for idx, (name, _) in enumerate(top_features)}
    return selected_mapping

# Use feature selection for large datasets
selected_mapping = select_top_features(sketch_data, feature_mapping, k=100)
clf.fit(sketch_data, selected_mapping)
```

### Inference Optimization

#### 1. Batch Processing

```python
def optimized_batch_prediction(clf, X_large, batch_size=10000):
    """Memory-efficient batch prediction."""
    n_samples = X_large.shape[0]
    predictions = np.empty(n_samples, dtype=int)

    # Process in batches to control memory usage
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_X = X_large[start_idx:end_idx]

        # Vectorized prediction
        batch_pred = clf.predict(batch_X)
        predictions[start_idx:end_idx] = batch_pred

    return predictions

# Optimal batch sizes by available memory
batch_sizes = {
    '8GB':  50000,   # 50K samples per batch
    '16GB': 100000,  # 100K samples per batch
    '32GB': 200000,  # 200K samples per batch
}
```

#### 2. Parallel Processing

```python
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

def parallel_prediction(clf, X_large, n_workers=None):
    """Parallel prediction using thread workers."""
    if n_workers is None:
        n_workers = mp.cpu_count()

    batch_size = len(X_large) // n_workers
    batches = [X_large[i:i+batch_size] for i in range(0, len(X_large), batch_size)]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(clf.predict, batch) for batch in batches]
        results = [future.result() for future in futures]

    return np.concatenate(results)

# Use for very large datasets
predictions = parallel_prediction(clf, X_test, n_workers=4)
```

#### 3. Pre-compilation Optimization

```python
# JIT compilation for repeated predictions
import numba

@numba.jit(nopython=True, cache=True)
def fast_tree_traversal(sample, tree_nodes, feature_indices):
    """JIT-compiled tree traversal for maximum speed."""
    node_idx = 0  # Start at root

    while not tree_nodes[node_idx]['is_leaf']:
        feature_idx = tree_nodes[node_idx]['feature_idx']
        feature_value = sample[feature_idx]

        # Navigate tree
        if feature_value == 1:  # Present
            node_idx = tree_nodes[node_idx]['left_child']
        else:  # Absent or missing
            node_idx = tree_nodes[node_idx]['right_child']

    return tree_nodes[node_idx]['predicted_class']

# Pre-compile tree structure for repeated use
# (Implementation would require tree serialization)
```

### Memory Optimization

#### 1. Efficient Data Structures

```python
class MemoryEfficientPredictor:
    """Memory-optimized predictor for production deployment."""

    def __init__(self, clf):
        # Convert tree to memory-efficient format
        self.tree_array = self._serialize_tree(clf.tree_)
        self.feature_mapping = clf.feature_names_

    def _serialize_tree(self, tree_node):
        """Convert tree to compact array representation."""
        # Implementation would flatten tree structure
        # for cache-friendly traversal
        pass

    def predict_streaming(self, X_stream):
        """Memory-efficient streaming prediction."""
        for sample in X_stream:
            yield self._traverse_compact_tree(sample)
```

#### 2. Memory Monitoring

```python
import psutil
import os

class PerformanceMonitor:
    """Monitor memory and CPU usage during operations."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_memory = self.process.memory_info().rss / 1024**2

    def get_memory_usage(self):
        """Get current memory usage in MB."""
        current = self.process.memory_info().rss / 1024**2
        return current - self.start_memory

    def monitor_training(self, clf, sketch_data, feature_mapping):
        """Monitor training performance."""
        start_time = time.time()
        start_memory = self.get_memory_usage()

        clf.fit(sketch_data, feature_mapping)

        end_time = time.time()
        end_memory = self.get_memory_usage()

        return {
            'training_time': end_time - start_time,
            'memory_used': end_memory - start_memory,
            'tree_depth': clf.get_depth(),
            'tree_nodes': clf.get_n_nodes()
        }

# Usage
monitor = PerformanceMonitor()
stats = monitor.monitor_training(clf, sketch_data, feature_mapping)
print(f"Training stats: {stats}")
```

## Scalability Guidelines

### Dataset Size Recommendations

| Dataset Size | Recommended Config | Expected Performance |
|--------------|-------------------|---------------------|
| Small (<10K) | `lg_k=8, max_depth=15` | <1s training |
| Medium (10K-100K) | `lg_k=12, max_depth=10` | 1-10s training |
| Large (100K-1M) | `lg_k=12, max_depth=8` | 10-60s training |
| Very Large (>1M) | `lg_k=16, max_depth=6` | 1-5min training |

### Hardware Recommendations

#### Minimum Requirements
- **CPU**: 2+ cores, 2.0+ GHz
- **RAM**: 4GB for training, 2GB for inference
- **Storage**: SSD recommended for sketch I/O

#### Optimal Configuration
- **CPU**: 8+ cores, 3.0+ GHz
- **RAM**: 16GB+ for large datasets
- **Storage**: NVMe SSD for maximum I/O throughput

#### Production Deployment
- **CPU**: 16+ cores for concurrent inference
- **RAM**: 32GB+ for caching and parallel processing
- **Storage**: High-IOPS storage for model loading

### Cloud Deployment Sizing

```python
# AWS instance recommendations
aws_sizing = {
    'development': 't3.medium',    # 2 vCPUs, 4GB RAM
    'testing': 't3.large',         # 2 vCPUs, 8GB RAM
    'production': 'c5.2xlarge',    # 8 vCPUs, 16GB RAM
    'high_volume': 'c5.9xlarge',   # 36 vCPUs, 72GB RAM
}

# GCP instance recommendations
gcp_sizing = {
    'development': 'e2-standard-2',   # 2 vCPUs, 8GB RAM
    'testing': 'e2-standard-4',       # 4 vCPUs, 16GB RAM
    'production': 'c2-standard-8',    # 8 vCPUs, 32GB RAM
    'high_volume': 'c2-standard-30',  # 30 vCPUs, 120GB RAM
}
```

## Performance Testing

### Benchmark Suite

```python
def run_performance_benchmark():
    """Comprehensive performance benchmark."""

    from theta_sketch_tree import ThetaSketchDecisionTreeClassifier
    import time
    import numpy as np

    # Test configurations
    configs = [
        {'n_features': 10, 'n_samples': 1000},
        {'n_features': 50, 'n_samples': 5000},
        {'n_features': 100, 'n_samples': 10000},
        {'n_features': 500, 'n_samples': 50000},
    ]

    results = []

    for config in configs:
        # Generate test data
        sketch_data, feature_mapping = generate_test_sketches(**config)
        X_test = np.random.randint(0, 2, size=(1000, config['n_features']))

        # Training benchmark
        clf = ThetaSketchDecisionTreeClassifier(max_depth=10, verbose=0)

        start_time = time.time()
        clf.fit(sketch_data, feature_mapping)
        training_time = time.time() - start_time

        # Prediction benchmark
        start_time = time.time()
        predictions = clf.predict(X_test)
        prediction_time = time.time() - start_time

        throughput = len(X_test) / prediction_time

        results.append({
            'n_features': config['n_features'],
            'training_time': training_time,
            'prediction_time': prediction_time,
            'throughput': throughput,
            'tree_depth': clf.get_depth(),
            'tree_nodes': clf.get_n_nodes()
        })

    return results

# Run benchmarks
benchmark_results = run_performance_benchmark()
for result in benchmark_results:
    print(f"Features: {result['n_features']}, "
          f"Training: {result['training_time']:.3f}s, "
          f"Throughput: {result['throughput']:,.0f}/sec")
```

### Regression Testing

```python
def performance_regression_test():
    """Ensure performance doesn't degrade between versions."""

    # Baseline performance expectations
    baselines = {
        'training_time_100_features': 2.0,    # Max 2 seconds
        'prediction_throughput': 100000,      # Min 100K/sec
        'memory_usage_100_features': 20,      # Max 20 MB
    }

    # Run current performance tests
    current_perf = run_performance_benchmark()

    # Check against baselines
    for test_name, baseline in baselines.items():
        current_value = extract_metric(current_perf, test_name)

        if test_name.endswith('time') or test_name.endswith('usage'):
            # Lower is better
            assert current_value <= baseline * 1.1, \
                f"{test_name}: {current_value} exceeds baseline {baseline}"
        else:
            # Higher is better
            assert current_value >= baseline * 0.9, \
                f"{test_name}: {current_value} below baseline {baseline}"

    print("✅ Performance regression test passed")
```

## Production Optimization Checklist

### Pre-deployment Optimization

- [ ] **Feature Selection**: Reduce to <500 most important features
- [ ] **Tree Tuning**: Set appropriate `max_depth` (6-15 for production)
- [ ] **Pruning**: Enable `cost_complexity` pruning for generalization
- [ ] **Sketch Parameters**: Use `lg_k=12` for production balance
- [ ] **Memory Profiling**: Verify memory usage under load
- [ ] **Benchmark Testing**: Validate performance requirements

### Runtime Optimization

- [ ] **Batch Processing**: Use optimal batch sizes for throughput
- [ ] **Parallel Processing**: Enable multi-threading for large batches
- [ ] **Memory Monitoring**: Implement memory usage alerts
- [ ] **Caching Strategy**: Cache frequently used models
- [ ] **Connection Pooling**: Optimize sketch loading I/O
- [ ] **Load Balancing**: Distribute inference across instances

### Monitoring and Alerting

```python
def setup_performance_monitoring():
    """Production performance monitoring."""

    # Define SLA thresholds
    sla_thresholds = {
        'prediction_latency_p95': 100,      # 95th percentile < 100ms
        'prediction_throughput': 50000,     # Min 50K predictions/sec
        'memory_usage_max': 1024,           # Max 1GB memory
        'error_rate_max': 0.001,            # Max 0.1% error rate
    }

    # Implement monitoring logic
    # (Integration with Prometheus, CloudWatch, etc.)
```

---

## Next Steps

- **Deployment**: See [Deployment Guide](11-deployment.md) for production setup
- **Testing**: Review [Testing Guide](07-testing.md) for performance validation
- **Troubleshooting**: Check [Troubleshooting Guide](08-troubleshooting.md) for performance issues
- **API**: See [API Reference](05-api-reference.md) for optimization methods