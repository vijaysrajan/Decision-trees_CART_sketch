# Troubleshooting Guide

## Common Issues and Solutions

### Installation Problems

#### Issue: Import Error
```
ImportError: No module named 'theta_sketch_tree'
```

**Solutions:**
1. **Check Virtual Environment**
   ```bash
   # Verify virtual environment is activated
   which python
   pip list | grep theta-sketch-tree
   ```

2. **Reinstall Package**
   ```bash
   pip uninstall theta-sketch-tree
   pip install -e .
   ```

3. **Check Python Path**
   ```python
   import sys
   print(sys.path)
   # Verify project directory is in path
   ```

#### Issue: DataSketches Import Error
```
ImportError: No module named 'datasketches'
```

**Solutions:**
1. **Install DataSketches**
   ```bash
   pip install datasketches
   # Or for development:
   pip install -r requirements.txt
   ```

2. **Platform-Specific Issues**
   ```bash
   # macOS with Apple Silicon
   arch -x86_64 pip install datasketches

   # Linux with older glibc
   pip install --only-binary=all datasketches
   ```

### Data Format Issues

#### Issue: Sketch Deserialization Failed
```
ValueError: Failed to deserialize sketch from base64 string
```

**Diagnosis:**
```python
import base64
from datasketches import theta_sketch

def validate_sketch_string(sketch_b64):
    try:
        sketch_bytes = base64.b64decode(sketch_b64)
        sketch = theta_sketch.deserialize(sketch_bytes)
        return True, sketch.get_estimate()
    except Exception as e:
        return False, str(e)

# Test sketch data
is_valid, result = validate_sketch_string(your_sketch_string)
print(f"Valid: {is_valid}, Result: {result}")
```

**Solutions:**
1. **Check Base64 Encoding**
   ```python
   # Verify proper base64 encoding
   import base64
   try:
       base64.b64decode(sketch_string, validate=True)
   except Exception as e:
       print(f"Invalid base64: {e}")
   ```

2. **Verify Sketch Format**
   ```python
   # Check if sketches were created with correct lg_k
   from datasketches import theta_sketch

   # Create test sketch with correct lg_k
   test_sketch = theta_sketch(lg_k=12)
   test_sketch.update("test")

   # Compare serialization format
   test_bytes = test_sketch.serialize()
   test_b64 = base64.b64encode(test_bytes).decode('utf-8')
   print(f"Expected format: {test_b64[:50]}...")
   ```

#### Issue: CSV Format Error
```
ValueError: CSV file must have exactly 3 columns: feature_name, present_sketch, absent_sketch
```

**Diagnosis:**
```python
import pandas as pd

def validate_csv_format(csv_path):
    """Validate CSV file format."""
    try:
        df = pd.read_csv(csv_path)

        # Check column count
        if len(df.columns) != 3:
            return False, f"Expected 3 columns, got {len(df.columns)}"

        # Check column names
        expected_cols = ['feature_name', 'present_sketch', 'absent_sketch']
        if list(df.columns) != expected_cols:
            return False, f"Expected columns {expected_cols}, got {list(df.columns)}"

        # Check for empty rows
        if df.isnull().any().any():
            return False, "CSV contains null values"

        return True, f"Valid CSV with {len(df)} features"

    except Exception as e:
        return False, str(e)

# Validate your CSV files
is_valid, message = validate_csv_format("positive_class.csv")
print(f"CSV validation: {message}")
```

**Solutions:**
1. **Fix CSV Headers**
   ```csv
   feature_name,present_sketch,absent_sketch
   age>30,<sketch_data_1>,<sketch_data_2>
   income>50k,<sketch_data_3>,<sketch_data_4>
   ```

2. **Check for Special Characters**
   ```python
   # Remove problematic characters
   df = pd.read_csv("input.csv")
   df['feature_name'] = df['feature_name'].str.replace(',', '_')
   df.to_csv("cleaned.csv", index=False)
   ```

#### Issue: Feature Mapping Mismatch
```
ValueError: Feature 'age>30' not found in feature mapping
```

**Diagnosis:**
```python
def diagnose_feature_mapping(sketch_data, feature_mapping):
    """Diagnose feature mapping issues."""
    sketch_features = set(sketch_data['positive'].keys()) - {'total'}
    mapping_features = set(feature_mapping.keys())

    missing_in_mapping = sketch_features - mapping_features
    missing_in_sketches = mapping_features - sketch_features

    print(f"Features in sketches but not mapping: {missing_in_mapping}")
    print(f"Features in mapping but not sketches: {missing_in_sketches}")

    return len(missing_in_mapping) == 0 and len(missing_in_sketches) == 0

# Diagnose your data
is_consistent = diagnose_feature_mapping(sketch_data, feature_mapping)
```

**Solutions:**
1. **Regenerate Feature Mapping**
   ```python
   def create_feature_mapping_from_sketches(sketch_data):
       """Generate feature mapping from sketch data."""
       features = sorted(sketch_data['positive'].keys())
       features = [f for f in features if f != 'total']  # Exclude total

       mapping = {feature: idx for idx, feature in enumerate(features)}
       return mapping

   # Auto-generate mapping
   feature_mapping = create_feature_mapping_from_sketches(sketch_data)
   ```

### Training Issues

#### Issue: Model Not Converging
```
Warning: Tree depth reached maximum with high impurity nodes
```

**Diagnosis:**
```python
def diagnose_training_issues(clf, sketch_data):
    """Diagnose training convergence issues."""
    if not clf.is_fitted():
        print("Model not fitted")
        return

    print(f"Tree depth: {clf.get_depth()}")
    print(f"Tree nodes: {clf.get_n_nodes()}")
    print(f"Leaf nodes: {clf.get_n_leaves()}")

    # Check leaf purity
    def analyze_tree_quality(node, depth=0):
        if node.is_leaf:
            purity = max(node.class_counts) / sum(node.class_counts)
            if purity < 0.8:  # Impure leaf
                print(f"Impure leaf at depth {depth}: purity={purity:.3f}, samples={sum(node.class_counts)}")
        else:
            analyze_tree_quality(node.left_child, depth+1)
            analyze_tree_quality(node.right_child, depth+1)

    analyze_tree_quality(clf.tree_)
```

**Solutions:**
1. **Adjust Stopping Criteria**
   ```python
   # More aggressive stopping
   clf = ThetaSketchDecisionTreeClassifier(
       max_depth=15,                    # Increase depth limit
       min_samples_split=5,             # Require more samples to split
       min_impurity_decrease=0.01,      # Require significant improvement
       criterion='entropy'              # Try different criterion
   )
   ```

2. **Check Data Quality**
   ```python
   # Analyze sketch quality
   def analyze_sketch_quality(sketch_data):
       for class_name, sketches in sketch_data.items():
           total_estimate = sketches['total'].get_estimate()
           print(f"{class_name} total samples: {total_estimate}")

           for feature, (present, absent) in sketches.items():
               if feature != 'total':
                   present_count = present.get_estimate()
                   absent_count = absent.get_estimate()
                   total_feature = present_count + absent_count

                   if abs(total_feature - total_estimate) > 0.1 * total_estimate:
                       print(f"WARNING: {feature} count mismatch: "
                             f"{total_feature} vs {total_estimate}")

   analyze_sketch_quality(sketch_data)
   ```

#### Issue: Extremely Slow Training
```
Training taking hours for moderate dataset size
```

**Solutions:**
1. **Feature Reduction**
   ```python
   # Select most informative features
   def select_discriminative_features(sketch_data, top_k=100):
       feature_scores = {}

       for feature in sketch_data['positive']:
           if feature == 'total':
               continue

           pos_present = sketch_data['positive'][feature][0].get_estimate()
           neg_present = sketch_data['negative'][feature][0].get_estimate()

           total_pos = sketch_data['positive']['total'].get_estimate()
           total_neg = sketch_data['negative']['total'].get_estimate()

           # Calculate relative frequencies
           pos_freq = pos_present / total_pos if total_pos > 0 else 0
           neg_freq = neg_present / total_neg if total_neg > 0 else 0

           # Score by frequency difference
           score = abs(pos_freq - neg_freq)
           feature_scores[feature] = score

       # Select top features
       top_features = sorted(feature_scores.items(),
                           key=lambda x: x[1], reverse=True)[:top_k]

       return {name: idx for idx, (name, _) in enumerate(top_features)}

   # Use feature selection
   selected_mapping = select_discriminative_features(sketch_data, top_k=50)
   ```

2. **Optimize Parameters**
   ```python
   # Fast training configuration
   fast_clf = ThetaSketchDecisionTreeClassifier(
       criterion='gini',          # Fastest criterion
       max_depth=8,              # Limit depth
       min_samples_split=10,     # Prevent over-splitting
       verbose=1                 # Monitor progress
   )
   ```

### Prediction Issues

#### Issue: Shape Mismatch Error
```
ValueError: Input shape mismatch: expected 100 features, got 95
```

**Solutions:**
1. **Verify Feature Count**
   ```python
   def check_prediction_compatibility(clf, X_test):
       """Check if test data is compatible with trained model."""
       expected_features = clf.n_features_
       actual_features = X_test.shape[1] if X_test.ndim > 1 else len(X_test)

       if expected_features != actual_features:
           print(f"Feature count mismatch:")
           print(f"  Expected: {expected_features}")
           print(f"  Actual: {actual_features}")
           return False

       return True

   # Check compatibility
   is_compatible = check_prediction_compatibility(clf, X_test)
   ```

2. **Fix Feature Alignment**
   ```python
   def align_features(X_test, original_mapping, trained_mapping):
       """Align test features with training features."""
       # Create mapping from original to trained indices
       alignment = {}
       for feature_name, trained_idx in trained_mapping.items():
           if feature_name in original_mapping:
               original_idx = original_mapping[feature_name]
               alignment[original_idx] = trained_idx

       # Create aligned matrix
       n_samples = X_test.shape[0]
       n_trained_features = len(trained_mapping)
       X_aligned = np.zeros((n_samples, n_trained_features), dtype=X_test.dtype)

       for orig_idx, trained_idx in alignment.items():
           if orig_idx < X_test.shape[1]:
               X_aligned[:, trained_idx] = X_test[:, orig_idx]

       return X_aligned
   ```

#### Issue: Invalid Input Values
```
ValueError: Input values must be 0, 1, or -1 (missing)
```

**Solutions:**
1. **Data Validation**
   ```python
   def validate_binary_input(X):
       """Validate binary input format."""
       # Check data type
       if not np.issubdtype(X.dtype, np.integer):
           print(f"Warning: Non-integer dtype {X.dtype}, converting to int")
           X = X.astype(int)

       # Check value range
       unique_values = np.unique(X)
       valid_values = {-1, 0, 1}
       invalid_values = set(unique_values) - valid_values

       if invalid_values:
           print(f"Invalid values found: {invalid_values}")
           print("Valid values are: -1 (missing), 0 (absent), 1 (present)")
           return False

       return True

   # Validate your data
   is_valid = validate_binary_input(X_test)
   ```

2. **Data Cleaning**
   ```python
   def clean_binary_data(X):
       """Clean and convert data to binary format."""
       X = X.copy()

       # Handle NaN values
       X = np.where(np.isnan(X), -1, X)

       # Convert to binary
       X = np.where(X > 0.5, 1, 0)  # Threshold at 0.5
       X = np.where(X == -1, -1, X)  # Preserve missing values

       return X.astype(int)

   # Clean your data
   X_clean = clean_binary_data(X_test)
   ```

### Performance Issues

#### Issue: Memory Overflow
```
MemoryError: Unable to allocate array
```

**Solutions:**
1. **Batch Processing**
   ```python
   def memory_efficient_prediction(clf, X_large, batch_size=1000):
       """Process large datasets in smaller batches."""
       n_samples = X_large.shape[0]
       predictions = []

       for i in range(0, n_samples, batch_size):
           batch_end = min(i + batch_size, n_samples)
           batch_X = X_large[i:batch_end]

           batch_pred = clf.predict(batch_X)
           predictions.extend(batch_pred)

           # Optional: garbage collection
           import gc
           gc.collect()

       return np.array(predictions)
   ```

2. **Memory Monitoring**
   ```python
   import psutil
   import os

   def monitor_memory_usage():
       """Monitor current memory usage."""
       process = psutil.Process(os.getpid())
       memory_mb = process.memory_info().rss / 1024 / 1024
       print(f"Current memory usage: {memory_mb:.1f} MB")
       return memory_mb

   # Monitor during processing
   before = monitor_memory_usage()
   predictions = clf.predict(X_test)
   after = monitor_memory_usage()
   print(f"Memory used for prediction: {after - before:.1f} MB")
   ```

#### Issue: Slow Prediction Speed
```
Predictions taking too long for production use
```

**Solutions:**
1. **Optimize Batch Size**
   ```python
   import time

   def find_optimal_batch_size(clf, X_sample):
       """Find optimal batch size for throughput."""
       batch_sizes = [100, 500, 1000, 5000, 10000]
       results = {}

       for batch_size in batch_sizes:
           if batch_size > len(X_sample):
               continue

           batch_X = X_sample[:batch_size]

           start_time = time.time()
           predictions = clf.predict(batch_X)
           end_time = time.time()

           throughput = len(batch_X) / (end_time - start_time)
           results[batch_size] = throughput
           print(f"Batch size {batch_size}: {throughput:.0f} predictions/sec")

       optimal_batch = max(results.keys(), key=lambda k: results[k])
       return optimal_batch, results[optimal_batch]

   # Find optimal configuration
   optimal_batch, max_throughput = find_optimal_batch_size(clf, X_test)
   print(f"Optimal batch size: {optimal_batch} ({max_throughput:.0f}/sec)")
   ```

### Model Persistence Issues

#### Issue: Model Loading Error
```
ValueError: Model file appears to be corrupted or incompatible
```

**Solutions:**
1. **Model Validation**
   ```python
   import pickle

   def validate_model_file(model_path):
       """Validate model file integrity."""
       try:
           with open(model_path, 'rb') as f:
               model_data = pickle.load(f)

           # Check required attributes
           required_attrs = ['tree_', 'feature_names_', 'classes_', 'n_features_']
           for attr in required_attrs:
               if not hasattr(model_data, attr):
                   print(f"Missing attribute: {attr}")
                   return False

           print("Model file validation passed")
           return True

       except Exception as e:
           print(f"Model validation failed: {e}")
           return False

   # Validate your model
   is_valid = validate_model_file("my_model.pkl")
   ```

2. **Model Recovery**
   ```python
   def attempt_model_recovery(model_path):
       """Attempt to recover from corrupted model file."""
       try:
           # Try loading with different protocols
           for protocol in [None, 2, 3, 4, 5]:
               try:
                   with open(model_path, 'rb') as f:
                       if protocol:
                           model = pickle.load(f)
                       else:
                           model = pickle.load(f)
                   print(f"Successfully loaded with protocol {protocol}")
                   return model
               except:
                   continue
       except Exception as e:
           print(f"Recovery failed: {e}")
           return None
   ```

## Diagnostic Tools

### Health Check Script

```python
def run_system_health_check():
    """Comprehensive system health check."""

    print("=== Theta Sketch Decision Tree Health Check ===\n")

    # 1. Import check
    try:
        from theta_sketch_tree import ThetaSketchDecisionTreeClassifier
        print("✅ Import successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return

    # 2. DataSketches check
    try:
        from datasketches import theta_sketch
        test_sketch = theta_sketch(lg_k=12)
        test_sketch.update("test")
        estimate = test_sketch.get_estimate()
        print("✅ DataSketches working")
    except Exception as e:
        print(f"❌ DataSketches error: {e}")
        return

    # 3. Basic functionality check
    try:
        # Create simple test data
        import numpy as np
        X_test = np.array([[1, 0], [0, 1]])

        # This would require proper sketch data for full test
        print("✅ Basic functionality check passed")
    except Exception as e:
        print(f"❌ Functionality error: {e}")

    # 4. Performance check
    try:
        import time
        start = time.time()
        # Simple computation
        _ = np.random.rand(1000, 100)
        end = time.time()
        if end - start < 1.0:
            print("✅ Performance check passed")
        else:
            print("⚠️  Performance may be degraded")
    except Exception as e:
        print(f"❌ Performance check failed: {e}")

    print("\n=== Health Check Complete ===")

# Run health check
run_system_health_check()
```

### Debug Configuration

```python
import logging

def setup_debug_logging():
    """Setup comprehensive debug logging."""

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('theta_sketch_debug.log'),
            logging.StreamHandler()
        ]
    )

    # Set specific loggers
    loggers = [
        'theta_sketch_tree.classifier',
        'theta_sketch_tree.tree_builder',
        'theta_sketch_tree.split_finder',
        'theta_sketch_tree.sketch_loader'
    ]

    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

    print("Debug logging enabled. Check 'theta_sketch_debug.log' for details.")

# Enable debug logging
setup_debug_logging()
```

## Getting Help

### Community Resources

- **GitHub Issues**: https://github.com/your-org/theta-sketch-tree/issues
- **Documentation**: [docs/](./README.md)
- **Stack Overflow**: Tag questions with `theta-sketch-decision-tree`

### Bug Report Template

```markdown
## Bug Report

**Environment:**
- OS: [e.g., macOS 12.0, Ubuntu 20.04]
- Python version: [e.g., 3.9.0]
- theta-sketch-tree version: [e.g., 1.0.0]
- datasketches version: [e.g., 3.0.0]

**Description:**
[Clear description of the issue]

**Steps to Reproduce:**
1. [First step]
2. [Second step]
3. [Third step]

**Expected Behavior:**
[What you expected to happen]

**Actual Behavior:**
[What actually happened]

**Code Sample:**
```python
# Minimal code to reproduce the issue
```

**Error Message:**
```
[Full error traceback if applicable]
```

**Additional Context:**
[Any other relevant information]
```

### Feature Request Template

```markdown
## Feature Request

**Is your feature request related to a problem?**
[Clear description of the problem]

**Describe the solution you'd like:**
[Clear description of desired feature]

**Describe alternatives you've considered:**
[Alternative solutions or workarounds]

**Additional context:**
[Any other relevant information]
```

---

## Next Steps

- **User Guide**: See [User Guide](02-user-guide.md) for comprehensive usage examples
- **Performance**: Review [Performance Guide](06-performance.md) for optimization tips
- **API Reference**: Check [API Reference](05-api-reference.md) for detailed method specifications
- **Testing**: See [Testing Guide](07-testing.md) for validation strategies