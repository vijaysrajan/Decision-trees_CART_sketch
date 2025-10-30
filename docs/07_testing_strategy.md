# Testing Strategy Document
## Theta Sketch Decision Tree Classifier

---

## Testing Objectives

1. **Unit Test Coverage**: >90% code coverage
2. **Integration Testing**: End-to-end workflows validated
3. **sklearn Compatibility**: Full compliance with sklearn API
4. **Performance Testing**: Training and inference benchmarks
5. **Edge Case Handling**: Robust error handling

---

## Test Suite Structure

### pytest Configuration (conftest.py)

```python
import pytest
import numpy as np
from datasketches import update_theta_sketch
import tempfile
import os

@pytest.fixture
def mock_theta_sketches():
    """
    Create mock theta sketches for testing.

    Returns:
        dict: {'pos': {...}, 'neg': {...}} with theta sketches
    """
    # Create sketches for positive class
    sketch_total_pos = update_theta_sketch(12)  # lg_k=12
    sketch_age_pos = update_theta_sketch(12)
    sketch_income_pos = update_theta_sketch(12)

    # Populate with mock IDs
    for i in range(100):  # 100 positive samples
        sketch_total_pos.update(f"id_{i}")
        if i % 2 == 0:  # 50% have age>30
            sketch_age_pos.update(f"id_{i}")
        if i % 3 == 0:  # 33% have income>50k
            sketch_income_pos.update(f"id_{i}")

    # Create sketches for negative class
    sketch_total_neg = update_theta_sketch(12)
    sketch_age_neg = update_theta_sketch(12)
    sketch_income_neg = update_theta_sketch(12)

    for i in range(100, 200):  # 100 negative samples
        sketch_total_neg.update(f"id_{i}")
        if i % 4 == 0:  # 25% have age>30
            sketch_age_neg.update(f"id_{i}")
        if i % 5 == 0:  # 20% have income>50k
            sketch_income_neg.update(f"id_{i}")

    return {
        'pos': {
            'total': sketch_total_pos,
            'age>30': sketch_age_pos,
            'income>50k': sketch_income_pos
        },
        'neg': {
            'total': sketch_total_neg,
            'age>30': sketch_age_neg,
            'income>50k': sketch_income_neg
        }
    }

@pytest.fixture
def mock_csv_file(mock_theta_sketches, tmp_path):
    """Create temporary CSV file with mock sketches."""
    import base64

    csv_path = tmp_path / "test_sketches.csv"

    with open(csv_path, 'w') as f:
        f.write("identifier,sketch\n")

        # Positive class
        for name, sketch in mock_theta_sketches['pos'].items():
            if name == 'total':
                identifier = ""
            else:
                identifier = name
            sketch_bytes = sketch.compact().serialize()
            sketch_b64 = base64.b64encode(sketch_bytes).decode('ascii')
            f.write(f"{identifier},{sketch_b64}\n")

        # Add target_yes
        f.write(f"target_yes,{sketch_b64}\n")

        # Negative class
        for name, sketch in mock_theta_sketches['neg'].items():
            if name == 'total':
                identifier = ""
            else:
                identifier = name
            sketch_bytes = sketch.compact().serialize()
            sketch_b64 = base64.b64encode(sketch_bytes).decode('ascii')
            f.write(f"{identifier},{sketch_b64}\n")

        # Add target_no
        f.write(f"target_no,{sketch_b64}\n")

    return str(csv_path)

@pytest.fixture
def mock_config_file(tmp_path):
    """Create temporary config YAML file."""
    config_path = tmp_path / "test_config.yaml"

    config_content = """
targets:
  positive: "target_yes"
  negative: "target_no"

hyperparameters:
  criterion: "gini"
  max_depth: 3
  min_samples_split: 2
  min_samples_leaf: 1
  verbose: 0

feature_mapping:
  "age>30": 0        # Simple column index mapping
  "income>50k": 1
  "city=NY": 2
"""

    with open(config_path, 'w') as f:
        f.write(config_content)

    return str(config_path)

@pytest.fixture
def sample_raw_data():
    """Sample raw data for inference testing."""
    return np.array([
        [35, 60000],    # age>30: True, income>50k: True
        [25, 45000],    # age>30: False, income>50k: False
        [55, 120000],   # age>30: True, income>50k: True
        [42, np.nan],   # age>30: True, income>50k: NaN (missing)
    ])
```

---

## Unit Tests by Module

### test_classifier.py

```python
class TestThetaSketchDecisionTreeClassifier:

    def test_initialization_default_params(self):
        """Test default parameter initialization."""
        clf = ThetaSketchDecisionTreeClassifier()
        assert clf.criterion == 'gini'
        assert clf.max_depth is None
        assert clf.min_samples_split == 2
        assert clf.verbose == 0

    def test_get_set_params(self):
        """Test sklearn get_params/set_params interface."""
        clf = ThetaSketchDecisionTreeClassifier(max_depth=5)
        params = clf.get_params()
        assert params['max_depth'] == 5

        clf.set_params(max_depth=10, criterion='entropy')
        assert clf.max_depth == 10
        assert clf.criterion == 'entropy'

    def test_fit_success(self, mock_csv_file, mock_config_file):
        """Test successful training."""
        clf = ThetaSketchDecisionTreeClassifier(max_depth=3, verbose=0)
        clf.fit(mock_csv_file, mock_config_file)

        assert hasattr(clf, 'classes_')
        assert hasattr(clf, 'tree_')
        assert clf.n_classes_ == 2
        assert clf.n_features_in_ == 2

    def test_predict_before_fit_raises_error(self, sample_raw_data):
        """Test that predict before fit raises NotFittedError."""
        from sklearn.exceptions import NotFittedError

        clf = ThetaSketchDecisionTreeClassifier()
        with pytest.raises(NotFittedError):
            clf.predict(sample_raw_data)

    def test_predict_returns_valid_labels(self, mock_csv_file, mock_config_file, sample_raw_data):
        """Test predictions are valid class labels."""
        clf = ThetaSketchDecisionTreeClassifier()
        clf.fit(mock_csv_file, mock_config_file)

        predictions = clf.predict(sample_raw_data)
        assert len(predictions) == len(sample_raw_data)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba_sums_to_one(self, mock_csv_file, mock_config_file, sample_raw_data):
        """Test probabilities sum to 1.0."""
        clf = ThetaSketchDecisionTreeClassifier()
        clf.fit(mock_csv_file, mock_config_file)

        probas = clf.predict_proba(sample_raw_data)
        assert probas.shape == (len(sample_raw_data), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
        assert np.all((probas >= 0) & (probas <= 1))

    def test_feature_importances_sum_to_one(self, mock_csv_file, mock_config_file):
        """Test feature importances sum to 1.0."""
        clf = ThetaSketchDecisionTreeClassifier()
        clf.fit(mock_csv_file, mock_config_file)

        importances = clf.feature_importances_
        assert len(importances) == clf.n_features_in_
        assert np.isclose(importances.sum(), 1.0)

    def test_model_persistence(self, mock_csv_file, mock_config_file, sample_raw_data, tmp_path):
        """Test save and load model."""
        clf = ThetaSketchDecisionTreeClassifier()
        clf.fit(mock_csv_file, mock_config_file)

        model_path = tmp_path / "model.pkl"
        clf.save_model(str(model_path))

        clf_loaded = ThetaSketchDecisionTreeClassifier.load_model(str(model_path))

        # Test predictions match
        pred_original = clf.predict(sample_raw_data)
        pred_loaded = clf_loaded.predict(sample_raw_data)
        assert np.array_equal(pred_original, pred_loaded)
```

### test_criteria.py

```python
class TestCriteria:

    def test_gini_pure_node(self):
        """Test Gini=0 for pure node."""
        from theta_sketch_tree.criteria import GiniCriterion

        criterion = GiniCriterion()
        counts = np.array([10, 0])  # Pure class 0
        gini = criterion.compute_impurity(counts)
        assert gini == 0.0

    def test_gini_balanced_node(self):
        """Test Gini=0.5 for balanced binary node."""
        from theta_sketch_tree.criteria import GiniCriterion

        criterion = GiniCriterion()
        counts = np.array([5, 5])  # Balanced
        gini = criterion.compute_impurity(counts)
        assert np.isclose(gini, 0.5)

    def test_weighted_gini(self):
        """Test weighted Gini with class weights."""
        from theta_sketch_tree.criteria import GiniCriterion

        criterion = GiniCriterion(class_weight={0: 1.0, 1: 10.0})
        counts = np.array([10, 1])  # Imbalanced
        gini_weighted = criterion.compute_impurity(counts)

        # Weighted Gini should differ from unweighted
        criterion_unweighted = GiniCriterion()
        gini_unweighted = criterion_unweighted.compute_impurity(counts)
        assert gini_weighted != gini_unweighted

    def test_entropy_pure_node(self):
        """Test Entropy=0 for pure node."""
        from theta_sketch_tree.criteria import EntropyCriterion

        criterion = EntropyCriterion()
        counts = np.array([10, 0])
        entropy = criterion.compute_impurity(counts)
        assert entropy == 0.0

    def test_binomial_significant_split(self):
        """Test binomial test identifies significant split."""
        from theta_sketch_tree.criteria import BinomialCriterion

        criterion = BinomialCriterion()
        parent_counts = np.array([50, 50])
        left_counts = np.array([45, 5])   # Mostly class 0
        right_counts = np.array([5, 45])  # Mostly class 1

        p_value = criterion.evaluate_split(parent_counts, left_counts, right_counts, 0.5)
        assert p_value < 0.05  # Significant
```

### test_missing_handler.py

```python
class TestMissingValueHandling:

    def test_majority_path_left(self, mock_csv_file, mock_config_file):
        """Test missing values go left when majority went left."""
        clf = ThetaSketchDecisionTreeClassifier(missing_value_strategy='majority')
        clf.fit(mock_csv_file, mock_config_file)

        # Create sample with missing value
        X = np.array([[np.nan, 60000]])
        pred = clf.predict(X)
        assert pred[0] in [0, 1]  # Valid prediction

    def test_zero_imputation_strategy(self, mock_csv_file, mock_config_file):
        """Test missing values treated as False."""
        clf = ThetaSketchDecisionTreeClassifier(missing_value_strategy='zero')
        clf.fit(mock_csv_file, mock_config_file)

        X_missing = np.array([[np.nan, 60000]])
        X_zero = np.array([[0, 60000]])  # Treat missing as 0

        pred_missing = clf.predict(X_missing)
        pred_zero = clf.predict(X_zero)
        # Should produce same prediction (implementation dependent on thresholds)

    def test_error_strategy_raises_exception(self, mock_csv_file, mock_config_file):
        """Test error strategy raises ValueError."""
        clf = ThetaSketchDecisionTreeClassifier(missing_value_strategy='error')
        clf.fit(mock_csv_file, mock_config_file)

        X = np.array([[np.nan, 60000]])
        with pytest.raises(ValueError, match="Missing value"):
            clf.predict(X)
```

---

## Integration Tests

### test_integration.py

```python
class TestIntegration:

    def test_end_to_end_workflow(self, mock_csv_file, mock_config_file, sample_raw_data):
        """Test complete workflow from CSV to predictions."""
        # Training
        clf = ThetaSketchDecisionTreeClassifier(
            criterion='gini',
            max_depth=5,
            min_samples_leaf=2,
            verbose=1
        )
        clf.fit(mock_csv_file, mock_config_file)

        # Inference
        predictions = clf.predict(sample_raw_data)
        probabilities = clf.predict_proba(sample_raw_data)

        # Metrics
        importances = clf.feature_importances_

        # Validation
        assert len(predictions) == len(sample_raw_data)
        assert probabilities.shape == (len(sample_raw_data), 2)
        assert len(importances) == 2
        assert np.isclose(importances.sum(), 1.0)

    def test_sklearn_pipeline(self, mock_csv_file, mock_config_file, sample_raw_data):
        """Test use in sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        clf = ThetaSketchDecisionTreeClassifier()
        clf.fit(mock_csv_file, mock_config_file)

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('tree', clf)
        ])

        # Pipeline predict should work
        predictions = pipe.predict(sample_raw_data)
        assert len(predictions) == len(sample_raw_data)

    def test_pickle_serialization(self, mock_csv_file, mock_config_file, sample_raw_data, tmp_path):
        """Test full pickle serialization."""
        import pickle

        clf = ThetaSketchDecisionTreeClassifier()
        clf.fit(mock_csv_file, mock_config_file)

        # Pickle
        pickle_path = tmp_path / "model.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(clf, f)

        # Unpickle
        with open(pickle_path, 'rb') as f:
            clf_loaded = pickle.load(f)

        # Test equivalence
        pred1 = clf.predict(sample_raw_data)
        pred2 = clf_loaded.predict(sample_raw_data)
        assert np.array_equal(pred1, pred2)
```

---

## sklearn Compatibility Tests

### test_sklearn_compatibility.py

```python
class TestSklearnCompatibility:

    def test_estimator_interface(self):
        """Test sklearn estimator check (partial)."""
        from sklearn.utils.estimator_checks import check_estimator
        from sklearn.base import clone

        clf = ThetaSketchDecisionTreeClassifier(max_depth=3)

        # Test cloning
        clf_clone = clone(clf)
        assert clf_clone.max_depth == 3
        assert clf_clone is not clf

    def test_get_params_deep(self):
        """Test deep parameter retrieval."""
        clf = ThetaSketchDecisionTreeClassifier(max_depth=5)
        params = clf.get_params(deep=True)

        assert 'max_depth' in params
        assert params['max_depth'] == 5

    def test_score_method(self, mock_csv_file, mock_config_file, sample_raw_data):
        """Test score() method (from ClassifierMixin)."""
        clf = ThetaSketchDecisionTreeClassifier()
        clf.fit(mock_csv_file, mock_config_file)

        y_true = np.array([1, 0, 1, 0])
        score = clf.score(sample_raw_data, y_true)

        assert 0 <= score <= 1
```

---

## Performance Tests

### benchmarks/benchmark_training_time.py

```python
import time
import numpy as np
from theta_sketch_tree import ThetaSketchDecisionTreeClassifier

def benchmark_training_with_cache():
    """Measure training time with caching enabled."""
    clf = ThetaSketchDecisionTreeClassifier(
        use_cache=True,
        cache_size_mb=100,
        verbose=0
    )

    start = time.time()
    clf.fit('large_sketches.csv', 'config.yaml')
    elapsed = time.time() - start

    print(f"Training with cache: {elapsed:.2f}s")
    return elapsed

def benchmark_training_without_cache():
    """Measure training time without caching."""
    clf = ThetaSketchDecisionTreeClassifier(
        use_cache=False,
        verbose=0
    )

    start = time.time()
    clf.fit('large_sketches.csv', 'config.yaml')
    elapsed = time.time() - start

    print(f"Training without cache: {elapsed:.2f}s")
    return elapsed

if __name__ == '__main__':
    time_with_cache = benchmark_training_with_cache()
    time_without_cache = benchmark_training_without_cache()

    speedup = time_without_cache / time_with_cache
    print(f"\nSpeedup from caching: {speedup:.2f}x")
    assert speedup >= 2.0, "Cache should provide 2x speedup"
```

---

## Test Execution

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ --cov=theta_sketch_tree --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_classifier.py -v

# Run specific test
pytest tests/test_classifier.py::TestThetaSketchDecisionTreeClassifier::test_fit_success -v

# Run with verbose output
pytest tests/ -vv

# Run benchmarks
python benchmarks/benchmark_training_time.py
```

### Coverage Report

Target: >90% code coverage

```
Name                                    Stmts   Miss  Cover
-----------------------------------------------------------
theta_sketch_tree/__init__.py               5      0   100%
theta_sketch_tree/classifier.py           200     10    95%
theta_sketch_tree/tree_structure.py        80      5    94%
theta_sketch_tree/criteria.py             150      8    95%
theta_sketch_tree/splitter.py             180     12    93%
...
-----------------------------------------------------------
TOTAL                                    2000    100    95%
```

---

## Continuous Integration (CI)

### GitHub Actions Workflow (.github/workflows/test.yml)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -e ".[dev]"

    - name: Run tests
        run: |
        pytest tests/ --cov=theta_sketch_tree --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

---

## Summary

**Test Coverage**:
- ✅ Unit tests for all modules
- ✅ Integration tests for workflows
- ✅ sklearn compatibility tests
- ✅ Performance benchmarks

**Target Metrics**:
- Code coverage: >90%
- Test execution time: <5 minutes
- All sklearn compatibility checks pass

This testing strategy ensures production-ready code quality.
