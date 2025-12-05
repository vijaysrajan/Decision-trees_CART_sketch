# Testing Guide

## Overview

The Theta Sketch Decision Tree implements a comprehensive testing strategy with **420 tests achieving 97% coverage**. Testing is split into unit tests (focused coverage) and integration tests (real-world validation).

## Test Architecture

### Unit Tests (`./test-unit.sh`)
- **Focus**: Individual component functionality
- **Coverage Target**: >90% per module
- **Execution**: Fast (<30 seconds)
- **Purpose**: Development workflow validation

### Integration Tests (`./test-integration.sh`)
- **Focus**: End-to-end workflows with real data
- **Coverage**: Not measured (performance-focused)
- **Execution**: Moderate (1-2 minutes)
- **Purpose**: Production readiness validation

### Performance Tests (`./test-performance.sh`)
- **Focus**: Scalability and regression testing
- **Metrics**: Training time, prediction throughput
- **Execution**: Extended (5-10 minutes)
- **Purpose**: Performance validation

## Running Tests

### Quick Validation
```bash
# Unit tests only (development workflow)
./test-unit.sh

# Integration tests only (feature validation)
./test-integration.sh

# All tests with coverage
./test-all.sh
```

### Specific Test Execution
```bash
# Individual test files
./venv/bin/pytest tests/test_classifier.py -v
./venv/bin/pytest tests/test_criteria.py -v
./venv/bin/pytest tests/test_integration.py -v

# Specific test methods
./venv/bin/pytest tests/test_classifier.py::TestThetaSketchDecisionTreeClassifier::test_fit_basic -v
./venv/bin/pytest tests/test_integration.py::TestIntegration::test_full_pipeline_with_mock_sketches -v
```

### Coverage Analysis
```bash
# Generate coverage report
./venv/bin/pytest tests/ --cov=theta_sketch_tree --cov-report=html --cov-report=term-missing

# View HTML coverage report
open htmlcov/index.html
```

## Test Structure

### Core Module Tests

**`tests/test_classifier.py`** - Main classifier functionality
- Initialization and parameter validation
- Training workflow (fit method)
- Prediction methods (predict, predict_proba, decision_function)
- Feature importance calculation
- Edge case handling

**`tests/test_criteria.py`** - Split criteria validation
- Mathematical correctness for all criteria
- Edge case handling (zero counts, pure nodes)
- Performance characteristics

**`tests/test_tree_builder.py`** - Tree construction algorithm
- Recursive building logic
- Stopping criteria enforcement
- Node metadata management

**`tests/test_tree_traverser.py`** - Inference logic
- Binary input processing
- Missing value handling
- Vectorized prediction

### Integration Tests

**`tests/test_integration.py`** - End-to-end workflows
- Complete training and prediction pipeline
- Multiple criteria validation
- Model persistence roundtrip

**`tests/test_binary_classification_sketches.py`** - Real mushroom dataset
- Sketch loading and validation
- Tree structure comparison
- Performance regression testing

### Specialized Tests

**`tests/test_enhanced_mushroom_validation.py`** - Advanced validation
- Recursive tree structure comparison
- Mathematical correctness validation
- Production scenario testing

## Test Data

### Mock Sketches (`tests/test_mock_sketches.py`)
- Lightweight sketch simulation
- Deterministic behavior for unit tests
- Fast execution without real sketch overhead

### Real Data Fixtures
- **Mushroom Dataset**: `tests/fixtures/mushroom_*_lg_k_11.csv`
- **Feature Mappings**: `tests/fixtures/mushroom_feature_mapping.json`
- **Baseline Results**: `tests/integration/mushroom/baselines/`

## Writing Tests

### Unit Test Template
```python
import pytest
import numpy as np
from theta_sketch_tree import ThetaSketchDecisionTreeClassifier

class TestNewFeature:
    """Test suite for new feature functionality."""

    @pytest.fixture
    def sample_data(self):
        """Fixture providing test data."""
        # Setup test data
        return test_data

    def test_basic_functionality(self, sample_data):
        """Test basic feature operation."""
        # Arrange
        clf = ThetaSketchDecisionTreeClassifier()

        # Act
        result = clf.some_method(sample_data)

        # Assert
        assert result is not None
        assert len(result) == expected_length

    def test_edge_case_handling(self):
        """Test edge case behavior."""
        clf = ThetaSketchDecisionTreeClassifier()

        with pytest.raises(ValueError, match="Expected error message"):
            clf.some_method(invalid_input)

    @pytest.mark.parametrize("param,expected", [
        ("value1", "result1"),
        ("value2", "result2"),
    ])
    def test_parameterized_behavior(self, param, expected):
        """Test behavior across parameter values."""
        clf = ThetaSketchDecisionTreeClassifier()
        result = clf.some_method(param)
        assert result == expected
```

### Integration Test Template
```python
class TestNewIntegration:
    """Integration test for new workflow."""

    def test_end_to_end_workflow(self, mushroom_data):
        """Test complete workflow with real data."""
        df, sketches, feature_mapping = mushroom_data

        # Train classifier
        clf = ThetaSketchDecisionTreeClassifier(criterion='gini')
        clf.fit(sketches, feature_mapping)

        # Create test data
        X_test = self._convert_to_binary_matrix(df.head(100), feature_mapping)

        # Make predictions
        predictions = clf.predict(X_test)
        probabilities = clf.predict_proba(X_test)

        # Validate results
        assert len(predictions) == 100
        assert probabilities.shape == (100, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
```

## Test Configuration

### pytest Configuration (`pytest.ini`)
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    performance: marks tests as performance tests
addopts =
    --strict-markers
    --strict-config
    --tb=short
    -ra
```

### Test Environment Setup
```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate
pip install -r requirements.txt
pip install pytest pytest-cov pytest-mock

# Run tests with proper environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
./test-all.sh
```

## Continuous Integration

### GitHub Actions Workflow
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run unit tests
      run: ./test-unit.sh

    - name: Run integration tests
      run: ./test-integration.sh

    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
```

## Test Coverage Goals

### Module-Level Coverage Targets
- **`classifier.py`**: >95% (core functionality)
- **`tree_builder.py`**: >90% (complex algorithm)
- **`tree_traverser.py`**: >95% (inference critical)
- **`criteria.py`**: >90% (mathematical correctness)
- **`sketch_loader.py`**: >85% (I/O handling)
- **`validation_utils.py`**: >90% (input validation)

### Coverage Monitoring
```bash
# Generate coverage report
./test-all.sh

# Check coverage thresholds
./venv/bin/pytest --cov=theta_sketch_tree --cov-fail-under=90

# Identify coverage gaps
./venv/bin/pytest --cov=theta_sketch_tree --cov-report=term-missing | grep "TOTAL"
```

## Performance Testing

### Benchmark Tests
```python
def test_training_performance():
    """Validate training performance meets benchmarks."""
    import time

    # Setup test data
    sketch_data, feature_mapping = create_test_data(n_features=100)

    # Measure training time
    clf = ThetaSketchDecisionTreeClassifier(max_depth=10)

    start_time = time.time()
    clf.fit(sketch_data, feature_mapping)
    training_time = time.time() - start_time

    # Validate performance
    assert training_time < 5.0, f"Training took {training_time:.2f}s, expected <5s"

def test_prediction_throughput():
    """Validate prediction throughput meets requirements."""
    clf = trained_classifier()  # Assume pre-trained
    X_test = np.random.randint(0, 2, size=(10000, 100))

    start_time = time.time()
    predictions = clf.predict(X_test)
    prediction_time = time.time() - start_time

    throughput = len(X_test) / prediction_time
    assert throughput > 100000, f"Throughput {throughput:.0f}/sec, expected >100K/sec"
```

## Test Maintenance

### Regular Test Review
- **Weekly**: Review failing tests and coverage reports
- **Monthly**: Update test data and baselines
- **Quarterly**: Performance regression analysis
- **Annually**: Test architecture review

### Test Quality Checklist
- [ ] Tests are independent and isolated
- [ ] Test names clearly describe functionality
- [ ] Edge cases and error conditions covered
- [ ] Performance benchmarks included
- [ ] Integration tests validate real workflows
- [ ] Test data is realistic and comprehensive

## Debugging Tests

### Common Issues
```python
# Debug failing tests
./venv/bin/pytest tests/test_classifier.py::test_failing -v -s --tb=long

# Run with debugger
./venv/bin/pytest tests/test_classifier.py::test_failing --pdb

# Disable test capture for debugging
./venv/bin/pytest tests/test_classifier.py::test_failing -s
```

### Test Data Validation
```python
def validate_test_data():
    """Validate test data integrity."""
    # Check sketch data consistency
    sketch_data = load_test_sketches()

    for class_name, sketches in sketch_data.items():
        total_estimate = sketches['total'].get_estimate()

        for feature, (present, absent) in sketches.items():
            if feature != 'total':
                feature_total = present.get_estimate() + absent.get_estimate()

                assert abs(feature_total - total_estimate) < 0.1 * total_estimate, \
                    f"Inconsistent counts for {class_name}/{feature}"
```

---

## Next Steps

- **User Guide**: See [User Guide](02-user-guide.md) for API usage examples
- **Performance**: Review [Performance Guide](06-performance.md) for optimization
- **Troubleshooting**: Check [Troubleshooting Guide](08-troubleshooting.md) for common issues
- **Architecture**: Read [Architecture Overview](03-architecture.md) for system design