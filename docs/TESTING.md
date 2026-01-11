# Testing Strategy

This project uses a dual testing approach with different coverage requirements for unit tests vs integration tests.

## Test Types

### Unit Tests
- **Purpose**: Test individual components and methods in isolation
- **Coverage Requirement**: 90% minimum
- **Examples**: `test_tree_traverser.py`, `test_criteria.py`, `test_feature_importance.py`

### Integration Tests
- **Purpose**: Test end-to-end workflows and real-world scenarios
- **Coverage Requirement**: None (capture data but no fail threshold)
- **Examples**: `test_mushroom_regression.py`, `test_integration.py`, `test_binary_classification_sketches.py`

## Running Tests

### All Tests (No Coverage Requirements)
```bash
# Run everything with coverage capture but no pass/fail requirements
./test-all.sh

# Or manually:
pytest
```

### Unit Tests Only (With Coverage Requirements)
```bash
# Run unit tests with 90% coverage requirement
./test-unit.sh

# Or manually:
pytest -c pytest-unit.ini
```

### Integration Tests Only (No Coverage Requirements)
```bash
# Run integration tests with coverage capture but no requirements
./test-integration.sh

# Or manually:
pytest -c pytest-integration.ini
```

### Specific Test Files
```bash
# Run specific unit test with coverage requirement
./test-unit.sh tests/test_tree_traverser.py

# Run specific integration test without coverage requirement
./test-integration.sh tests/test_mushroom_regression.py
```

## Test Markers

Tests are marked using pytest markers:

```python
# Mark entire file as integration test
pytestmark = pytest.mark.integration

# Mark individual test as integration
@pytest.mark.integration
def test_end_to_end_workflow():
    pass
```

Available markers:
- `integration`: Marks integration/end-to-end tests (excluded from unit test coverage)
- `slow`: Marks slow tests (can skip with `-m "not slow"`)
- `sklearn`: Marks sklearn compatibility tests

## Configuration Files

- `pytest.ini`: Default configuration (all tests, no coverage fail)
- `pytest-unit.ini`: Unit test configuration (90% coverage requirement, excludes integration)
- `pytest-integration.ini`: Integration test configuration (includes only integration tests, no coverage fail)

## Why This Approach?

**Unit Tests with Coverage Requirements:**
- Ensure comprehensive testing of individual components
- Catch edge cases and error conditions
- Maintain high code quality standards

**Integration Tests without Coverage Requirements:**
- Focus on end-to-end functionality validation
- Demonstrate real-world usage patterns
- Serve as regression tests for overall system behavior
- Avoid false failures due to coverage thresholds

## Adding New Tests

### Unit Test
```python
"""Test individual component functionality."""
import pytest
from theta_sketch_tree.component import Component

class TestComponent:
    def test_method_behavior(self):
        # Test specific method functionality
        pass
```

### Integration Test
```python
"""Test end-to-end workflow."""
import pytest

# Mark as integration test
pytestmark = pytest.mark.integration

class TestEndToEnd:
    def test_full_pipeline(self):
        # Test complete workflow
        pass
```

## Coverage Analysis

- **Unit Test Coverage**: Available via `./test-unit.sh` - enforced at 90%
- **Integration Test Coverage**: Available via `./test-integration.sh` - informational only
- **Combined Coverage**: Available via `./test-all.sh` - informational only

Use `htmlcov/index.html` to view detailed coverage reports after running tests.