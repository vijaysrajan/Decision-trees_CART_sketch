"""
Pytest configuration and fixtures.

This module provides reusable test fixtures for the test suite.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# TODO: Install datasketches library first
# from datasketches import update_theta_sketch


@pytest.fixture
def sample_binary_data():
    """
    Binary feature data for inference testing.

    Features (all binary 0/1):
      - Column 0: age>30
      - Column 1: city=NewYork
      - Column 2: gender=M
      - Column 3: income>50k
    """
    return np.array([
        [1, 0, 1, 1],      # age>30, not NY, male, high income
        [0, 1, 0, 0],      # age≤30, NY, female, low income
        [1, 1, 1, 1],      # All features = 1
        [1, 0, np.nan, 1], # Missing gender
    ])


@pytest.fixture
def sample_binary_dataframe():
    """Binary features as DataFrame (tests various missing value types)."""
    return pd.DataFrame({
        'age>30': [1, 0, 1, 1],
        'city=NY': [0, 1, '', 0],      # Empty string = missing
        'gender=M': [1, 0, 1, None],   # None = missing
        'income>50k': [1, 0, 1, pd.NA] # pd.NA = missing
    })


@pytest.fixture
def sample_labels():
    """Sample binary labels."""
    return np.array([1, 0, 1, 0])


# TODO: Implement mock_theta_sketches fixture
# This requires datasketches library
# See docs/07_testing_strategy.md for implementation

@pytest.fixture
def tmp_config_file(tmp_path):
    """Create temporary config YAML file."""
    import yaml

    config = {
        'targets': {
            'positive': 'target_yes',
            'negative': 'target_no'
        },
        'hyperparameters': {
            'criterion': 'gini',
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'verbose': 0
        },
        'feature_mapping': {
            'age>30': 0,
            'city=NY': 1,
            'gender=M': 2,
            'income>50k': 3
        }
    }

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    return str(config_path)
