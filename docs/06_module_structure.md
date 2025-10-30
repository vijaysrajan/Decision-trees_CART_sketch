# Module Structure Document
## Theta Sketch Decision Tree Classifier - Complete File Organization

---

## Directory Structure

```
Decision-trees_CART_sketch/
├── README.md
├── setup.py
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── theta_sketch_tree/                  # Main package
│   ├── __init__.py                     # Package exports
│   ├── classifier.py                   # ThetaSketchDecisionTreeClassifier (main API)
│   ├── tree_structure.py               # Tree, TreeNode classes
│   ├── sketch_loader.py                # SketchLoader (CSV parsing)
│   ├── config_parser.py                # ConfigParser (YAML/JSON parsing)
│   ├── tree_builder.py                 # TreeBuilder (recursive construction)
│   ├── splitter.py                     # SplitEvaluator (split selection)
│   ├── criteria.py                     # All criterion classes
│   │   # - BaseCriterion (abstract)
│   │   # - GiniCriterion
│   │   # - EntropyCriterion
│   │   # - GainRatioCriterion
│   │   # - BinomialCriterion
│   │   # - ChiSquareCriterion
│   ├── tree_traverser.py               # TreeTraverser (inference navigation)
│   ├── missing_handler.py              # MissingValueHandler
│   ├── pruner.py                       # BasePruner, PrePruner, PostPruner
│   ├── feature_importance.py           # FeatureImportanceCalculator
│   ├── metrics.py                      # MetricsCalculator (ROC, AUC)
│   ├── cache.py                        # SketchCache (LRU cache)
│   └── utils.py                        # Helper functions
│
├── tests/                              # Test suite
│   ├── __init__.py
│   ├── conftest.py                     # Pytest fixtures (mock sketches)
│   ├── test_classifier.py              # Main API tests
│   ├── test_tree_structure.py          # Tree/TreeNode tests
│   ├── test_sketch_loader.py           # CSV parsing tests
│   ├── test_config_parser.py           # Config parsing tests
│   ├── test_tree_builder.py            # Tree building tests
│   ├── test_splitter.py                # Split evaluation tests
│   ├── test_criteria.py                # Criterion calculation tests
│   ├── test_tree_traverser.py          # Tree traversal tests
│   ├── test_missing_handler.py         # Missing value handling tests
│   ├── test_pruner.py                  # Pruning tests
│   ├── test_feature_importance.py      # Feature importance tests
│   ├── test_metrics.py                 # Metrics calculation tests
│   ├── test_cache.py                   # Cache tests
│   ├── test_integration.py             # End-to-end integration tests
│   ├── test_sklearn_compatibility.py   # sklearn compatibility tests
│   └── test_data/                      # Test data files
│       ├── test_sketches.csv
│       ├── test_config.yaml
│       └── test_config_invalid.yaml
│
├── examples/                           # Example scripts and notebooks
│   ├── 01_basic_usage.py
│   ├── 02_medical_use_case.py
│   ├── 03_hyperparameter_tuning.py
│   ├── 04_feature_importance.py
│   ├── notebooks/
│   │   ├── basic_classification.ipynb
│   │   ├── medical_readmission.ipynb
│   │   └── model_interpretability.ipynb
│   └── data/
│       ├── example_sketches.csv
│       └── example_config.yaml
│
├── benchmarks/                         # Performance benchmarks
│   ├── benchmark_training_time.py
│   ├── benchmark_inference_time.py
│   ├── benchmark_cache_effectiveness.py
│   └── benchmark_memory_usage.py
│
└── docs/                               # Documentation
    ├── 01_high_level_architecture.md
    ├── 02_low_level_design.md
    ├── 03_algorithms.md
    ├── 04_data_formats.md
    ├── 05_api_design.md
    ├── 06_module_structure.md          # This file
    ├── 07_testing_strategy.md
    ├── 08_implementation_roadmap.md
    └── images/                         # Diagrams and figures
        ├── architecture_diagram.png
        ├── class_hierarchy.png
        └── workflow_diagram.png
```

---

## Module Descriptions

### theta_sketch_tree/__init__.py

```python
"""
Theta Sketch Decision Tree Classifier

A sklearn-compatible decision tree classifier that trains on theta sketches
but performs inference on raw tabular data.
"""

__version__ = '0.1.0'
__author__ = 'Your Name'
__license__ = 'MIT'

from .classifier import ThetaSketchDecisionTreeClassifier
from .tree_structure import Tree, TreeNode

__all__ = [
    'ThetaSketchDecisionTreeClassifier',
    'Tree',
    'TreeNode',
]
```

### theta_sketch_tree/classifier.py

**Lines of Code**: ~500-700
**Dependencies**: sklearn, numpy, all other modules

Main API class implementing sklearn estimator interface.

**Key Methods**:
- `__init__()`: Initialize with hyperparameters
- `fit()`: Train on CSV sketches + config
- `predict()`: Predict class labels
- `predict_proba()`: Predict probabilities
- `save_model()` / `load_model()`: Persistence
- `export_tree_json()`: Tree export
- `compute_roc_curve()`: ROC metrics
- `plot_feature_importance()`: Visualization

### theta_sketch_tree/tree_structure.py

**Lines of Code**: ~200-300
**Dependencies**: numpy

Tree data structures.

**Classes**:
- `TreeNode`: Single node (internal or leaf)
- `Tree`: Complete tree with statistics

### theta_sketch_tree/sketch_loader.py

**Lines of Code**: ~150-200
**Dependencies**: csv, base64, datasketches

Load and deserialize theta sketches from CSV.

**Class**: `SketchLoader`
**Key Methods**:
- `load()`: Parse CSV and return sketch dicts
- `_decode_sketch_bytes()`: Decode base64/hex
- `_deserialize_sketch()`: Bytes → ThetaSketch

### theta_sketch_tree/config_parser.py

**Lines of Code**: ~200-250
**Dependencies**: yaml (or json)

Parse YAML/JSON configuration files.

**Class**: `ConfigParser`
**Key Methods**:
- `load()`: Load and parse config file
- `parse_feature_mapping()`: Parse simple feature mapping (Dict[str, int])
- `validate_config()`: Schema validation

### theta_sketch_tree/tree_builder.py

**Lines of Code**: ~300-400
**Dependencies**: numpy, tree_structure, splitter, pruner

Recursive tree construction.

**Class**: `TreeBuilder`
**Key Methods**:
- `build_tree()`: Recursive tree building
- `_should_stop_splitting()`: Stopping criteria

### theta_sketch_tree/splitter.py

**Lines of Code**: ~400-500
**Dependencies**: numpy, criteria, cache

Split evaluation and selection.

**Class**: `SplitEvaluator`
**Key Methods**:
- `find_best_split()`: Evaluate all features
- `_evaluate_feature_split()`: Single feature
- `_compute_child_sketches()`: Sketch operations
- `_select_features_to_try()`: Feature sampling

### theta_sketch_tree/criteria.py

**Lines of Code**: ~500-600
**Dependencies**: numpy, scipy.stats

All split criteria implementations.

**Classes**:
- `BaseCriterion`: Abstract base class
- `GiniCriterion`: Gini impurity
- `EntropyCriterion`: Shannon entropy
- `GainRatioCriterion`: C4.5 gain ratio
- `BinomialCriterion`: Binomial test
- `ChiSquareCriterion`: Chi-square test

### theta_sketch_tree/tree_traverser.py

**Lines of Code**: ~150-200
**Dependencies**: numpy, pandas, tree_structure

Navigate tree for predictions.

**Class**: `TreeTraverser`
**Key Methods**:
- `predict()`: Batch predictions
- `predict_proba()`: Batch probabilities
- `_traverse_to_leaf()`: Single sample traversal

### theta_sketch_tree/missing_handler.py

**Lines of Code**: ~50-100
**Dependencies**: None (utility functions)

Missing value handling utilities.

**Functions**:
- `is_missing()`: Check if value is missing
- `handle_missing()`: Apply strategy

### theta_sketch_tree/pruner.py

**Lines of Code**: ~300-400
**Dependencies**: numpy, tree_structure

Pre and post pruning.

**Classes**:
- `BasePruner`: Abstract base
- `PrePruner`: Early stopping
- `PostPruner`: Cost-complexity pruning

### theta_sketch_tree/feature_importance.py

**Lines of Code**: ~150-200
**Dependencies**: numpy, tree_structure

Feature importance calculation.

**Class**: `FeatureImportanceCalculator`
**Key Methods**:
- `compute_gini_importance()`: Impurity-based
- `compute_split_frequency_importance()`: Count-based

### theta_sketch_tree/metrics.py

**Lines of Code**: ~100-150
**Dependencies**: sklearn.metrics, matplotlib

Performance metrics and visualization.

**Class**: `MetricsCalculator`
**Key Methods**:
- `compute_roc_curve()`: ROC data
- `compute_precision_recall_curve()`: PR data
- `plot_roc_curve()`: ROC plot
- `plot_pr_curve()`: PR plot

### theta_sketch_tree/cache.py

**Lines of Code**: ~150-200
**Dependencies**: hashlib

LRU cache for sketch operations.

**Class**: `SketchCache`
**Key Methods**:
- `get_key()`: Generate cache key
- `get()`: Retrieve from cache
- `put()`: Store with LRU eviction
- `clear()`: Clear cache

### theta_sketch_tree/utils.py

**Lines of Code**: ~100-150
**Dependencies**: numpy

Utility functions.

**Functions**:
- `validate_array()`: Input validation
- `compute_class_counts()`: Aggregate counts
- `log_message()`: Logging helper

---

## Dependencies

### Required Packages (requirements.txt)

```
# Core dependencies
numpy>=1.21.0
scikit-learn>=1.0.0
scipy>=1.7.0
pandas>=1.3.0
pyyaml>=5.4.0

# Apache DataSketches
datasketches>=3.0.0

# Visualization (optional)
matplotlib>=3.4.0
seaborn>=0.11.0

# Development
pytest>=7.0.0
pytest-cov>=3.0.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.950
```

### setup.py

```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="theta-sketch-tree",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="CART Decision Tree trained on theta sketches",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/theta-sketch-tree",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "pyyaml>=5.4.0",
        "datasketches>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "viz": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
    },
)
```

---

## Import Structure

### Internal Imports

```python
# In classifier.py
from .tree_structure import Tree, TreeNode
from .sketch_loader import SketchLoader
from .config_parser import ConfigParser
from .tree_builder import TreeBuilder
from .splitter import SplitEvaluator
from .tree_traverser import TreeTraverser
from .cache import SketchCache
from .feature_importance import FeatureImportanceCalculator
from .metrics import MetricsCalculator

# In tree_builder.py
from .tree_structure import TreeNode
from .splitter import SplitEvaluator
from .pruner import PrePruner, PostPruner
from .criteria import (
    GiniCriterion,
    EntropyCriterion,
    GainRatioCriterion,
    BinomialCriterion,
    ChiSquareCriterion
)
```

### External Imports

```python
# Standard library
import os
import csv
import json
import pickle
import hashlib
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from abc import ABC, abstractmethod

# Third-party
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix
)
from scipy.stats import binomtest, chi2_contingency
import matplotlib.pyplot as plt
from datasketches import compact_theta_sketch
```

---

## Code Organization Guidelines

### 1. File Size Limits
- Keep modules under 800 lines
- Split large modules into sub-modules if needed
- Example: `criteria.py` could be split into `criteria/gini.py`, `criteria/entropy.py`, etc.

### 2. Naming Conventions
- Classes: PascalCase (`ThetaSketchDecisionTreeClassifier`)
- Functions/methods: snake_case (`compute_roc_curve`)
- Constants: UPPER_SNAKE_CASE (`DEFAULT_CACHE_SIZE_MB`)
- Private methods: prefix with underscore (`_compute_child_sketches`)

### 3. Documentation
- All public classes/functions have NumPy-style docstrings
- All modules have module-level docstrings
- Type hints for all function signatures

### 4. Testing
- One test file per module
- Test coverage target: >90%
- Use pytest fixtures for mock sketches

---

## Build and Install

```bash
# Development installation
pip install -e ".[dev,viz]"

# Run tests
pytest tests/ --cov=theta_sketch_tree --cov-report=html

# Code formatting
black theta_sketch_tree/ tests/

# Linting
flake8 theta_sketch_tree/ tests/

# Type checking
mypy theta_sketch_tree/

# Build distribution
python setup.py sdist bdist_wheel
```

---

## Summary

**Total Estimated Lines of Code**: ~4,000-5,000
**Number of Modules**: 16 core modules
**Number of Test Files**: 16 test files
**Test Coverage Target**: >90%

This structure provides:
✅ **Clear separation of concerns**
✅ **Testable components**
✅ **sklearn compatibility**
✅ **Extensible architecture**
✅ **Production-ready packaging**
