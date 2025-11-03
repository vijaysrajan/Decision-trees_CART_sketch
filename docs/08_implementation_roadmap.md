# Implementation Roadmap
## Theta Sketch Decision Tree Classifier - Week-by-Week Plan

---

## Overview

**Total Duration**: 6 weeks
**Daily Commitment**: 4-6 hours
**Code Quality**: Production-grade with >90% test coverage

---

## Week 1: Foundation & Data Loading

### Day 1-2: Project Setup ✅ COMPLETE
- [x] Initialize git repository
- [x] Create directory structure
- [x] Setup virtual environment (Python 3.13)
- [x] Install dependencies (datasketches, sklearn, numpy, scipy, pandas, pyyaml, etc.)
- [x] Configure pytest, black, flake8, mypy
- [x] Write README.md skeleton
- [x] Create pyproject.toml, .flake8, pytest.ini
- [x] API refactored: fit() now uses sketch_data structure (separated data loading from fitting)
- [x] Helper functions created: load_sketches(), load_config() in __init__.py
- [x] Complete documentation suite (12 files including deployment strategies and tuning guide)

**Deliverables**: ✅ DONE
- Working development environment
- All dependencies installed
- Complete project structure with all module scaffolds
- All documentation complete
- API refactored for better separation of concerns

**Status**: Completed 2025-11-02

### Day 3-4: CSV Sketch Loader ⏸️ IN PROGRESS ← **YOU ARE HERE**
- [ ] Implement `SketchLoader` class
  - **CSV parsing (3-column format ONLY)**: identifier, sketch_feature_present, sketch_feature_absent
  - Base64/hex decoding
  - ThetaSketch deserialization
  - **Support dual-class mode (positive + negative CSVs) and one-vs-all mode (positive + total CSVs)**
  - **Build unified sketch_data structure with tuples**
    - `{'age>30': (sketch_feature_present, sketch_feature_absent)}` for all features
    - `{'total': sketch}` - single sketch (not tuple)
  - **Implement _compute_negative_from_total() for one-vs-all mode**
  - **Validation: Check sketch_feature_present + sketch_feature_absent ≈ total (within error bounds)**
  - Error handling and detailed error messages
- [ ] Write unit tests for `SketchLoader` (target >80% coverage)
  - Test 3-column CSV parsing
  - Test dual-class mode
  - Test one-vs-all mode (negative computation)
  - Test validation (sketch cardinality checks)
- [ ] Create test fixtures with real theta sketches
  - 3-column format only

**Files**: `sketch_loader.py`, `tests/test_sketch_loader.py`, `tests/conftest.py`

**Next Actions**:
1. Implement `_decode_sketch_bytes()` method
2. Implement `_deserialize_sketch()` method
3. Implement `_parse_csv()` method (enforce 3-column format)
4. **Implement tuple creation for all features**
5. **Implement _compute_negative_from_total() for one-vs-all mode**
6. **Implement validation: sketch_feature_present.estimate() + sketch_feature_absent.estimate() ≈ total.estimate()**
7. Implement `load()` main method with mode detection
8. Create CSV test fixtures (3-column format)
9. Write comprehensive tests for both modes

**Why 3-column format is mandatory**:
- Eliminates a_not_b operations during tree building
- **29% error reduction** at all tree depths
- Critical for imbalanced datasets (CTR, fraud) and deep trees (depth ≥3)
- Simpler API - no backward compatibility complexity
- See docs/04_data_formats.md for detailed specification

### Day 5-7: Config Parser ⏸️ PENDING
- [ ] Implement `ConfigParser` class
  - YAML/JSON parsing
  - Feature mapping simplified (no lambdas - just Dict[str, int])
  - Config validation
    - **Validate: targets has 'positive' AND either 'negative' OR 'total' (mutually exclusive)**
    - **Error if both 'negative' and 'total' are present**
  - Schema checking
- [ ] Write unit tests for `ConfigParser` (target >80% coverage)
  - Test dual-class mode validation (positive + negative)
  - Test one-vs-all mode validation (positive + total)
  - Test error when both negative and total provided
- [ ] Create test config files (YAML and JSON)
  - Examples for both dual-class and one-vs-all modes

**Files**: `config_parser.py`, `tests/test_config_parser.py`

**Week 1 Milestone**: Can load sketches from CSV and parse config files

**Checkpoint Test**:
```bash
pytest tests/test_sketch_loader.py tests/test_config_parser.py -v --cov=theta_sketch_tree --cov-report=term
```
**Success Criteria**: All tests pass, coverage >80%

---

## Week 2: Tree Structure & Splitting Logic

### Day 1-2: Tree Data Structures
- [ ] Implement `TreeNode` class
  - Internal node attributes
  - Leaf node attributes
  - `make_leaf()` method
  - `set_split()` method
  - `to_dict()` for JSON export
- [ ] Implement `Tree` class
  - Tree statistics computation
  - `to_dict()` method
- [ ] Write unit tests

**Files**: `tree_structure.py`, `tests/test_tree_structure.py`

### Day 3-5: Split Criteria
- [ ] Implement `BaseCriterion` (abstract)
- [ ] Implement `GiniCriterion`
  - Standard Gini
  - Weighted Gini
- [ ] Implement `EntropyCriterion`
  - Shannon entropy
  - Information gain calculation
- [ ] Implement `GainRatioCriterion`
- [ ] Write comprehensive unit tests
  - Pure nodes
  - Balanced nodes
  - Imbalanced nodes
  - Weighted cases

**Files**: `criteria.py`, `tests/test_criteria.py`

### Day 6-7: Statistical Criteria
- [ ] Implement `BinomialCriterion`
  - Binomial test using scipy
  - P-value calculation
  - Bonferroni correction
- [ ] Implement `ChiSquareCriterion`
  - Chi-square test
  - Contingency table construction
- [ ] Write statistical tests
  - Significant splits
  - Non-significant splits
  - Edge cases

**Week 2 Milestone**: All split criteria implemented and tested

---

## Week 3: Tree Building & Training

### Day 1-3: Split Evaluator
- [ ] Implement `SplitEvaluator` class
  - `find_best_split()` method
  - `_evaluate_feature_split()` method
  - `_compute_child_sketches()` method
  - Sketch set operations (intersection, a_not_b)
  - Feature sampling (max_features)
- [ ] Integrate with `SketchCache`
- [ ] Write unit tests
  - Test all features evaluated
  - Test best split selection
  - Test cache integration
  - Test max_features sampling

**Files**: `splitter.py`, `tests/test_splitter.py`

### Day 4-5: Sketch Cache
- [ ] Implement `SketchCache` class
  - LRU eviction policy
  - Cache key generation
  - Size tracking
  - get/put methods
- [ ] Write cache tests
  - Cache hits/misses
  - LRU eviction
  - Size limits
- [ ] Performance benchmarks

**Files**: `cache.py`, `tests/test_cache.py`

### Day 6-7: Tree Builder
- [ ] Implement `TreeBuilder` class
  - `build_tree()` recursive method
  - Stopping criteria checks
  - Node creation
  - Impurity calculations
- [ ] Write integration tests
  - Build complete trees
  - Verify tree structure
  - Test stopping criteria

**Files**: `tree_builder.py`, `tests/test_tree_builder.py`

**Week 3 Milestone**: Can build complete decision tree from sketches

---

## Week 4: Inference & Missing Values

**Note**: All feature transformation (raw → binary) happens externally. Inference works directly with binary (0/1) features.

### Day 1-2: Tree Traverser
- [ ] Implement `TreeTraverser` class
  - `predict()` method
  - `predict_proba()` method
  - `_traverse_to_leaf()` method
  - Missing value handling (majority path)
- [ ] Write unit tests
  - Test prediction correctness
  - Test probability sums
  - Test missing value strategies

**Files**: `tree_traverser.py`, `tests/test_tree_traverser.py`

### Day 5: Missing Value Handler
- [ ] Implement `MissingValueHandler` utilities
  - Missing value detection
  - Majority path logic
  - Strategy implementations
- [ ] Write unit tests
  - Test all strategies (majority, zero, error)
  - Test edge cases (all missing)

**Files**: `missing_handler.py`, `tests/test_missing_handler.py`

### Day 6-7: Main Classifier (Part 1)
- [ ] Implement `ThetaSketchDecisionTreeClassifier`
  - `__init__()` with all hyperparameters
  - `fit()` method skeleton
  - Integration of all components
  - sklearn attribute setting
- [ ] Write basic integration tests

**Files**: `classifier.py`, `tests/test_classifier.py`

**Week 4 Milestone**: End-to-end training and inference working

---

## Week 5: Advanced Features & Pruning

### Day 1-2: Pruning
- [ ] Implement `PrePruner` class
  - Early stopping criteria
  - min_impurity_decrease check
  - min_samples checks
  - max_depth check
- [ ] Implement `PostPruner` class
  - Cost-complexity pruning (CCP)
  - Alpha sequence computation
  - Pruning logic
- [ ] Write pruning tests

**Files**: `pruner.py`, `tests/test_pruner.py`

### Day 3: Feature Importance
- [ ] Implement `FeatureImportanceCalculator`
  - Gini importance
  - Split frequency importance
  - Tree traversal for aggregation
- [ ] Write tests
  - Test importance sums to 1.0
  - Test consistency

**Files**: `feature_importance.py`, `tests/test_feature_importance.py`

### Day 4-5: Metrics & Visualization
- [ ] Implement `MetricsCalculator` class
  - `compute_roc_curve()`
  - `compute_precision_recall_curve()`
  - `plot_roc_curve()`
  - `plot_pr_curve()`
- [ ] Add visualization methods to classifier
  - `plot_feature_importance()`
  - `plot_tree()` (optional)
- [ ] Write visualization tests

**Files**: `metrics.py`, `tests/test_metrics.py`

### Day 6-7: Classifier Completion
- [ ] Complete `ThetaSketchDecisionTreeClassifier`
  - All sklearn methods (predict, predict_proba, score, etc.)
  - Class weight handling
  - Feature importance property
  - Model persistence (save/load)
  - JSON export
- [ ] Write comprehensive classifier tests

**Week 5 Milestone**: Feature-complete classifier with all capabilities

---

## Week 6: Testing, Documentation & Polish

### Day 1-2: Integration Tests
- [ ] Write end-to-end integration tests
  - Complete workflow tests
  - sklearn Pipeline compatibility
  - GridSearchCV compatibility
  - Pickle serialization
- [ ] Write sklearn compatibility tests
  - Estimator checks
  - Cloning
  - Parameter getters/setters

**Files**: `tests/test_integration.py`, `tests/test_sklearn_compatibility.py`

### Day 3: Performance Benchmarks
- [ ] Write benchmark scripts
  - Training time with/without cache
  - Inference time at various scales
  - Memory usage profiling
  - Cache effectiveness measurement
- [ ] Run benchmarks and document results
- [ ] Optimize bottlenecks if needed

**Files**: `benchmarks/*.py`

### Day 4: Documentation
- [ ] Write comprehensive README.md
  - Installation instructions
  - Quick start guide
  - API documentation
  - Examples
- [ ] Complete module docstrings
- [ ] Add inline code comments
- [ ] Create example notebooks

**Files**: `README.md`, `examples/*.py`, `examples/notebooks/*.ipynb`

### Day 5-6: Code Quality & Review
- [ ] Run full test suite
  - Achieve >90% code coverage
  - Fix any failing tests
- [ ] Run code quality tools
  - black (formatting)
  - flake8 (linting)
  - mypy (type checking)
- [ ] Code review and refactoring
- [ ] Performance profiling

### Day 7: Release Preparation
- [ ] Final testing on multiple Python versions (3.8, 3.9, 3.10)
- [ ] Update CHANGELOG.md
- [ ] Tag version 0.1.0
- [ ] Build distribution packages
- [ ] Write release notes

**Week 6 Milestone**: Production-ready v0.1.0 release

---

## Detailed Task Breakdown

### Critical Path Items (Must Complete)

1. ✅ Sketch loading (Week 1)
2. ✅ Config parsing (Week 1)
3. ✅ Tree data structures (Week 2)
4. ✅ Split criteria (Week 2)
5. ✅ Split evaluation (Week 3)
6. ✅ Tree building (Week 3)
7. ✅ Feature transformation (Week 4)
8. ✅ Tree traversal (Week 4)
9. ✅ Main classifier (Week 4-5)
10. ✅ Testing >90% coverage (Week 6)

### Optional Enhancements (Post-v0.1.0)

- [ ] C4.5 multiway splits
- [ ] Bagging/Random Forest wrapper
- [ ] Built-in cross-validation
- [ ] Interactive tree visualization
- [ ] GPU-accelerated sketch operations
- [ ] Distributed training support

---

## Testing Checkpoints

### After Week 1
```bash
pytest tests/test_sketch_loader.py tests/test_config_parser.py -v
# Expected: All tests pass, coverage >80%
```

### After Week 2
```bash
pytest tests/test_tree_structure.py tests/test_criteria.py -v
# Expected: All tests pass, coverage >85%
```

### After Week 3
```bash
pytest tests/test_splitter.py tests/test_tree_builder.py tests/test_cache.py -v
# Expected: All tests pass, can build complete tree
```

### After Week 4
```bash
pytest tests/test_classifier.py tests/test_tree_traverser.py -v
# Expected: End-to-end workflow works (using pre-transformed binary features)
```

### After Week 5
```bash
pytest tests/ -v --cov=theta_sketch_tree --cov-report=term
# Expected: All features work, coverage >85%
```

### Final (Week 6)
```bash
pytest tests/ -v --cov=theta_sketch_tree --cov-report=html
# Expected: All tests pass, coverage >90%
```

---

## Code Review Checklist

Before marking complete, verify:

- [ ] All public methods have NumPy-style docstrings
- [ ] All functions have type hints
- [ ] Code passes black formatting
- [ ] Code passes flake8 linting
- [ ] Code passes mypy type checking
- [ ] Test coverage >90%
- [ ] All integration tests pass
- [ ] sklearn compatibility verified
- [ ] Performance benchmarks meet targets (>2x speedup with cache)
- [ ] Documentation complete
- [ ] Examples work correctly
- [ ] Can pickle/unpickle models
- [ ] JSON export works
- [ ] Missing value handling tested
- [ ] Class weighting tested
- [ ] All criteria tested
- [ ] Pruning tested

---

## Risk Mitigation

### Potential Risks

1. **Apache DataSketches Python API changes**
   - Mitigation: Pin specific version in requirements.txt
   - Test against multiple versions

2. **Performance bottlenecks in sketch operations**
   - Mitigation: Implement caching early (Week 3)
   - Profile and optimize in Week 6

3. **sklearn API incompatibilities**
   - Mitigation: Test sklearn compatibility continuously
   - Reference sklearn.tree.DecisionTreeClassifier source

4. **Complex statistical test edge cases**
   - Mitigation: Comprehensive unit tests for criteria
   - Validate against scipy examples

### Contingency Plans

- If behind schedule after Week 3: Skip post-pruning, focus on core functionality
- If behind schedule after Week 4: Skip visualization features
- If test coverage <90%: Dedicate extra time in Week 6

---

## Success Criteria

### v0.1.0 Release Requirements

✅ **Functional Requirements**:
- Trains on theta sketches from CSV
- Infers on raw tabular data
- Supports all 5 split criteria
- Handles missing values (majority path)
- Computes feature importance
- Calculates ROC/AUC metrics
- Supports pruning (pre and post)
- Handles class imbalance

✅ **Quality Requirements**:
- Test coverage >90%
- Code passes all quality checks (black, flake8, mypy)
- sklearn API fully compatible
- Can pickle/unpickle models
- Documentation complete

✅ **Performance Requirements**:
- Training speedup >2x with caching
- Inference time <1ms per sample (on typical hardware)
- Memory usage reasonable (<500MB for training)

---

## Post-Release Roadmap (v0.2.0+)

### v0.2.0 (2-3 weeks)
- Multi-class classification support
- C4.5 multiway splits
- Improved visualization (graphviz integration)
- More comprehensive examples

### v0.3.0 (1 month)
- Random Forest wrapper (bagging)
- Built-in cross-validation helpers
- SHAP value computation
- Model explainability features

### v1.0.0 (3 months)
- Production deployment guide
- Performance optimizations
- Distributed training support
- Comprehensive benchmarks vs sklearn

---

## Daily Progress Tracking

Use this template to track daily progress:

```
## Date: YYYY-MM-DD
Week: X, Day: Y

### Completed:
- [ ] Task 1
- [ ] Task 2

### In Progress:
- [ ] Task 3 (50% done)

### Blockers:
- None / Issue description

### Tomorrow's Plan:
- [ ] Task 4
- [ ] Task 5

### Notes:
- Any important observations or decisions
```

---

## Summary

This roadmap provides:

✅ **Week-by-week breakdown** of all implementation tasks
✅ **Clear milestones** at end of each week
✅ **Testing checkpoints** to ensure quality
✅ **Risk mitigation** strategies
✅ **Success criteria** for v0.1.0 release
✅ **Post-release roadmap** for future versions

**Estimated Total**: ~200-250 hours of work over 6 weeks

Follow this roadmap systematically to deliver a production-grade theta sketch decision tree classifier!
