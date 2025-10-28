# High-Level Architecture Document
## Theta Sketch Decision Tree Classifier

---

## 1. System Overview

The **ThetaSketchDecisionTreeClassifier** is a specialized decision tree implementation that:
- **Trains** on theta sketches (probabilistic set membership data structures)
- **Infers** on raw tabular data (standard numpy arrays/DataFrames)
- Maintains **sklearn API compatibility** for production use
- Supports **binary classification** with advanced statistical criteria

### Key Innovation
Traditional decision trees train and infer on the same data format. This implementation decouples training and inference:
- **Training**: Set-based operations on sketches (union, intersection, difference)
- **Inference**: Row-wise evaluation on raw features with feature mapping

### Scope
This implementation focuses on a **single decision tree classifier** with the following features:
- ✅ Binary classification (two classes)
- ✅ CART-style binary splits
- ✅ Multiple split criteria (Gini, Entropy, Gain Ratio, Binomial, Chi-Square)
- ✅ Advanced pruning (pre and post-pruning)
- ✅ Missing value handling (majority path method)
- ✅ Feature importance calculation
- ✅ ROC curve and performance metrics
- ✅ Full sklearn API compatibility

**Out of scope for this iteration**:
- ❌ Bagging/Random Forest wrappers (users can use sklearn's BaggingClassifier)
- ❌ Built-in cross-validation (users can use sklearn's cross_val_score)
- ❌ C4.5 multiway splits (binary CART splits only)

---

## 2. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PHASE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  CSV File Input                    Config File                   │
│  ┌──────────────────┐             ┌──────────────────┐          │
│  │ <total>, sketch  │             │ target_yes: ...  │          │
│  │ age>30, sketch   │             │ target_no: ...   │          │
│  │ income>50k,sketch│             │ hyperparams: ... │          │
│  └────────┬─────────┘             └────────┬─────────┘          │
│           │                                 │                    │
│           ▼                                 ▼                    │
│  ┌─────────────────────────────────────────────────┐            │
│  │      SketchLoader & ConfigParser                │            │
│  │  - Parse CSV sketches                           │            │
│  │  - Load target variables                        │            │
│  │  - Extract hyperparameters                      │            │
│  └────────────────────┬────────────────────────────┘            │
│                       │                                          │
│                       ▼                                          │
│  ┌─────────────────────────────────────────────────┐            │
│  │      ThetaSketchDecisionTreeClassifier          │            │
│  │                                                  │            │
│  │  fit(sketch_dict, feature_mapping, config)      │            │
│  │                                                  │            │
│  │  ┌─────────────────────────────────────┐        │            │
│  │  │      TreeBuilder                    │        │            │
│  │  │  - Build tree recursively           │        │            │
│  │  │  - Evaluate splits                  │        │            │
│  │  │  - Create nodes                     │        │            │
│  │  └──────────┬──────────────────────────┘        │            │
│  │             │                                    │            │
│  │             ▼                                    │            │
│  │  ┌─────────────────────────────────────┐        │            │
│  │  │      SplitEvaluator                 │        │            │
│  │  │  - Try all features                 │        │            │
│  │  │  - Compute sketch operations        │        │            │
│  │  │  - Evaluate criteria                │        │            │
│  │  │  - Use cache for speed              │        │            │
│  │  └──────────┬──────────────────────────┘        │            │
│  │             │                                    │            │
│  │             ▼                                    │            │
│  │  ┌─────────────────────────────────────┐        │            │
│  │  │   Criteria (Pluggable)              │        │            │
│  │  │  - Gini                             │        │            │
│  │  │  - Entropy                          │        │            │
│  │  │  - Gain Ratio                       │        │            │
│  │  │  - Binomial Test                    │        │            │
│  │  │  - Chi-Square Test                  │        │            │
│  │  └─────────────────────────────────────┘        │            │
│  │                                                  │            │
│  │  Output: Trained Tree Structure                 │            │
│  └──────────────────────┬───────────────────────────┘            │
│                         │                                        │
│                         ▼                                        │
│              ┌──────────────────────┐                            │
│              │   Tree Structure     │                            │
│              │  (Serializable)      │                            │
│              └──────────┬───────────┘                            │
└─────────────────────────┼────────────────────────────────────────┘
                          │
                          │ Model Persistence (pickle/json)
                          │
┌─────────────────────────┼────────────────────────────────────────┐
│                         ▼                                        │
│                  INFERENCE PHASE                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Raw Tabular Data (X_test)                                      │
│  ┌──────────────────────────────┐                               │
│  │  [35, 60000, 1]              │  (age, income, diabetes)      │
│  │  [25, 45000, 0]              │                               │
│  │  [55, 120000, 1]             │                               │
│  └────────────┬─────────────────┘                               │
│               │                                                  │
│               ▼                                                  │
│  ┌─────────────────────────────────────┐                        │
│  │   FeatureTransformer                │                        │
│  │  - Apply feature_mapping            │                        │
│  │  - Transform raw → binary features  │                        │
│  │  - age>30: lambda x: x > 30         │                        │
│  │  - income>50k: lambda x: x > 50000  │                        │
│  └────────────┬────────────────────────┘                        │
│               │                                                  │
│               ▼                                                  │
│  Binary Feature Matrix                                          │
│  ┌──────────────────────────────┐                               │
│  │  [1, 1, 1]  # age>30, inc>50k│                               │
│  │  [0, 0, 0]                   │                               │
│  │  [1, 1, 1]                   │                               │
│  └────────────┬─────────────────┘                               │
│               │                                                  │
│               ▼                                                  │
│  ┌─────────────────────────────────────┐                        │
│  │   TreeTraverser                     │                        │
│  │  - Navigate tree per sample         │                        │
│  │  - Handle missing values            │                        │
│  │  - Use majority path if needed      │                        │
│  └────────────┬────────────────────────┘                        │
│               │                                                  │
│               ▼                                                  │
│  ┌─────────────────────────────────────┐                        │
│  │   Predictions                       │                        │
│  │  - predict(): class labels          │                        │
│  │  - predict_proba(): probabilities   │                        │
│  └─────────────────────────────────────┘                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow

### 3.1 Training Data Flow

```
CSV Input Format (per target class):
┌────────────────────────────────┐
│ <total|empty>, <sketch_bytes>  │  ← Population sketch (total or empty marker)
│ age>30, <sketch_bytes>         │  ← Feature condition sketch
│ income>50k, <sketch_bytes>     │  ← Feature condition sketch
│ has_diabetes, <sketch_bytes>   │  ← Feature condition sketch
│ target_yes, <sketch_bytes>     │  ← Target variable sketch (positive class)
│ target_no, <sketch_bytes>      │  ← Target variable sketch (negative class)
└────────────────────────────────┘

Note: The first column can be:
- "total" or "" (empty string): Population sketch (all records)
- "dimension=value" or "item": Feature condition or item membership sketch
- "target_yes", "target_no": Target class sketches (names specified in config)
                ↓
    [Parse & Load Sketches]
                ↓
┌────────────────────────────────────────┐
│ sketch_dict = {                        │
│   'target_yes': {                      │
│     'total': ThetaSketch,              │
│     'age>30': ThetaSketch,             │
│     'income>50k': ThetaSketch,         │
│     ...                                │
│   },                                   │
│   'target_no': {                       │
│     'total': ThetaSketch,              │
│     'age>30': ThetaSketch,             │
│     ...                                │
│   }                                    │
│ }                                      │
└────────────────┬───────────────────────┘
                 ↓
        [Tree Building]
                 ↓
    For each node, evaluate splits:
                 ↓
┌────────────────────────────────────────┐
│ For feature "age>30":                  │
│                                        │
│ Class 0 (target_no):                   │
│   - n_total = sketch_total.estimate()  │
│   - n_with_feature = sketch_age.estimate()│
│   - n_without = n_total - n_with       │
│                                        │
│ Class 1 (target_yes):                  │
│   - n_total = sketch_total.estimate()  │
│   - n_with_feature = sketch_age.estimate()│
│   - n_without = n_total - n_with       │
│                                        │
│ Compute criterion:                     │
│   - Gini/Entropy/Binomial/etc.        │
│   - Select best split                  │
└────────────────┬───────────────────────┘
                 ↓
         [Create Node]
                 ↓
┌────────────────────────────────────────┐
│ Node {                                 │
│   feature: "age>30",                   │
│   left_child: Node (age ≤ 30),        │
│   right_child: Node (age > 30),       │
│   n_samples_left: 450,                 │
│   n_samples_right: 550,                │
│   missing_direction: 'right'           │
│ }                                      │
└────────────────┬───────────────────────┘
                 ↓
         [Recurse until stopping criteria]
                 ↓
        Final Tree Structure
```

### 3.2 Inference Data Flow

```
Raw Input:
┌──────────────────────────────┐
│ X = [[35, 60000, 1],         │
│      [25, 45000, 0]]         │
└────────────┬─────────────────┘
             ↓
[Apply Feature Mapping]
             ↓
feature_mapping = {
  'age>30': (0, lambda x: x > 30),
  'income>50k': (1, lambda x: x > 50000),
  'has_diabetes': (2, lambda x: x == 1)
}
             ↓
Transformed Features:
┌──────────────────────────────┐
│ [[True, True, True],         │  ← 35>30, 60k>50k, 1==1
│  [False, False, False]]      │  ← 25≤30, 45k<50k, 0!=1
└────────────┬─────────────────┘
             ↓
[Tree Traversal per Row]
             ↓
Row 0: [True, True, True]
  ├─ Root: Check age>30? → True → Go RIGHT
  ├─ Node: Check income>50k? → True → Go RIGHT
  └─ Leaf: Class = 1
             ↓
Row 1: [False, False, False]
  ├─ Root: Check age>30? → False → Go LEFT
  ├─ Node: Check has_diabetes? → False → Go LEFT
  └─ Leaf: Class = 0
             ↓
Predictions: [1, 0]
```

---

## 4. Component Interaction

### 4.1 Core Components

```
ThetaSketchDecisionTreeClassifier (Main API)
    │
    ├─► SketchLoader (CSV parsing)
    │   └─► Reads sketches from CSV
    │
    ├─► ConfigParser (Config loading)
    │   └─► Loads targets & hyperparameters
    │
    ├─► TreeBuilder (Training)
    │   ├─► SplitEvaluator
    │   │   ├─► SketchOperator (union, intersection, difference)
    │   │   ├─► SketchCache (performance)
    │   │   └─► CriterionCalculator
    │   │       ├─► GiniCriterion
    │   │       ├─► EntropyCriterion
    │   │       ├─► GainRatioCriterion
    │   │       ├─► BinomialCriterion
    │   │       └─► ChiSquareCriterion
    │   └─► Pruner
    │       ├─► PrePruner (min_samples, max_depth, min_impurity)
    │       └─► PostPruner (cost-complexity)
    │
    ├─► FeatureTransformer (Inference)
    │   └─► Applies feature_mapping to raw data
    │
    ├─► TreeTraverser (Inference)
    │   └─► MissingValueHandler (majority path)
    │
    └─► Utilities
        ├─► FeatureImportanceCalculator
        ├─► MetricsCalculator (ROC, AUC)
        └─► TreeVisualizer
```

### 4.2 Interaction Sequence

**Training Flow:**
```
1. User calls fit(csv_path, config_path)
2. SketchLoader.load(csv_path) → sketch_dict
3. ConfigParser.load(config_path) → targets, hyperparams, feature_mapping
4. TreeBuilder.build(sketch_dict, targets, hyperparams)
   a. SplitEvaluator.find_best_split(current_sketches, features)
      - For each feature:
        * SketchOperator.compute_splits(sketches)
        * CriterionCalculator.evaluate(split)
        * SketchCache.get_or_compute()
      - Return best split
   b. Create node with split
   c. Pruner.should_prune(node) → True/False
   d. Recurse on children if not pruned
5. Return trained tree
6. Compute feature_importances_
```

**Inference Flow:**
```
1. User calls predict(X_raw)
2. Check is_fitted → raise NotFittedError if not
3. FeatureTransformer.transform(X_raw, feature_mapping) → X_binary
4. For each row in X_binary:
   a. TreeTraverser.traverse(row, tree.root)
      - If missing value: MissingValueHandler.get_direction(node)
      - Follow tree until leaf
   b. Collect leaf prediction
5. Return predictions array
```

---

## 5. Integration with sklearn Ecosystem

### 5.1 Base Classes

```python
from sklearn.base import BaseEstimator, ClassifierMixin

class ThetaSketchDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    Inherits:
    - BaseEstimator: get_params(), set_params()
    - ClassifierMixin: score() method
    """
```

### 5.2 Validation Utilities

```python
from sklearn.utils.validation import (
    check_array,
    check_is_fitted
)
from sklearn.utils.multiclass import unique_labels

# Used in predict():
check_is_fitted(self)
X = check_array(X, dtype=np.float64)
```

### 5.3 Compatible Workflows

```python
# Pipeline
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('tree', ThetaSketchDecisionTreeClassifier())
])

# Grid Search
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(clf, param_grid, cv=5)

# Serialization
import pickle
pickle.dump(clf, f)
clf_loaded = pickle.load(f)
```

---

## 6. Key Design Decisions

### 6.1 Decoupled Training and Inference

**Rationale**: Theta sketches provide efficient set-based training on large populations, but inference requires row-level predictions.

**Solution**:
- Training: Operates on sketches (set cardinalities)
- Inference: Operates on raw features with transformation layer

### 6.2 Feature Mapping as Bridge

**Problem**: How to connect sketch-based splits to raw feature values?

**Solution**: `feature_mapping` dictionary:
```python
{
    'age>30': (0, lambda x: x > 30),  # column index, condition function
    'income>50k': (1, lambda x: x > 50000)
}
```

At training: Use sketch named 'age>30'
At inference: Apply lambda to column 0 of raw data

### 6.3 CSV-Based Sketch Input

**Format**:
```
<total>,<sketch_bytes>              # Population sketch
<dimension=value>,<sketch_bytes>    # Feature condition sketch
```

**Advantages**:
- Standard format for persistence
- Easy to generate from batch processing
- Supports versioning and auditing

### 6.4 Config-Driven Targets

**Config File** (YAML format):
```yaml
# Target class specification
targets:
  positive: "target_yes"    # Name of positive class sketch in CSV
  negative: "target_no"     # Name of negative class sketch in CSV

# Tree hyperparameters
hyperparameters:
  criterion: "gini"                    # gini, entropy, gain_ratio, binomial, binomial_chi
  max_depth: 10                        # Maximum tree depth (null = unlimited)
  min_samples_split: 20                # Min samples to split
  min_samples_leaf: 10                 # Min samples in leaf
  min_impurity_decrease: 0.001         # Min impurity decrease for split
  class_weight: "balanced"             # balanced, null, or dict
  missing_value_strategy: "majority"   # majority, zero, error
  pruning: "both"                      # null, pre, post, both
  ccp_alpha: 0.01                      # Cost-complexity parameter
  use_cache: true                      # Cache sketch operations
  cache_size_mb: 100                   # Cache size
  random_state: 42                     # Random seed
  verbose: 1                           # Verbosity level

# Feature mapping for inference (raw data → binary features)
feature_mapping:
  "age>30":
    column_index: 0     # Column in raw data
    operator: ">"       # Comparison operator
    threshold: 30       # Threshold value
  "income>50k":
    column_index: 1
    operator: ">"
    threshold: 50000
  "has_diabetes":
    column_index: 2
    operator: "=="
    threshold: 1
```

**Advantages**:
- Separates code from configuration
- Easy hyperparameter tuning without code changes
- Reproducible experiments (version control config files)
- Clear documentation of feature engineering

### 6.5 Pluggable Criterion Architecture

**Design**: Abstract base class with multiple implementations

**Rationale**: Different criteria for different use cases:
- **Gini/Entropy**: Fast, general-purpose
- **Gain Ratio**: Better for imbalanced features
- **Binomial/Chi-square**: Statistical rigor for medical applications

### 6.6 Missing Value Strategy

**Majority Path Method**:
- At training: Record which direction (left/right) had more samples
- At inference: Send missing values to majority direction

**Rationale**:
- No imputation needed (preserves data integrity)
- Computationally efficient
- Handles "not seen in training" features gracefully

---

## 7. Performance Considerations

### 7.1 Caching Strategy

**Cache sketch operations** (expensive):
- `sketch.get_estimate()`
- `sketch_a.intersection(sketch_b)`
- `sketch_a.union(sketch_b)`
- `sketch_a.a_not_b(sketch_b)`

**LRU Cache** with configurable size (default 100MB)

**Expected Speedup**: 2-5x for training

### 7.2 Lazy Evaluation

**Defer expensive computations**:
- Feature importances computed on demand
- Tree depth calculated when needed
- Metrics (ROC, AUC) computed only when requested

### 7.3 Efficient Tree Traversal

**Inference optimization**:
- Vectorized feature transformation where possible
- Direct array indexing for feature lookup
- No unnecessary copies of data

---

## 8. Extensibility Points

### 8.1 Custom Criteria

Users can add new split criteria by subclassing `BaseCriterion`:

```python
class CustomCriterion(BaseCriterion):
    def evaluate(self, split_data):
        # Custom logic
        return score
```

### 8.2 Custom Pruning Strategies

Pluggable pruning via `BasePruner`:

```python
class CustomPruner(BasePruner):
    def should_prune(self, node):
        # Custom logic
        return True/False
```

### 8.3 Alternative Sketch Formats

`SketchLoader` interface allows different sketch implementations:

```python
class CustomSketchLoader(BaseSketchLoader):
    def load(self, path):
        # Load from different format
        return sketch_dict
```

---

## 9. Error Handling Strategy

### 9.1 Input Validation

- **CSV Format**: Validate sketch format, check for required fields
- **Config File**: Validate YAML/JSON schema, check required keys
- **Feature Mapping**: Ensure consistency between training and inference

### 9.2 Runtime Checks

- **NotFittedError**: Predict called before fit
- **ValueError**: Invalid hyperparameters
- **KeyError**: Missing features in feature_mapping
- **TypeError**: Invalid sketch type

### 9.3 Graceful Degradation

- Missing values → majority path (not error)
- Invalid split → try next feature
- Insufficient samples → create leaf node

---

## 10. Testing Strategy

### 10.1 Unit Tests

- Each component tested independently
- Mock theta sketches for deterministic tests
- Edge cases (empty sketches, single class, etc.)

### 10.2 Integration Tests

- End-to-end: CSV → Training → Inference
- sklearn compatibility checks
- Serialization/deserialization

### 10.3 Performance Tests

- Training time benchmarks
- Cache effectiveness
- Memory usage profiling

---

## 11. Deployment Considerations

### 11.1 Model Persistence

**Supported formats**:
- Pickle (sklearn standard)
- JSON (tree structure export for interpretability)

### 11.2 Production API

```python
# Training (offline)
clf = ThetaSketchDecisionTreeClassifier()
clf.fit(csv_path="sketches.csv", config_path="config.yaml")
clf.save("model.pkl")

# Inference (online)
clf = ThetaSketchDecisionTreeClassifier.load("model.pkl")
predictions = clf.predict(X_test)
```

### 11.3 Monitoring

- Log training metrics (tree depth, number of splits)
- Track inference latency
- Monitor prediction distribution drift

---

## Summary

This architecture provides:

✅ **Sklearn Compatibility**: Drop-in replacement for decision trees
✅ **Sketch-Based Training**: Efficient set operations on theta sketches
✅ **Raw Data Inference**: Standard tabular data predictions
✅ **Production-Ready**: Error handling, logging, persistence
✅ **Extensible**: Pluggable criteria, pruning, loaders
✅ **Testable**: Clear component boundaries, mockable dependencies

The design enables training on privacy-preserving sketch summaries while maintaining the ability to make real-time predictions on raw data.
