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
│  CSV File Input (2 modes)          Config File                   │
│  ┌──────────────────┐             ┌──────────────────┐          │
│  │ MODE 1: Single   │             │ hyperparams: ... │          │
│  │ sketches.csv     │             │ feature_mapping: │          │
│  │ OR               │             │   age>30: 0      │          │
│  │ MODE 2: Dual CSVs│             │   income>50k: 1  │          │
│  │ target_yes.csv   │             └────────┬─────────┘          │
│  │ target_no.csv    │                      │                    │
│  └────────┬─────────┘                      │                    │
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
│  Binary Tabular Data (X_test)                                   │
│  ┌──────────────────────────────────────────┐                   │
│  │  [1, 0, 1, 1]  # Pre-computed binary     │                   │
│  │  [0, 1, 0, 0]  # age>30, city=NY, ...    │                   │
│  │  [1, 0, 1, 0]  # Features already 0/1    │                   │
│  └────────────┬───────────────────────────────┘                 │
│               │                                                  │
│               │ feature_mapping: {"age>30": 0, "city=NY": 1}    │
│               │ (just column indices, NO transformation)        │
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

**MODE 1: Single CSV with Intersections**
```
CSV Input Format (all sketches in one file):
┌────────────────────────────────┐
│ total, <sketch_bytes>          │  ← All records
│ age>30, <sketch_bytes>         │  ← Anyone with age>30 (any class)
│ income>50k, <sketch_bytes>     │  ← Anyone with income>50k (any class)
│ has_diabetes, <sketch_bytes>   │  ← Anyone with diabetes
│ target_yes, <sketch_bytes>     │  ← Positive class members
│ target_no, <sketch_bytes>      │  ← Negative class members
└────────────────────────────────┘
                ↓
    [Parse & Intersect Sketches]
    (target_yes ∩ age>30, target_no ∩ age>30, ...)
                ↓
```

**MODE 2: Dual CSV Pre-Intersected (RECOMMENDED)**
```
target_yes.csv:                    target_no.csv:
┌────────────────────────────┐    ┌────────────────────────────┐
│ total, <pos_class_sketch>  │    │ total, <neg_class_sketch>  │
│ age>30, <pos_AND_age_sketch>│   │ age>30, <neg_AND_age_sketch>│
│ income>50k, <pos_AND_inc>  │    │ income>50k, <neg_AND_inc>  │
│ has_diabetes, <pos_AND_dia>│    │ has_diabetes, <neg_AND_dia>│
└────────────────────────────┘    └────────────────────────────┘
    ↓                                   ↓
    [Parse Sketches - NO intersection needed]
                ↓
```

**Result (both modes):**
```
┌────────────────────────────────────────┐
│ sketch_dict = {                        │
│   'target_yes': {                      │
│     'total': ThetaSketch,              │  ← Positive class total
│     'age>30': ThetaSketch,             │  ← Positive AND age>30
│     'income>50k': ThetaSketch,         │  ← Positive AND income>50k
│     ...                                │
│   },                                   │
│   'target_no': {                       │
│     'total': ThetaSketch,              │  ← Negative class total
│     'age>30': ThetaSketch,             │  ← Negative AND age>30
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
Binary Input (features already transformed externally):
┌──────────────────────────────┐
│ X = [[1, 0, 1, 1],           │  ← age>30=1, city=NY=0, gender=M=1, income>50k=1
│      [0, 1, 0, 0]]           │  ← age>30=0, city=NY=1, gender=M=0, income>50k=0
└────────────┬─────────────────┘
             ↓
[Map feature names to column indices]
             ↓
feature_mapping = {
  'age>30': 0,        # Column 0 contains age>30 binary values
  'city=NY': 1,       # Column 1 contains city=NY binary values
  'gender=M': 2,      # Column 2 contains gender=M binary values
  'income>50k': 3     # Column 3 contains income>50k binary values
}
             ↓
[Tree Traversal per Row]
             ↓
Row 0: [1, 0, 1, 1]
  ├─ Root: Check age>30 (col 0)? → 1 (True) → Go RIGHT
  ├─ Node: Check income>50k (col 3)? → 1 (True) → Go RIGHT
  └─ Leaf: Class = 1
             ↓
Row 1: [0, 1, 0, 0]
  ├─ Root: Check age>30 (col 0)? → 0 (False) → Go LEFT
  ├─ Node: Check city=NY (col 1)? → 1 (True) → Go RIGHT
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
    ├─► TreeTraverser (Inference)
    │   └─► MissingValueHandler (majority path)
    │
    └─► Utilities
        ├─► FeatureImportanceCalculator
        ├─► MetricsCalculator (ROC, AUC)
        └─► TreeVisualizer
```

### 4.2 Interaction Sequence

**Training Flow (NEW - Separated Concerns):**
```
1. User loads data (separate from fitting):
   a. sketch_data = load_sketches(positive_csv, negative_csv)
   b. config = load_config(config_path)

2. User initializes classifier with hyperparameters:
   clf = ThetaSketchDecisionTreeClassifier(**config['hyperparameters'])

3. User calls fit(sketch_data, feature_mapping):
   a. Validate sketch_data structure (has 'positive', 'negative', 'total')
   b. TreeBuilder.build(sketch_data, hyperparams)
      - SplitEvaluator.find_best_split(current_sketches, features)
        * For each feature:
          - Get sketch_present and sketch_absent from sketch_data
          - CriterionCalculator.evaluate(split)
          - SketchCache.get_or_compute()
        * Return best split
      - Create node with split
      - Pruner.should_prune(node) → True/False
      - Recurse on children if not pruned
   c. Return trained tree
   d. Compute feature_importances_

Alternative (Convenience Method):
clf = ThetaSketchDecisionTreeClassifier.fit_from_csv(
    positive_csv='yes.csv',
    negative_csv='no.csv',
    config_path='config.yaml'
)
```

**Inference Flow:**
```
1. User calls predict(X)  # X is already binary features (0/1)
2. Check is_fitted → raise NotFittedError if not
3. For each row in X:
   a. TreeTraverser.traverse(row, tree.root)
      - Use feature_mapping to get column index for each split feature
      - If missing value: MissingValueHandler.get_direction(node)
      - Follow tree until leaf
   b. Collect leaf prediction
4. Return predictions array
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
- Inference: Operates on binary features (0/1 values, pre-transformed externally)

### 6.2 Feature Mapping as Bridge

**Problem**: How to connect sketch-based splits (named features) to inference columns?

**Solution**: `feature_mapping` dictionary maps feature names to column indices:
```python
{
    'age>30': 0,        # Feature 'age>30' is in column 0
    'income>50k': 1,    # Feature 'income>50k' is in column 1
    'city=NY': 2        # Feature 'city=NY' is in column 2
}
```

**At training**: Tree learns splits using sketch names like 'age>30'
**At inference**: Use feature_mapping to find which column contains 'age>30' (already binary 0/1)

**Note**: All transformations (age > 30, city == 'NY', etc.) happen BEFORE data reaches our classifier. Inference data arrives as pre-computed binary features.

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

# Feature mapping for inference (maps feature names to column indices)
feature_mapping:
  "age>30": 0           # Column 0 contains binary age>30 values (0/1)
  "income>50k": 1       # Column 1 contains binary income>50k values (0/1)
  "city=NY": 2          # Column 2 contains binary city=NY values (0/1)
  "has_diabetes": 3     # Column 3 contains binary diabetes indicator (0/1)
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
from theta_sketch_tree import load_sketches, load_config, ThetaSketchDecisionTreeClassifier

# Method 1: Explicit (more control)
sketch_data = load_sketches('target_yes.csv', 'target_no.csv')
config = load_config('config.yaml')
clf = ThetaSketchDecisionTreeClassifier(**config['hyperparameters'])
clf.fit(sketch_data, config['feature_mapping'])
clf.save("model.pkl")

# Method 2: Convenience (one-liner)
clf = ThetaSketchDecisionTreeClassifier.fit_from_csv(
    positive_csv='target_yes.csv',
    negative_csv='target_no.csv',
    config_path='config.yaml'
)
clf.save("model.pkl")

# Inference (online)
clf = ThetaSketchDecisionTreeClassifier.load("model.pkl")
predictions = clf.predict(X_test)  # X_test has binary features (0/1)
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
