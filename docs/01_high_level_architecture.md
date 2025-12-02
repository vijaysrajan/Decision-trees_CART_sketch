# High-Level Architecture Document
## Technical Architecture Decisions and Design Principles

---

## ⚠️ CRITICAL DESIGN DECISION: Sketch Pre-Computation Pattern

**All sketches are pre-computed directly from raw big data sources**. The training phase loads these pre-computed sketches from CSV files and uses ONLY intersection operations during tree building - NO a_not_b, union, or other set operations are used.

**Key Architectural Decisions**:
- CSV files contain sketches computed from the original big data pipeline
- For One-vs-All mode: The `total.csv` contains sketches of the ENTIRE dataset (unfiltered), NOT computed via set operations
- Loader performs NO set operations - it simply reads pre-computed sketches
- Negative class counts: Arithmetic subtraction at numeric level (`n_neg = n_total - n_pos`), NOT at sketch level

**Design Rationale**: This pattern ensures deterministic behavior, eliminates sketch operation errors, and maintains data lineage from big data sources.

---

## Architecture Design Principles

### 1. **Separation of Concerns**
- **Training subsystem**: Handles sketch loading and tree construction
- **Inference subsystem**: Handles raw data prediction and traversal
- **Persistence subsystem**: Handles model serialization independently
- **Validation subsystem**: Centralized input checking across all components

### 2. **Dependency Inversion**
- Core algorithm depends on abstractions (interfaces), not concrete implementations
- Split criteria are pluggable via abstract base classes
- Pruning methods follow strategy pattern for extensibility
- Data loading supports multiple formats via loader interfaces

### 3. **Fail-Fast Validation**
- Input validation at API boundaries prevents downstream errors
- Type checking with numpy typing for compile-time safety
- Comprehensive error messages with remediation suggestions
- Early detection of incompatible sketch formats

### 4. **Performance by Design**
- Sketch operations are read-only (no expensive copies)
- Tree traversal optimized for cache locality
- Missing value handling via majority vote (single pass)
- Feature importance calculated incrementally during training

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PHASE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  CSV File Input (Dual CSV)         Config File                   │
│  ┌──────────────────┐             ┌──────────────────┐          │
│  │ Dual-Class Mode: │             │ hyperparams: ... │          │
│  │  positive.csv    │             │ feature_mapping: │          │
│  │  negative.csv    │             │   age>30: 0      │          │
│  │ OR               │             │   income>50k: 1  │          │
│  │ One-vs-All Mode: │             └────────┬─────────┘          │
│  │  positive.csv    │                      │                    │
│  │  total.csv       │                      │                    │
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
│  │  │      TreeOrchestrator               │        │            │
│  │  │  - Coordinates tree building        │        │            │
│  │  │  - Manages logging and validation   │        │            │
│  │  │  - Handles pruning decisions        │        │            │
│  │  └──────────┬──────────────────────────┘        │            │
│  │             │                                    │            │
│  │             ▼                                    │            │
│  │  ┌─────────────────────────────────────┐        │            │
│  │  │      SplitFinder                    │        │            │
│  │  │  - Evaluates all feature splits     │        │            │
│  │  │  - Computes sketch intersections    │        │            │
│  │  │  - Applies split criteria           │        │            │
│  │  │  - Caches expensive operations      │        │            │
│  │  └──────────┬──────────────────────────┘        │            │
│  │             │                                    │            │
│  │             ▼                                    │            │
│  │  ┌─────────────────────────────────────┐        │            │
│  │  │   Criteria (Pluggable)              │        │            │
│  │  │  - Gini, Entropy, Chi-Square        │        │            │
│  │  │  - Gain Ratio, Binomial             │        │            │
│  │  │  - Custom criteria via inheritance  │        │            │
│  │  └─────────────────────────────────────┘        │            │
│  └─────────────────────────────────────────────────┘            │
│                       │                                          │
│                       ▼                                          │
│  ┌─────────────────────────────────────────────────┐            │
│  │            Trained TreeNode                     │            │
│  │  - feature_idx, threshold, children             │            │
│  │  - class counts, probabilities                  │            │
│  │  - missing value handling                       │            │
│  └─────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE PHASE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Binary Feature Matrix (NumPy)    Feature Mapping              │
│  ┌──────────────────┐             ┌──────────────────┐          │
│  │ X = [[1,0,1],    │             │ 0: "age>30"      │          │
│  │      [0,1,0],    │             │ 1: "income>50k"  │          │
│  │      [1,1,1]]    │             │ 2: "has_degree"  │          │
│  └────────┬─────────┘             └──────────────────┘          │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────┐            │
│  │      ThetaSketchDecisionTreeClassifier          │            │
│  │                                                  │            │
│  │  predict(X) / predict_proba(X)                   │            │
│  │                                                  │            │
│  │  ┌─────────────────────────────────────┐        │            │
│  │  │      TreeTraverser                  │        │            │
│  │  │  - Row-wise tree traversal          │        │            │
│  │  │  - Missing value majority vote      │        │            │
│  │  │  - Feature mapping application      │        │            │
│  │  │  - Batch prediction optimization    │        │            │
│  │  └─────────────────────────────────────┘        │            │
│  └─────────────────────────────────────────────────┘            │
│                       │                                          │
│                       ▼                                          │
│  ┌─────────────────────────────────────────────────┐            │
│  │         Predictions & Probabilities             │            │
│  │  - Class labels: [1, 0, 1]                     │            │
│  │  - Probabilities: [[0.2,0.8], [0.7,0.3], ...]  │            │
│  └─────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Critical Technical Decisions

### 1. **Sketch Data Structure Design**

**Decision**: Use tuple-based sketch storage with pre-computed intersections

```python
sketch_data = {
  'positive': {
    'total': ThetaSketch,                          # Pre-computed positive class
    'age>30': (ThetaSketch_present,                # Tuple: (positive AND age>30,
               ThetaSketch_absent),                #         positive AND age<=30)
    'income>50k': (ThetaSketch_present, ThetaSketch_absent),
    ...
  },
  'negative': {  # Same structure
    'total': ThetaSketch,
    'age>30': (ThetaSketch_present, ThetaSketch_absent),
    ...
  }
}
```

**Rationale**:
- ✅ **Deterministic**: No runtime sketch operations that could fail
- ✅ **Performance**: O(1) lookups instead of O(k) set operations
- ✅ **Debugging**: Clear data lineage from big data source
- ✅ **Memory**: Avoids creating temporary sketches during training

**Trade-offs**:
- ❌ **Storage**: 2x sketch storage (present + absent for each feature)
- ❌ **Preprocessing**: Big data pipeline must compute all combinations

### 2. **Component Orchestration Pattern**

**Decision**: TreeOrchestrator coordinates high-level operations, SplitFinder handles low-level evaluation

```python
class TreeOrchestrator:
    def build_tree(self):
        # High-level coordination
        while has_nodes_to_split():
            node = select_next_node()
            split = self.split_finder.find_best_split(node)
            self.apply_split_or_make_leaf(node, split)

class SplitFinder:
    def find_best_split(self, node):
        # Low-level split evaluation
        for feature in features:
            score = evaluate_feature_split(feature, node)
        return best_split
```

**Rationale**:
- ✅ **Single Responsibility**: Each component has one clear purpose
- ✅ **Testability**: Can unit test split logic independently
- ✅ **Extensibility**: Easy to add new split strategies
- ✅ **Debugging**: Clear separation of orchestration vs evaluation

### 3. **Missing Value Handling Strategy**

**Decision**: Majority-vote missing value handling during inference

```python
def traverse_to_leaf(sample, node):
    if is_missing(sample[node.feature_idx]):
        # Follow majority branch
        if node.left.n_samples >= node.right.n_samples:
            return traverse_to_leaf(sample, node.left)
        else:
            return traverse_to_leaf(sample, node.right)
    else:
        # Standard traversal
        return traverse_based_on_feature_value(sample, node)
```

**Rationale**:
- ✅ **Simple**: Easy to understand and debug
- ✅ **Fast**: Single comparison, no complex imputation
- ✅ **Statistically Sound**: Follows most probable path
- ✅ **Training-Inference Consistency**: Same strategy in both phases

**Alternatives Rejected**:
- ❌ **Surrogate Splits**: Too complex for binary features
- ❌ **Imputation**: Adds preprocessing dependency
- ❌ **Probabilistic Traversal**: Too slow for production

### 4. **API Design: sklearn Compatibility**

**Decision**: Full BaseEstimator and ClassifierMixin inheritance

```python
class ThetaSketchDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):           # X = sketch_dict, y = feature_mapping
    def predict(self, X):          # X = raw binary matrix
    def predict_proba(self, X):    # X = raw binary matrix
    def get_params(self, deep=True):
    def set_params(self, **params):
```

**Rationale**:
- ✅ **Ecosystem Compatibility**: Works with GridSearchCV, pipelines, etc.
- ✅ **User Familiarity**: No learning curve for sklearn users
- ✅ **Tool Integration**: Compatible with MLflow, joblib, etc.
- ✅ **Testing**: Can use sklearn's check_estimator tests

**Design Challenge**: Training data format (sketches) differs from inference data format (raw)
**Solution**: Use `fit(sketch_dict, feature_mapping)` pattern where:
- `sketch_dict` contains pre-computed sketches
- `feature_mapping` maps feature names to column indices for inference

---

## Performance Architecture Decisions

### 1. **Memory Management**

**Decision**: Copy-free sketch operations with read-only access
- Sketches are never modified during training
- Tree nodes store references, not copies
- Garbage collection handles cleanup automatically

### 2. **Computational Optimization**

**Decision**: Incremental feature importance calculation
- Calculate during training, not as post-processing
- Weighted impurity decrease computed per split
- Amortized O(1) per node instead of O(tree) post-hoc

### 3. **Caching Strategy**

**Decision**: Cache expensive sketch intersection results within single split evaluation
- Cache sketch estimates during feature evaluation
- Clear cache between nodes to control memory
- Avoid duplicate sketch intersection computations

---

## Extension Points

### 1. **Custom Split Criteria**
Implement `SplitCriterion` interface:
```python
class CustomCriterion(SplitCriterion):
    def evaluate_split(self, left_counts, right_counts, parent_counts):
        # Custom impurity/gain calculation
        return score
```

### 2. **Custom Pruning Methods**
Implement pruning function signature:
```python
def custom_prune(tree_root, validation_data, **kwargs):
    # Custom pruning logic
    return pruned_tree_root
```

### 3. **Custom Logging**
Override TreeLogger methods:
```python
class CustomTreeLogger(TreeLogger):
    def log_split_evaluation(self, feature, score):
        # Send to custom monitoring system
        pass
```

This architecture enables production deployment while maintaining research flexibility through well-defined extension points.