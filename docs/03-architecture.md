# Architecture Overview

## System Design Philosophy

The Theta Sketch Decision Tree classifier implements a unique **dual-phase architecture**: it trains on probabilistic data structures (theta sketches) for memory efficiency but performs inference on standard binary data for sklearn compatibility.

### Core Design Principles

1. **Separation of Concerns**: Distinct training and inference subsystems
2. **Sketch Pre-computation**: All sketches computed upstream, no runtime set operations
3. **sklearn Compatibility**: Drop-in replacement for DecisionTreeClassifier
4. **Performance by Design**: Cache-friendly traversal and vectorized operations
5. **Fail-Fast Validation**: Comprehensive input checking at API boundaries

## Critical Architectural Decision

### Sketch Pre-Computation Pattern

**All sketches are pre-computed directly from raw big data sources**. The training phase loads these pre-computed sketches from CSV files and uses ONLY intersection operations during tree building.

**Key Implementation Details:**
- CSV files contain sketches computed from the original big data pipeline
- For One-vs-All mode: `total.csv` contains sketches of the ENTIRE dataset (unfiltered)
- Loader performs NO set operations - it simply reads pre-computed sketches
- Negative class counts: Arithmetic subtraction at numeric level (`n_neg = n_total - n_pos`)

**Design Rationale:** This pattern ensures deterministic behavior, eliminates sketch operation errors, and maintains data lineage from big data sources.

## System Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PHASE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  CSV File Input                    Configuration                  │
│  ┌──────────────────┐             ┌──────────────────┐          │
│  │ positive.csv     │             │ Model Parameters │          │
│  │ negative.csv     │             │ Feature Mapping  │          │
│  │ (or total.csv)   │             │ Training Config  │          │
│  └────────┬─────────┘             └────────┬─────────┘          │
│           │                                │                    │
│           ▼                                ▼                    │
│  ┌─────────────────────────────────────────────────┐            │
│  │            SketchLoader                         │            │
│  │  • Parse CSV sketches                           │            │
│  │  • Validate sketch integrity                    │            │
│  │  • Load feature mappings                        │            │
│  └────────────────────┬────────────────────────────┘            │
│                       │                                          │
│                       ▼                                          │
│  ┌─────────────────────────────────────────────────┐            │
│  │    ThetaSketchDecisionTreeClassifier            │            │
│  │                                                  │            │
│  │  ┌─────────────────────────────────────┐        │            │
│  │  │        TreeBuilder                  │        │            │
│  │  │  • Recursive tree construction      │        │            │
│  │  │  • Stopping criteria evaluation     │        │            │
│  │  │  • Node metadata management         │        │            │
│  │  └──────────┬──────────────────────────┘        │            │
│  │             │                                    │            │
│  │             ▼                                    │            │
│  │  ┌─────────────────────────────────────┐        │            │
│  │  │        SplitFinder                  │        │            │
│  │  │  • Evaluate all feature splits      │        │            │
│  │  │  • Compute sketch intersections     │        │            │
│  │  │  • Apply split criteria             │        │            │
│  │  │  • Cache expensive operations       │        │            │
│  │  └──────────┬──────────────────────────┘        │            │
│  │             │                                    │            │
│  │             ▼                                    │            │
│  │  ┌─────────────────────────────────────┐        │            │
│  │  │     Split Criteria (Pluggable)      │        │            │
│  │  │  • Gini, Entropy, Gain Ratio        │        │            │
│  │  │  • Binomial, Chi-Square             │        │            │
│  │  └─────────────────────────────────────┘        │            │
│  └─────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE PHASE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Binary Data Input                 Trained Model                 │
│  ┌──────────────────┐             ┌──────────────────┐          │
│  │ X = [[1,0,1,0],  │             │ TreeStructure    │          │
│  │      [0,1,0,1],  │             │ FeatureMapping   │          │
│  │      [1,1,1,1]]  │             │ ModelMetadata    │          │
│  └────────┬─────────┘             └────────┬─────────┘          │
│           │                                │                    │
│           ▼                                ▼                    │
│  ┌─────────────────────────────────────────────────┐            │
│  │           TreeTraverser                         │            │
│  │  • Input validation                             │            │
│  │  • Missing value handling                       │            │
│  │  • Vectorized tree traversal                    │            │
│  │  • Probability computation                      │            │
│  └─────────────────────┬───────────────────────────┘            │
│                        │                                         │
│                        ▼                                         │
│  ┌─────────────────────────────────────────────────┐            │
│  │             Output                               │            │
│  │  • Class predictions: [0, 1, 0]                 │            │
│  │  • Probabilities: [[0.8,0.2], [0.3,0.7], ...]  │            │
│  │  • Decision scores: [-0.5, 0.3, -0.1]           │            │
│  └─────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

#### 1. Training Subsystem

**SketchLoader** (`sketch_loader.py`)
- Parses CSV files containing theta sketches
- Validates sketch format and integrity
- Handles dual-class and one-vs-all modes
- Loads feature mappings and configurations

**TreeBuilder** (`tree_builder.py`)
- Implements recursive CART algorithm
- Manages tree structure and node metadata
- Applies stopping criteria (depth, samples, impurity)
- Integrates with pruning strategies

**SplitFinder** (`split_finder.py`)
- Evaluates all possible feature splits
- Computes sketch intersection cardinalities
- Applies split criteria for best split selection
- Caches expensive sketch operations

**Split Criteria** (`criteria.py`)
- Pluggable criterion implementations
- Gini, Entropy, Gain Ratio, Binomial, Chi-Square
- Mathematical correctness validation
- Performance-optimized calculations

#### 2. Inference Subsystem

**TreeTraverser** (`tree_traverser.py`)
- Validates binary input format
- Handles missing values (-1) with majority vote
- Vectorized tree traversal for batch prediction
- Computes class probabilities and decision scores

**FeatureImportance** (`feature_importance.py`)
- Calculates weighted impurity decrease
- Provides feature ranking and analysis
- Supports top-k feature selection
- Integrates with sklearn feature importance API

#### 3. Supporting Subsystems

**ModelPersistence** (`model_persistence.py`)
- Complete model serialization/deserialization
- Preserves tree structure and metadata
- Version compatibility checking
- Integrity validation on load

**ValidationUtils** (`validation_utils.py`)
- Comprehensive input validation
- Type checking and range validation
- Error message generation with remediation
- Performance-optimized validation routines

**ModelEvaluation** (`model_evaluation.py`)
- ROC curve analysis and plotting
- Comprehensive metric computation
- Threshold analysis and optimization
- Cross-validation utilities

## Data Flow Architecture

### Training Data Flow

```
Raw Big Data → Sketch Computation → CSV Files → SketchLoader → TreeBuilder → Trained Model
                    (Upstream)         (Input)      (Parse)      (Train)     (Output)
```

**Key Points:**
- Sketch computation happens upstream in big data pipeline
- No runtime sketch operations during training
- Deterministic results from pre-computed sketches

### Inference Data Flow

```
Binary Matrix → Input Validation → TreeTraverser → Predictions/Probabilities
    (Input)         (Validate)        (Process)        (Output)
```

**Key Points:**
- Standard numpy arrays as input
- Full sklearn compatibility
- High-performance vectorized operations

## Design Patterns

### 1. Strategy Pattern - Split Criteria

```python
class SplitCriterion(ABC):
    @abstractmethod
    def evaluate_split(self, parent_counts, left_counts, right_counts):
        pass

class GiniCriterion(SplitCriterion):
    def evaluate_split(self, parent_counts, left_counts, right_counts):
        # Gini-specific implementation
        pass
```

**Benefits:**
- Easy addition of new criteria
- Runtime criterion selection
- Isolated testing per criterion

### 2. Template Method - Tree Building

```python
class TreeBuilder:
    def build_tree(self, sketch_data, feature_mapping):
        root = self._create_root_node(sketch_data)
        self._recursive_build(root, sketch_data, feature_mapping, depth=0)
        return root

    def _recursive_build(self, node, sketches, mapping, depth):
        # Template method with hooks for customization
        if self._should_stop(node, depth):
            self._make_leaf(node)
            return

        best_split = self._find_best_split(node, sketches, mapping)
        if best_split is None:
            self._make_leaf(node)
            return

        self._apply_split(node, best_split)
        # Recursive calls for children...
```

### 3. Facade Pattern - Main Classifier

```python
class ThetaSketchDecisionTreeClassifier:
    def __init__(self, **kwargs):
        self._tree_builder = TreeBuilder()
        self._tree_traverser = TreeTraverser()
        self._validator = ValidationUtils()

    def fit(self, sketch_data, feature_mapping):
        # Coordinates all subsystems through simple interface
        self._validator.validate_sketch_data(sketch_data)
        self.tree_ = self._tree_builder.build_tree(sketch_data, feature_mapping)

    def predict(self, X):
        # Simple interface hiding complex traversal logic
        return self._tree_traverser.predict(X, self.tree_)
```

## Performance Architecture

### Memory Optimization

1. **Read-Only Sketches**: No sketch copying during training
2. **Lazy Loading**: Tree nodes created on-demand
3. **Cache-Friendly Traversal**: Breadth-first node layout
4. **Batch Processing**: Vectorized operations for inference

### Computational Optimization

1. **Sketch Intersection Caching**: Expensive operations cached per split
2. **Early Stopping**: Multiple stopping criteria to prevent overcomputation
3. **Vectorized Prediction**: Batch traversal for multiple samples
4. **JIT Compilation**: NumPy operations optimized by runtime

### Scalability Design

1. **Linear Memory Growth**: O(n_features + n_nodes) memory complexity
2. **Efficient Tree Depth**: Logarithmic prediction time complexity
3. **Streaming Support**: Batch prediction for large datasets
4. **Parallel Processing**: Thread-safe inference operations

## Security and Reliability

### Input Validation
- Comprehensive type and range checking
- Malformed sketch detection
- Feature mapping validation
- Binary input format enforcement

### Error Handling
- Graceful degradation for edge cases
- Informative error messages with remediation
- Logging for debugging and monitoring
- Recovery strategies for common failures

### Testing Architecture
- Unit tests for individual components
- Integration tests for end-to-end workflows
- Performance regression tests
- Property-based testing for mathematical correctness

## Extension Points

### Adding New Split Criteria

1. Inherit from `SplitCriterion` abstract base class
2. Implement `evaluate_split()` method
3. Add criterion to factory mapping
4. Write comprehensive unit tests

### Adding New Pruning Methods

1. Inherit from `PruningStrategy` abstract base class
2. Implement pruning logic
3. Integrate with TreeBuilder
4. Add configuration parameters

### Adding New Data Sources

1. Implement `SketchDataLoader` interface
2. Handle format-specific parsing
3. Ensure sketch validation
4. Add comprehensive error handling

## Future Architecture Considerations

### Planned Enhancements

1. **Distributed Training**: Multi-node sketch processing
2. **GPU Acceleration**: CUDA-optimized sketch operations
3. **Online Learning**: Incremental tree updates
4. **Multi-class Support**: Extension beyond binary classification

### Scalability Roadmap

1. **Horizontal Scaling**: Distributed inference servers
2. **Vertical Scaling**: Memory-mapped tree structures
3. **Edge Deployment**: Lightweight inference engines
4. **Cloud Integration**: Serverless prediction services

---

## Next Steps

- **Algorithm Details**: See [Algorithm Reference](04-algorithms.md) for mathematical foundations
- **Implementation**: Check [API Reference](05-api-reference.md) for detailed method specifications
- **Performance**: Review [Performance Guide](06-performance.md) for optimization strategies
- **Testing**: Read [Testing Guide](07-testing.md) for validation approaches