# API Reference

## Core Classes

### ThetaSketchDecisionTreeClassifier

The main classifier implementing sklearn's estimator interface for training on theta sketches and inference on binary data.

```python
class ThetaSketchDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    CART Decision Tree trained on theta sketches with binary data inference.

    Provides sklearn compatibility while enabling memory-efficient training
    on large datasets through probabilistic sketch data structures.
    """
```

#### Constructor Parameters

```python
def __init__(
    self,
    criterion: str = 'gini',
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    min_impurity_decrease: float = 0.0,
    pruning: Optional[str] = None,
    validation_fraction: float = 0.2,
    class_weight: Optional[Union[str, dict]] = None,
    missing_value_strategy: str = 'majority',
    verbose: int = 0,
    random_state: Optional[int] = None
)
```

**Parameters:**

- **criterion** *(str, default='gini')*: Split criterion
  - `'gini'`: Gini impurity (fastest)
  - `'entropy'`: Information gain
  - `'gain_ratio'`: Gain ratio (handles multi-valued bias)
  - `'binomial'`: Binomial statistical test
  - `'chi_square'`: Chi-square statistical test

- **max_depth** *(int or None, default=None)*: Maximum tree depth
  - `None`: No depth limit
  - `int`: Specific depth limit

- **min_samples_split** *(int, default=2)*: Minimum samples to split node

- **min_samples_leaf** *(int, default=1)*: Minimum samples in leaf node

- **min_impurity_decrease** *(float, default=0.0)*: Minimum impurity decrease for split

- **pruning** *(str or None, default=None)*: Post-training pruning method
  - `None`: No pruning
  - `'cost_complexity'`: Cost-complexity pruning (recommended)
  - `'validation'`: Validation set pruning
  - `'reduced_error'`: Reduced error pruning
  - `'min_impurity'`: Minimum impurity pruning

- **validation_fraction** *(float, default=0.2)*: Fraction for validation pruning

- **class_weight** *(str, dict or None, default=None)*: Class balancing
  - `None`: No weighting
  - `'balanced'`: Inverse frequency weighting
  - `dict`: Custom class weights

- **missing_value_strategy** *(str, default='majority')*: Missing value handling
  - `'majority'`: Follow majority path in tree
  - `'zero'`: Treat as feature absent (0)
  - `'error'`: Raise error on missing values

- **verbose** *(int, default=0)*: Verbosity level
  - `0`: Silent
  - `1`: Progress updates
  - `2`: Detailed logging

- **random_state** *(int or None, default=None)*: Random seed for reproducibility

#### Core Methods

##### Training Methods

```python
def fit(self, sketch_data: Dict[str, Any], feature_mapping: Dict[str, int]) -> 'ThetaSketchDecisionTreeClassifier'
```
**Train classifier on theta sketch data.**

**Parameters:**
- `sketch_data`: Dictionary containing positive/negative or positive/total sketches
- `feature_mapping`: Mapping from feature names to column indices

**Returns:** Fitted classifier instance

**Example:**
```python
sketch_data = {
    'positive': {
        'total': positive_total_sketch,
        'feature1': (present_sketch, absent_sketch),
        'feature2': (present_sketch, absent_sketch)
    },
    'negative': {
        'total': negative_total_sketch,
        'feature1': (present_sketch, absent_sketch),
        'feature2': (present_sketch, absent_sketch)
    }
}

feature_mapping = {'feature1': 0, 'feature2': 1}
clf = ThetaSketchDecisionTreeClassifier()
clf.fit(sketch_data, feature_mapping)
```

```python
@classmethod
def fit_from_csv(
    cls,
    positive_csv: str,
    negative_csv: str,
    feature_mapping_json: str,
    **kwargs
) -> 'ThetaSketchDecisionTreeClassifier'
```
**Direct training from CSV files.**

**Parameters:**
- `positive_csv`: Path to positive class sketches CSV
- `negative_csv`: Path to negative class sketches CSV
- `feature_mapping_json`: Path to feature mapping JSON file
- `**kwargs`: Classifier parameters

**Returns:** Fitted classifier instance

```python
@classmethod
def fit_from_config(cls, config_path: str) -> 'ThetaSketchDecisionTreeClassifier'
```
**Training from YAML configuration file.**

##### Prediction Methods

```python
def predict(self, X: np.ndarray) -> np.ndarray
```
**Predict class labels for binary input data.**

**Parameters:**
- `X`: Binary feature matrix (n_samples, n_features), values in {0, 1, -1}
  - `0`: Feature absent
  - `1`: Feature present
  - `-1`: Missing value

**Returns:** Class labels array (n_samples,)

```python
def predict_proba(self, X: np.ndarray) -> np.ndarray
```
**Predict class probabilities.**

**Returns:** Probability matrix (n_samples, n_classes)

```python
def decision_function(self, X: np.ndarray) -> np.ndarray
```
**Compute decision function values.**

**Returns:** Decision scores (n_samples,)

##### Analysis Methods

```python
@property
def feature_importances_(self) -> np.ndarray
```
**Feature importance scores (weighted impurity decrease).**

**Returns:** Importance array (n_features,) summing to 1.0

```python
def get_feature_importance_dict(self) -> Dict[str, float]
```
**Feature importance as name-value dictionary.**

**Returns:** Dictionary mapping feature names to importance scores

```python
def get_top_features(self, top_k: int = 10) -> List[Tuple[str, float]]
```
**Get top-k most important features.**

**Parameters:**
- `top_k`: Number of top features to return

**Returns:** List of (feature_name, importance) tuples, sorted by importance

```python
def get_depth(self) -> int
```
**Get maximum depth of trained tree.**

```python
def get_n_nodes(self) -> int
```
**Get total number of nodes in tree.**

```python
def get_n_leaves(self) -> int
```
**Get number of leaf nodes in tree.**

#### Properties

```python
@property
def classes_(self) -> np.ndarray
```
**Classes seen during fit.**

```python
@property
def n_features_(self) -> int
```
**Number of features seen during fit.**

```python
@property
def feature_names_(self) -> List[str]
```
**Feature names seen during fit.**

```python
@property
def tree_(self) -> TreeNode
```
**Access to underlying tree structure.**

```python
def is_fitted(self) -> bool
```
**Check if classifier has been fitted.**

## Data Loading Functions

### Sketch Loading

```python
def load_sketches(
    positive_csv: str,
    negative_csv: str,
    lg_k: int = 12
) -> Dict[str, Any]
```
**Load sketches from CSV files.**

**Parameters:**
- `positive_csv`: Path to positive class sketches
- `negative_csv`: Path to negative/total class sketches
- `lg_k`: Sketch parameter (must match CSV data)

**Returns:** Sketch data dictionary for training

### Configuration Loading

```python
def load_config(config_path: str) -> Dict[str, Any]
```
**Load configuration from YAML file.**

**Expected YAML structure:**
```yaml
model_parameters:
  criterion: "gini"
  max_depth: 10
  min_samples_split: 2

data_sources:
  positive_csv: "positive_class.csv"
  negative_csv: "negative_class.csv"
  feature_mapping_json: "mapping.json"

training_parameters:
  lg_k: 12
  verbose: 1
```

## Model Persistence

### ModelPersistence Class

```python
class ModelPersistence:
    """Handle model serialization and deserialization."""

    @staticmethod
    def save_model(clf: ThetaSketchDecisionTreeClassifier, filepath: str) -> None:
        """Save complete model with metadata."""

    @staticmethod
    def load_model(filepath: str) -> ThetaSketchDecisionTreeClassifier:
        """Load model with integrity validation."""

    @staticmethod
    def get_model_info(filepath: str) -> Dict[str, Any]:
        """Get model metadata without loading full model."""
```

**Example:**
```python
from theta_sketch_tree.model_persistence import ModelPersistence

# Save trained model
ModelPersistence.save_model(clf, 'my_model.pkl')

# Load model
clf_loaded = ModelPersistence.load_model('my_model.pkl')

# Check model info
info = ModelPersistence.get_model_info('my_model.pkl')
print(f"Model info: {info}")
```

## Model Evaluation

### ModelEvaluator Class

```python
class ModelEvaluator:
    """Comprehensive model evaluation and visualization."""

    def __init__(self, classifier: ThetaSketchDecisionTreeClassifier):
        """Initialize with fitted classifier."""

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """Plot ROC curve with AUC score."""

    def plot_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> pd.DataFrame:
        """Analyze performance across decision thresholds."""
```

### Evaluation Function

```python
def evaluate_model(
    classifier: ThetaSketchDecisionTreeClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_plots: bool = False,
    output_dir: str = './'
) -> Dict[str, Any]:
    """One-step comprehensive model evaluation."""
```

**Returns comprehensive evaluation report including:**
- Accuracy, Precision, Recall, F1-Score
- ROC AUC score
- Confusion matrix
- Feature importance analysis
- ROC curve plot (optional)
- Threshold analysis (optional)

## Validation Utilities

### Input Validation

```python
def validate_sketch_data(sketch_data: Dict[str, Any]) -> None:
    """Validate sketch data structure and integrity."""

def validate_feature_mapping(
    feature_mapping: Dict[str, int],
    sketch_data: Dict[str, Any]
) -> None:
    """Validate feature mapping consistency."""

def validate_binary_input(X: np.ndarray) -> None:
    """Validate binary input matrix for prediction."""
```

### Data Validation

```python
def validate_csv_format(csv_path: str) -> bool:
    """Validate CSV sketch file format."""

def validate_sketch_consistency(
    positive_csv: str,
    negative_csv: str,
    lg_k: int
) -> bool:
    """Validate sketch parameter consistency."""
```

## Command Line Interface

### Training Script

```bash
python run_binary_classification.py <dataset.csv> <target_column> [options]
```

**Options:**
- `--lg_k`: Sketch parameter (default: 12)
- `--max_depth`: Maximum tree depth
- `--criterion`: Split criterion
- `--pruning`: Pruning method
- `--output_model`: Model save path
- `--verbose`: Verbosity level
- `--sample_size`: Sample subset for training
- `--validation_split`: Validation fraction
- `--save_evaluation`: Save evaluation plots

**Example:**
```bash
python run_binary_classification.py data.csv target \
    --lg_k 14 \
    --max_depth 10 \
    --criterion entropy \
    --pruning cost_complexity \
    --output_model model.pkl \
    --verbose 1 \
    --save_evaluation
```

### Batch Processing

```bash
python tools/batch_trainer.py \
    --input_dir ./datasets/ \
    --output_dir ./models/ \
    --config config.yaml \
    --parallel 4
```

## Error Handling

### Exception Classes

```python
class SketchDataError(ValueError):
    """Raised for invalid sketch data structure."""

class FeatureMappingError(ValueError):
    """Raised for invalid feature mapping."""

class ModelNotFittedError(ValueError):
    """Raised when using unfitted model."""

class InvalidInputError(ValueError):
    """Raised for invalid prediction input."""
```

### Common Error Patterns

```python
# Handling sketch data errors
try:
    clf.fit(sketch_data, feature_mapping)
except SketchDataError as e:
    print(f"Sketch data error: {e}")

# Handling prediction errors
try:
    predictions = clf.predict(X_test)
except ModelNotFittedError:
    print("Model must be fitted before prediction")
except InvalidInputError as e:
    print(f"Invalid input: {e}")
```

## Type Hints

### Core Type Definitions

```python
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np

# Sketch data structure
SketchData = Dict[str, Dict[str, Union[Any, Tuple[Any, Any]]]]

# Feature mapping
FeatureMapping = Dict[str, int]

# Model parameters
ModelParams = Dict[str, Union[str, int, float, bool, None]]

# Evaluation metrics
Metrics = Dict[str, float]

# Feature importance
FeatureImportance = List[Tuple[str, float]]
```

## Performance Considerations

### Memory Efficiency

```python
# For large datasets, use batch prediction
def predict_large_dataset(clf, X_large, batch_size=10000):
    """Memory-efficient prediction for large datasets."""
    n_samples = X_large.shape[0]
    predictions = np.empty(n_samples, dtype=int)

    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_pred = clf.predict(X_large[i:end_idx])
        predictions[i:end_idx] = batch_pred

    return predictions
```

### Speed Optimization

```python
# Pre-compile for repeated predictions (if available)
clf.prepare_fast_prediction()  # Optional optimization

# Vectorized batch processing
predictions = clf.predict(X_batch)  # Optimized for batches
```

## Integration Examples

### sklearn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Note: ThetaSketch classifier expects binary input
pipeline = Pipeline([
    ('classifier', ThetaSketchDecisionTreeClassifier(criterion='gini'))
])

# Custom training required for sketch data
# pipeline.fit(X, y)  # Not directly supported
# Instead:
clf = ThetaSketchDecisionTreeClassifier()
clf.fit(sketch_data, feature_mapping)

# Then use for prediction in pipeline context
predictions = clf.predict(X_binary)
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# Custom scoring function for sketch-based training
def sketch_cross_val_score(sketch_data, feature_mapping, cv=5):
    """Custom cross-validation for sketch data."""
    scores = []

    for train_idx, val_idx in cv_splitter.split(sketch_data):
        # Custom splitting logic for sketch data
        train_sketches = split_sketch_data(sketch_data, train_idx)

        clf = ThetaSketchDecisionTreeClassifier()
        clf.fit(train_sketches, feature_mapping)

        # Evaluate on validation set
        score = evaluate_on_validation(clf, val_idx)
        scores.append(score)

    return np.array(scores)
```

---

## Next Steps

- **User Guide**: See [User Guide](02-user-guide.md) for comprehensive examples
- **Performance**: Review [Performance Guide](06-performance.md) for optimization
- **Testing**: Check [Testing Guide](07-testing.md) for validation strategies
- **Troubleshooting**: See [Troubleshooting Guide](08-troubleshooting.md) for common issues