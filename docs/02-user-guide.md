# User Guide - Complete API Reference

## Overview

The Theta Sketch Decision Tree is a scikit-learn compatible classifier with a unique training approach: it learns from **theta sketches** (probabilistic data structures) but makes predictions on **raw binary data**. This design enables privacy-preserving machine learning on large datasets with full sklearn compatibility.

## Architecture Philosophy

- **Training Phase**: Uses theta sketches for memory-efficient learning
- **Inference Phase**: Works with standard binary feature matrices
- **sklearn Compatible**: Drop-in replacement for DecisionTreeClassifier
- **Production Ready**: Full model persistence and validation

## Complete API Usage

### 1. Basic Training and Prediction

```python
from theta_sketch_tree import ThetaSketchDecisionTreeClassifier, load_sketches
import numpy as np

# Load pre-computed sketch data
sketch_data = load_sketches('positive_class.csv', 'negative_class.csv')

# Load feature mapping
import json
with open('feature_mapping.json', 'r') as f:
    feature_mapping = json.load(f)

# Create classifier with advanced parameters
clf = ThetaSketchDecisionTreeClassifier(
    criterion='gini',           # or 'entropy', 'gain_ratio', 'binomial', 'chi_square'
    max_depth=10,               # Maximum tree depth
    min_samples_split=2,        # Minimum samples to split
    min_samples_leaf=1,         # Minimum samples in leaf
    min_impurity_decrease=0.0,  # Minimum impurity decrease for split
    pruning=None,               # or 'cost_complexity', 'validation', 'reduced_error'
    verbose=0                   # Verbosity level (0=silent, 1=progress, 2=detailed)
)

# Train the model
clf.fit(sketch_data, feature_mapping)

# Predict on binary data
X_test = np.array([
    [1, 0, 1, 0],  # Sample 1: features 0,2 present, 1,3 absent
    [0, 1, 0, 1],  # Sample 2: features 1,3 present, 0,2 absent
    [1, 1, 0, 0],  # Sample 3: features 0,1 present, 2,3 absent
])

predictions = clf.predict(X_test)                    # Class predictions: [0, 1, ...]
probabilities = clf.predict_proba(X_test)            # Probability matrix: [[0.2, 0.8], ...]
decision_scores = clf.decision_function(X_test)      # Raw decision scores
```

### 2. One-Step Training Methods

```python
# Method 1: Direct CSV training
clf = ThetaSketchDecisionTreeClassifier.fit_from_csv(
    positive_csv='positive_class.csv',
    negative_csv='negative_class.csv',
    feature_mapping_json='feature_mapping.json',
    criterion='gini',
    max_depth=10,
    pruning='cost_complexity'  # Add pruning during training
)

# Method 2: With configuration file
clf = ThetaSketchDecisionTreeClassifier.fit_from_config(
    config_path='model_config.yaml'
)
```

### 3. Advanced Split Criteria

```python
# Gini Impurity (default, fast)
clf_gini = ThetaSketchDecisionTreeClassifier(criterion='gini')

# Entropy (information gain)
clf_entropy = ThetaSketchDecisionTreeClassifier(criterion='entropy')

# Gain Ratio (handles bias toward multi-valued attributes)
clf_gain_ratio = ThetaSketchDecisionTreeClassifier(criterion='gain_ratio')

# Binomial Test (statistical significance testing)
clf_binomial = ThetaSketchDecisionTreeClassifier(criterion='binomial')

# Chi-Square Test (categorical feature optimization)
clf_chi_square = ThetaSketchDecisionTreeClassifier(criterion='chi_square')
```

### 4. Pruning Strategies

```python
# Cost-Complexity Pruning (recommended for most cases)
clf = ThetaSketchDecisionTreeClassifier(
    pruning='cost_complexity',
    min_impurity_decrease=0.01  # Controls pruning aggressiveness
)

# Validation Pruning (for small datasets)
clf = ThetaSketchDecisionTreeClassifier(
    pruning='validation',
    validation_fraction=0.2  # Use 20% for validation
)

# Reduced-Error Pruning (post-training)
clf = ThetaSketchDecisionTreeClassifier(pruning='reduced_error')

# Manual Impurity Pruning
clf = ThetaSketchDecisionTreeClassifier(
    pruning='min_impurity',
    min_impurity_decrease=0.05  # Remove nodes with low information gain
)
```

### 5. Model Analysis and Interpretability

```python
# Feature importance (weighted impurity decrease)
feature_importance = clf.feature_importances_
print(f"Feature importance array: {feature_importance}")

# Feature importance dictionary with names
importance_dict = clf.get_feature_importance_dict()
print(f"Feature importance: {importance_dict}")

# Top-k most important features
top_features = clf.get_top_features(top_k=5)
print(f"Top 5 features: {top_features}")  # [(name, importance), ...]

# Tree structure analysis
print(f"Tree depth: {clf.get_depth()}")
print(f"Number of nodes: {clf.get_n_nodes()}")
print(f"Number of leaves: {clf.get_n_leaves()}")

# Model metadata
print(f"Classes: {clf.classes_}")
print(f"Number of features: {clf.n_features_}")
print(f"Feature names: {clf.feature_names_}")
```

### 6. Model Persistence

```python
from theta_sketch_tree.model_persistence import ModelPersistence

# Save complete model with metadata
ModelPersistence.save_model(clf, 'my_model.pkl')

# Load model
clf_loaded = ModelPersistence.load_model('my_model.pkl')

# Verify model integrity
original_predictions = clf.predict(X_test)
loaded_predictions = clf_loaded.predict(X_test)
assert np.array_equal(original_predictions, loaded_predictions)
```

### 7. Model Evaluation

```python
from theta_sketch_tree.model_evaluation import ModelEvaluator, evaluate_model
import matplotlib.pyplot as plt

# Create evaluator
evaluator = ModelEvaluator(clf)

# Prepare test data with true labels
y_true = np.array([0, 1, 0, 1, 0])  # True labels
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# Compute metrics
metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"ROC AUC: {metrics['roc_auc']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")

# Plot ROC curve
evaluator.plot_roc_curve(y_true, y_proba)
plt.show()

# Threshold analysis
thresholds_df = evaluator.plot_threshold_analysis(y_true, y_proba)
print(thresholds_df.head())

# One-step evaluation
full_report = evaluate_model(clf, X_test, y_true,
                           save_plots=True, output_dir='./evaluation_results/')
```

## Data Format Specifications

### Required Input Files

#### 1. Sketch CSV Files

**Dual-Class Mode** (recommended for balanced datasets):
```
positive_class.csv    # Sketches from positive class samples
negative_class.csv    # Sketches from negative class samples
```

**One-vs-All Mode** (recommended for imbalanced datasets):
```
positive_class.csv    # Sketches from positive class samples
total_population.csv  # Sketches from ENTIRE dataset (all classes)
```

#### 2. CSV Format Structure

```csv
feature_name,present_sketch,absent_sketch
age>30,<base64_encoded_sketch_1>,<base64_encoded_sketch_2>
income>50k,<base64_encoded_sketch_3>,<base64_encoded_sketch_4>
education=graduate,<base64_encoded_sketch_5>,<base64_encoded_sketch_6>
```

**Critical Requirements:**
- Column names must be exactly: `feature_name`, `present_sketch`, `absent_sketch`
- Sketches must be base64-encoded theta sketch objects
- All sketches must use the same `lg_k` parameter
- Feature names must match between positive and negative CSV files

#### 3. Feature Mapping JSON

```json
{
  "age>30": 0,
  "income>50k": 1,
  "education=graduate": 2,
  "marital_status=married": 3
}
```

#### 4. Configuration YAML (Optional)

```yaml
model_parameters:
  criterion: "gini"
  max_depth: 10
  min_samples_split: 2
  min_samples_leaf: 1
  pruning: "cost_complexity"
  min_impurity_decrease: 0.01
  verbose: 1

data_sources:
  positive_csv: "positive_class.csv"
  negative_csv: "negative_class.csv"
  feature_mapping_json: "feature_mapping.json"

training_parameters:
  lg_k: 12
  sample_validation: true
```

### Prediction Data Format

```python
# Binary feature matrix: rows=samples, cols=features
X_test = np.array([
    [1, 0, 1, 0],  # Sample 1: features [0,2] present, [1,3] absent
    [0, 1, 0, 1],  # Sample 2: features [1,3] present, [0,2] absent
    [1, 1, 1, 1],  # Sample 3: all features present
    [0, 0, 0, 0],  # Sample 4: all features absent
], dtype=int)

# Each column corresponds to a feature in feature_mapping
# Values must be 0 (absent) or 1 (present)
```

## Advanced Usage Patterns

### 1. Cross-Validation

```python
from sklearn.model_selection import cross_val_score
from theta_sketch_tree.validation import SketchDataSplitter

# Custom splitter for sketch data
splitter = SketchDataSplitter(n_splits=5, shuffle=True, random_state=42)

# Cross-validation scores
cv_scores = cross_val_score(clf, sketch_data, feature_mapping,
                          cv=splitter, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

### 2. Hyperparameter Optimization

```python
from sklearn.model_selection import GridSearchCV
from theta_sketch_tree.hyperparameter_tuning import SketchTreeSearchCV

# Define parameter grid
param_grid = {
    'criterion': ['gini', 'entropy', 'gain_ratio'],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_impurity_decrease': [0.0, 0.01, 0.05],
    'pruning': [None, 'cost_complexity', 'validation']
}

# Optimize hyperparameters
search = SketchTreeSearchCV(
    ThetaSketchDecisionTreeClassifier(),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

search.fit(sketch_data, feature_mapping)
best_clf = search.best_estimator_

print(f"Best parameters: {search.best_params_}")
print(f"Best CV score: {search.best_score_:.3f}")
```

### 3. Ensemble Methods

```python
from sklearn.ensemble import VotingClassifier
from theta_sketch_tree import ThetaSketchDecisionTreeClassifier

# Create ensemble of different criteria
estimators = [
    ('gini_tree', ThetaSketchDecisionTreeClassifier(criterion='gini', max_depth=10)),
    ('entropy_tree', ThetaSketchDecisionTreeClassifier(criterion='entropy', max_depth=10)),
    ('gain_ratio_tree', ThetaSketchDecisionTreeClassifier(criterion='gain_ratio', max_depth=8))
]

# Voting classifier
ensemble = VotingClassifier(estimators, voting='soft')

# Train each estimator (requires custom training loop)
for name, estimator in estimators:
    estimator.fit(sketch_data, feature_mapping)

# Make ensemble predictions
predictions = ensemble.predict(X_test)
```

### 4. Missing Value Handling

```python
# Missing values represented as -1 in binary data
X_with_missing = np.array([
    [1, 0, -1, 0],  # Missing value in feature 2
    [0, -1, 1, 1],  # Missing value in feature 1
    [1, 1, 0, -1],  # Missing value in feature 3
])

# Classifier handles missing values with majority vote
predictions = clf.predict(X_with_missing)
probabilities = clf.predict_proba(X_with_missing)
```

## Command Line Interface

### Basic Training

```bash
# Train on CSV data
python run_binary_classification.py dataset.csv target_column \
    --lg_k 12 \
    --max_depth 10 \
    --criterion gini \
    --output_model model.pkl \
    --verbose 1
```

### Advanced Training

```bash
# Training with pruning and validation
python run_binary_classification.py dataset.csv target_column \
    --lg_k 14 \
    --max_depth 15 \
    --criterion entropy \
    --pruning cost_complexity \
    --min_impurity_decrease 0.01 \
    --validation_split 0.2 \
    --sample_size 10000 \
    --verbose 2 \
    --save_evaluation \
    --output_dir ./results/
```

### Batch Processing

```bash
# Process multiple datasets
python tools/batch_trainer.py \
    --input_dir ./datasets/ \
    --output_dir ./models/ \
    --config config.yaml \
    --parallel 4
```

## Performance Optimization

### Memory Efficiency

```python
# For large datasets, use streaming prediction
def predict_in_batches(clf, X, batch_size=10000):
    """Predict large datasets in memory-efficient batches."""
    n_samples = X.shape[0]
    predictions = np.empty(n_samples, dtype=int)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_X = X[start_idx:end_idx]
        predictions[start_idx:end_idx] = clf.predict(batch_X)

    return predictions

# Usage for large datasets
large_predictions = predict_in_batches(clf, X_large, batch_size=5000)
```

### Speed Optimization

```python
# Pre-compile for repeated predictions
clf.prepare_fast_prediction()  # Optional pre-compilation step

# Vectorized prediction for maximum speed
predictions = clf.predict(X_test)  # Optimized for batch processing
```

## Error Handling and Validation

### Input Validation

```python
from theta_sketch_tree.validation_utils import validate_sketch_data, validate_feature_mapping

# Validate sketch data structure
try:
    validate_sketch_data(sketch_data)
    validate_feature_mapping(feature_mapping, sketch_data)
    print("✅ Data validation passed")
except ValueError as e:
    print(f"❌ Data validation failed: {e}")
```

### Model Health Checks

```python
# Check model state
assert clf.is_fitted(), "Model must be fitted before prediction"

# Validate prediction input
def safe_predict(clf, X):
    """Predict with comprehensive input validation."""
    try:
        # Basic shape validation
        assert X.ndim == 2, f"Expected 2D array, got {X.ndim}D"
        assert X.shape[1] == clf.n_features_, f"Expected {clf.n_features_} features, got {X.shape[1]}"

        # Value range validation
        assert np.all((X >= -1) & (X <= 1)), "Values must be -1 (missing), 0 (absent), or 1 (present)"

        return clf.predict(X)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

# Safe prediction with validation
predictions = safe_predict(clf, X_test)
```

## Best Practices

### 1. Data Preparation
- Ensure consistent `lg_k` values across all sketches
- Validate sketch quality before training
- Use appropriate sketch size for your data scale
- Pre-compute sketches from your big data pipeline

### 2. Model Configuration
- Start with `criterion='gini'` for speed, try others for accuracy
- Use `pruning='cost_complexity'` for better generalization
- Set `max_depth` based on your data complexity
- Enable `verbose=1` for training progress monitoring

### 3. Performance Tuning
- Use batch prediction for large datasets
- Enable model preparation for repeated inference
- Monitor memory usage with large feature sets
- Consider feature selection for high-dimensional data

### 4. Production Deployment
- Always save feature mapping with the model
- Implement input validation in production pipelines
- Monitor prediction latency and accuracy over time
- Use model versioning for reproducibility

## Troubleshooting

### Common Issues

**1. Sketch Format Errors**
```python
# Problem: "Sketch deserialization failed"
# Solution: Verify lg_k consistency and base64 encoding
```

**2. Feature Mapping Mismatches**
```python
# Problem: "Feature mapping size mismatch"
# Solution: Ensure mapping covers all features in sketch data
```

**3. Prediction Shape Errors**
```python
# Problem: "Input shape mismatch"
# Solution: Verify X_test has correct number of features
```

**4. Memory Issues**
```python
# Problem: Out of memory during prediction
# Solution: Use batch prediction or reduce feature dimensions
```

For more troubleshooting help, see [Troubleshooting Guide](08-troubleshooting.md).

---

## Next Steps

- **Architecture**: Read [Architecture Overview](03-architecture.md) for system design
- **Algorithms**: See [Algorithm Reference](04-algorithms.md) for mathematical details
- **Performance**: Check [Performance Guide](06-performance.md) for optimization
- **Deployment**: Review [Deployment Guide](11-deployment.md) for production setup