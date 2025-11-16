# Theta Sketch Decision Tree - User Guide

## Overview

The Theta Sketch Decision Tree is a scikit-learn compatible classifier that trains on probabilistic data structures (theta sketches) but performs inference on binary tabular data. This unique approach enables decision tree learning from large-scale datasets using approximate cardinality estimates.

## Quick Start

### Basic Usage

```python
from theta_sketch_tree import ThetaSketchDecisionTreeClassifier
import numpy as np

# Load your data (see data format section below)
from theta_sketch_tree import load_sketches, load_config

# Method 1: Load from CSV files
sketch_data = load_sketches('positive_class.csv', 'negative_class.csv')
config = load_config('config.yaml')

# Create and train classifier
clf = ThetaSketchDecisionTreeClassifier(
    criterion='gini',
    max_depth=10,
    min_samples_split=2
)

clf.fit(sketch_data, config['feature_mapping'])

# Make predictions on binary data
X_test = np.array([
    [1, 0, 1],  # feature1=present, feature2=absent, feature3=present
    [0, 1, 0],  # feature1=absent, feature2=present, feature3=absent
    [1, 1, 1]   # all features present
])

predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

print(f"Predictions: {predictions}")
print(f"Probabilities: {probabilities}")
```

### One-Step Training

```python
# Method 2: One-step training from files
clf = ThetaSketchDecisionTreeClassifier.fit_from_csv(
    positive_csv='positive_class.csv',
    negative_csv='negative_class.csv',
    config_path='config.yaml'
)

predictions = clf.predict(X_test)
```

## Data Format

### Sketch Data Structure

The classifier expects sketch data in a specific nested dictionary format:

```python
sketch_data = {
    'positive': {
        'total': theta_sketch_positive_total,
        'feature1': (sketch_feature1_present, sketch_feature1_absent),
        'feature2': (sketch_feature2_present, sketch_feature2_absent),
        # ... more features
    },
    'negative': {
        'total': theta_sketch_negative_total,
        'feature1': (sketch_feature1_present, sketch_feature1_absent),
        'feature2': (sketch_feature2_present, sketch_feature2_absent),
        # ... more features
    }
}
```

### Feature Mapping

```python
feature_mapping = {
    'feature1': 0,  # Maps to column 0 in X_test
    'feature2': 1,  # Maps to column 1 in X_test
    'feature3': 2,  # Maps to column 2 in X_test
}
```

### CSV Format

#### Positive Class CSV (`positive_class.csv`)
```csv
sketch_name,sketch_type,estimated_cardinality
total,total,10000
feature1_present,present,7000
feature1_absent,absent,3000
feature2_present,present,6000
feature2_absent,absent,4000
```

#### Configuration YAML (`config.yaml`)
```yaml
feature_mapping:
  feature1: 0
  feature2: 1

hyperparameters:
  criterion: gini
  max_depth: 10
  min_samples_split: 2
```

## Advanced Usage

### Feature Importance Analysis

```python
# Get feature importances as array
importances = clf.feature_importances_
print(f"Feature importances: {importances}")

# Get as dictionary with feature names
importance_dict = clf.get_feature_importance_dict()
print(f"Feature importance dict: {importance_dict}")

# Get top K most important features
top_features = clf.get_top_features(top_k=5)
print(f"Top 5 features: {top_features}")
```

### Split Criteria Options

```python
# Different split criteria available
clf_gini = ThetaSketchDecisionTreeClassifier(criterion='gini')        # Default, fastest
clf_entropy = ThetaSketchDecisionTreeClassifier(criterion='entropy')  # Information gain
clf_gain_ratio = ThetaSketchDecisionTreeClassifier(criterion='gain_ratio')  # Gain ratio
clf_binomial = ThetaSketchDecisionTreeClassifier(criterion='binomial')     # Binomial test
clf_chi2 = ThetaSketchDecisionTreeClassifier(criterion='binomial_chi')     # Chi-square test
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

# Use GridSearchCV (requires sklearn-compatible evaluation)
grid_search = GridSearchCV(
    ThetaSketchDecisionTreeClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)

# Note: This requires binary training data for CV
# grid_search.fit(X_train, y_train)
```

### Custom Missing Value Handling

```python
from theta_sketch_tree.tree_traverser import TreeTraverser

# Train model
clf.fit(sketch_data, feature_mapping)

# Create custom traverser with different missing value strategy
traverser = TreeTraverser(clf.tree_, missing_value_strategy='zero')

# Handle missing values as False/absent
X_with_missing = np.array([
    [1, np.nan, 1],  # Second feature missing
    [np.nan, 1, 0],  # First feature missing
])

predictions = traverser.predict(X_with_missing)
```

## Complete Example: Mushroom Classification

Here's a complete example using the mushroom dataset:

```python
import numpy as np
import pandas as pd
from theta_sketch_tree import ThetaSketchDecisionTreeClassifier

# Step 1: Create synthetic theta sketches (normally you'd load real sketches)
class MockThetaSketch:
    def __init__(self, cardinality):
        self.cardinality = cardinality

    def get_estimate(self):
        return self.cardinality

    def intersection(self, other):
        return MockThetaSketch(min(self.cardinality, other.cardinality) * 0.8)

# Step 2: Create sketch data for mushroom features
sketch_data = {
    'positive': {  # Edible mushrooms
        'total': MockThetaSketch(4208),
        'cap-shape=bell': (MockThetaSketch(452), MockThetaSketch(3756)),
        'cap-surface=smooth': (MockThetaSketch(3244), MockThetaSketch(964)),
        'cap-color=brown': (MockThetaSketch(2284), MockThetaSketch(1924)),
        'has-odor': (MockThetaSketch(400), MockThetaSketch(3808)),
        'gill-size=broad': (MockThetaSketch(3656), MockThetaSketch(552)),
        'stalk-shape=enlarging': (MockThetaSketch(1228), MockThetaSketch(2980)),
        'ring-type=pendant': (MockThetaSketch(1296), MockThetaSketch(2912)),
    },
    'negative': {  # Poisonous mushrooms
        'total': MockThetaSketch(3916),
        'cap-shape=bell': (MockThetaSketch(86), MockThetaSketch(3830)),
        'cap-surface=smooth': (MockThetaSketch(1556), MockThetaSketch(2360)),
        'cap-color=brown': (MockThetaSketch(1968), MockThetaSketch(1948)),
        'has-odor': (MockThetaSketch(3472), MockThetaSketch(444)),
        'gill-size=broad': (MockThetaSketch(1728), MockThetaSketch(2188)),
        'stalk-shape=enlarging': (MockThetaSketch(1192), MockThetaSketch(2724)),
        'ring-type=pendant': (MockThetaSketch(1744), MockThetaSketch(2172)),
    }
}

feature_mapping = {
    'cap-shape=bell': 0,
    'cap-surface=smooth': 1,
    'cap-color=brown': 2,
    'has-odor': 3,
    'gill-size=broad': 4,
    'stalk-shape=enlarging': 5,
    'ring-type=pendant': 6
}

# Step 3: Train classifier
clf = ThetaSketchDecisionTreeClassifier(
    criterion='gini',
    max_depth=8,
    min_samples_split=5,
    verbose=1
)

print("Training classifier on mushroom data...")
clf.fit(sketch_data, feature_mapping)

# Step 4: Make predictions
test_mushrooms = np.array([
    [0, 1, 1, 0, 1, 0, 1],  # Smooth, brown, no odor, broad gills -> likely edible
    [0, 0, 0, 1, 0, 1, 0],  # Not smooth, not brown, has odor -> likely poisonous
    [1, 1, 1, 0, 1, 1, 1],  # Bell-shaped, smooth, brown, no odor -> likely edible
])

predictions = clf.predict(test_mushrooms)
probabilities = clf.predict_proba(test_mushrooms)

print(f"\\nPredictions: {['Edible' if p == 1 else 'Poisonous' for p in predictions]}")
print(f"Confidence scores:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    confidence = prob[pred]
    print(f"  Mushroom {i+1}: {confidence:.2f} confidence")

# Step 5: Analyze feature importance
print(f"\\nFeature Importance Analysis:")
importance_dict = clf.get_feature_importance_dict()
for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feature}: {importance:.3f}")
```

## Performance Considerations

### Training Performance
- Linear scaling with feature count: ~1.5s for 100 features
- Gini criterion is fastest, binomial is slowest
- Memory usage scales reasonably with dataset size

### Prediction Performance
- >400K samples/sec throughput
- Excellent scalability for batch prediction
- Constant time per sample regardless of feature count

### Optimization Tips
1. Use `criterion='gini'` for maximum speed
2. Set `max_depth` to limit tree size and training time
3. Use `min_samples_split` to prevent overfitting and reduce training time
4. Consider early stopping for large datasets

## Troubleshooting

### Common Issues

**Issue**: `ValueError: sketch_data must contain 'positive' key`
**Solution**: Ensure your sketch data dictionary has the correct structure with 'positive' and 'negative' keys.

**Issue**: `ValueError: X has N features, but classifier was fitted with M features`
**Solution**: Check that your test data X has the same number of columns as specified in feature_mapping.

**Issue**: Training is too slow
**Solution**: Reduce max_depth, increase min_samples_split, or switch to 'gini' criterion.

**Issue**: Poor prediction accuracy
**Solution**: Check that your sketch data accurately represents the underlying distributions. Verify feature_mapping is correct.

### Debugging Tips

1. Enable verbose output: `ThetaSketchDecisionTreeClassifier(verbose=2)`
2. Check feature importance to validate model learning
3. Inspect tree structure for sensible splits
4. Verify sketch cardinality estimates are reasonable

## API Reference

See individual docstrings for detailed parameter descriptions:

- `ThetaSketchDecisionTreeClassifier`: Main classifier class
- `ThetaSketchDecisionTreeClassifier.fit()`: Train the model
- `ThetaSketchDecisionTreeClassifier.predict()`: Make predictions
- `ThetaSketchDecisionTreeClassifier.predict_proba()`: Get class probabilities
- `ThetaSketchDecisionTreeClassifier.feature_importances_`: Feature importance scores
- `load_sketches()`: Load sketch data from CSV files
- `load_config()`: Load configuration from YAML files