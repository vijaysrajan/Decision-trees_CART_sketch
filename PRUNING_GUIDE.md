# 🌳 Advanced Pruning Methods Guide

This guide demonstrates how to use the advanced pruning methods available in the Theta Sketch Decision Tree classifier to prevent overfitting and improve model generalization.

## 🎯 Overview

The classifier supports **5 pruning methods**:

| Method | Description | Best Use Case |
|--------|-------------|---------------|
| `none` | No pruning (baseline) | Small, high-quality datasets |
| `validation` | Validation-based pruning | When you have validation data |
| `cost_complexity` | Cost-complexity pruning | General overfitting prevention |
| `reduced_error` | Reduced error pruning | Conservative accuracy preservation |
| `min_impurity` | Minimum impurity decrease | Remove low-benefit splits |

## 📊 Quick Comparison Example

```bash
# Test all pruning methods on mushroom dataset
./venv/bin/python run_binary_classification.py \
    ./tests/resources/agaricus-lepiota.csv class \
    --lg_k 14 --max_depth 8 --sample_size 1500 \
    --pruning cost_complexity --verbose 1
```

**Results on Mushroom Dataset (1500 samples):**

| Method | Nodes | Reduction | Effectiveness |
|--------|-------|-----------|--------------|
| None | 17 | - | Baseline |
| Cost-complexity | 9 | -8 (47%) | High 🔥 |
| Validation | 17 | 0 (0%) | Conservative |
| Min impurity | 15 | -2 (12%) | Low |

## 🛠 Command Line Usage

### Basic Syntax
```bash
./venv/bin/python run_binary_classification.py <dataset.csv> <target_column> [pruning_options]
```

### Pruning Parameters
```bash
--pruning {none,validation,cost_complexity,reduced_error,min_impurity}
--min_impurity_decrease FLOAT     # For min_impurity pruning (default: 0.0)
--validation_fraction FLOAT       # Fraction for validation pruning (default: 0.2)
```

## 📝 Method-Specific Examples

### 1. Cost-Complexity Pruning (Recommended)
```bash
# Best general-purpose pruning method
./venv/bin/python run_binary_classification.py \
    data.csv target --pruning cost_complexity \
    --lg_k 14 --max_depth 10
```
**When to use**: Default choice for most datasets. Provides good balance between complexity reduction and accuracy preservation.

### 2. Validation-Based Pruning
```bash
# Uses validation accuracy to guide pruning
./venv/bin/python run_binary_classification.py \
    data.csv target --pruning validation \
    --validation_fraction 0.3 --lg_k 14
```
**When to use**: When you have sufficient data for validation and want accuracy-driven pruning.

### 3. Minimum Impurity Decrease Pruning
```bash
# Remove splits with minimal benefit
./venv/bin/python run_binary_classification.py \
    data.csv target --pruning min_impurity \
    --min_impurity_decrease 0.01 --lg_k 14
```
**When to use**: Conservative pruning to remove only clearly unnecessary splits.

### 4. Reduced Error Pruning
```bash
# Conservative accuracy-preserving pruning
./venv/bin/python run_binary_classification.py \
    data.csv target --pruning reduced_error \
    --validation_fraction 0.25 --lg_k 14
```
**When to use**: When maintaining training accuracy is critical.

## 🧪 Python API Examples

### Basic Usage
```python
from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier
from tests.test_binary_classification_sketches import (
    create_binary_classification_sketches,
    create_binary_classification_feature_mapping
)

# Create classifier with pruning
clf = ThetaSketchDecisionTreeClassifier(
    criterion='gini',
    max_depth=10,
    pruning='cost_complexity',          # Enable pruning
    min_impurity_decrease=0.01,         # Pruning threshold
    validation_fraction=0.2,            # Validation split
    verbose=1
)

# Fit with sketches (standard workflow)
clf.fit(sketches, feature_mapping)
```

### Advanced Usage with Validation Data
```python
from sklearn.model_selection import train_test_split

# Split dataset for validation-based pruning
train_df, val_df = train_test_split(df, test_size=0.3, stratify=df['class'])
X_val = val_df.drop(columns=['class'])
y_val = val_df['class']

# Convert validation data to binary features
X_val_binary = convert_to_binary_features(X_val, feature_mapping)

# Fit with validation data
clf.fit(sketches, feature_mapping, X_val=X_val_binary, y_val=y_val.values)
```

### Pruning Comparison Script
```python
import pandas as pd
from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier

def compare_pruning_methods(df, target_col):
    """Compare all pruning methods on a dataset."""

    # Prepare data
    sketches = create_binary_classification_sketches(df, lg_k=14)
    mapping = create_binary_classification_feature_mapping(sketches)

    methods = ['none', 'cost_complexity', 'validation', 'min_impurity']
    results = []

    for method in methods:
        clf = ThetaSketchDecisionTreeClassifier(
            pruning=method,
            max_depth=8,
            min_impurity_decrease=0.01 if method == 'min_impurity' else 0.0,
            verbose=0
        )
        clf.fit(sketches, mapping)

        results.append({
            'method': method,
            'nodes': clf._count_tree_nodes(),
            'leaves': clf._count_tree_leaves(),
            'depth': clf.tree_.depth
        })

    return pd.DataFrame(results)

# Usage
df = pd.read_csv('your_dataset.csv')
comparison = compare_pruning_methods(df, 'target_column')
print(comparison)
```

## 📈 Performance Guidelines

### Dataset Size Recommendations

| Dataset Size | Recommended Pruning | Rationale |
|-------------|-------------------|-----------|
| < 500 samples | `none` or `min_impurity` | Small datasets rarely overfit |
| 500-2000 | `cost_complexity` | Balanced approach |
| 2000-5000 | `validation` | Sufficient data for validation |
| > 5000 | `cost_complexity` or `validation` | Either works well |

### Parameter Tuning Guidelines

**min_impurity_decrease values:**
- `0.001`: Very conservative (removes almost nothing)
- `0.01`: Moderate pruning
- `0.05`: Aggressive pruning

**validation_fraction values:**
- `0.1`: Small validation set (use with large datasets)
- `0.2`: Standard choice
- `0.3`: Large validation set (use with medium datasets)

## 🎯 Best Practices

### 1. Start with Cost-Complexity
```bash
# Default recommendation
--pruning cost_complexity --max_depth 8
```

### 2. Use Validation Pruning for Critical Applications
```bash
# When accuracy is paramount
--pruning validation --validation_fraction 0.3
```

### 3. Combine with Appropriate Tree Depth
```bash
# Deeper trees need more aggressive pruning
--max_depth 12 --pruning cost_complexity
```

### 4. Monitor Pruning Statistics
All pruning methods report:
- Nodes removed
- Compression ratio
- Pruning effectiveness

Example output:
```
Applying cost_complexity pruning...
Pruning complete: 8 nodes removed
Compression ratio: 0.529
```

## ⚠️ Common Issues and Solutions

### Issue: "No pruning occurred"
**Cause**: Tree is already optimal or threshold too low
**Solution**: Increase `min_impurity_decrease` or try different method

### Issue: "Over-pruning to single node"
**Cause**: Dataset has weak patterns or validation set too small
**Solution**: Use `min_impurity` instead or increase validation fraction

### Issue: "JSON serialization errors"
**Cause**: Numpy data types in tree structure
**Solution**: This has been fixed in the current version

## 🚀 Advanced Features

### Model Persistence with Pruning
```bash
# Save pruned model
./venv/bin/python run_binary_classification.py \
    data.csv target --pruning cost_complexity \
    --save_model my_pruned_model

# Load and inspect
./venv/bin/python run_binary_classification.py \
    dummy_file dummy_col --model_info my_pruned_model.pkl
```

### Feature Importance after Pruning
Pruning automatically updates feature importance rankings to reflect the simplified tree structure.

## 📊 Real Dataset Examples

See the complete examples in:
- `tools/test_pruning_demo.py` - Basic pruning demonstration
- `tools/analyze_mushroom_pruning.py` - Comprehensive mushroom dataset analysis
- `EXAMPLES.md` - More real-world use cases

## 🔍 Next Steps

1. **Experiment** with different pruning methods on your data
2. **Compare** pruning effectiveness using the comparison scripts
3. **Tune** parameters based on your specific requirements
4. **Save** best-performing pruned models for production use