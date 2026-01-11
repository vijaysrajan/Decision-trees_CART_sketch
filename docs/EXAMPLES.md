# 📚 Theta Sketch Decision Tree Examples

This document provides practical examples of using the Theta Sketch Decision Tree classifier with and without pruning on real datasets.

## 🎯 Quick Start Examples

### Example 1: Basic Classification (No Pruning)
```bash
# Train on mushroom dataset without pruning
./venv/bin/python run_binary_classification.py \
    ./tests/resources/agaricus-lepiota.csv class \
    --lg_k 14 --max_depth 8 --criterion gini \
    --verbose 1 --sample_size 2000

# Output:
# Tree without pruning: 17 nodes, depth 5
# Training completed successfully!
```

### Example 2: With Cost-Complexity Pruning
```bash
# Same dataset with pruning for better generalization
./venv/bin/python run_binary_classification.py \
    ./tests/resources/agaricus-lepiota.csv class \
    --lg_k 14 --max_depth 8 --criterion gini \
    --pruning cost_complexity --verbose 1 --sample_size 2000

# Output:
# Applying cost_complexity pruning...
# Pruning complete: 8 nodes removed
# Compression ratio: 0.529
# Tree with pruning: 9 nodes, depth 4
```

## 🍄 Mushroom Dataset Examples

### Complete Pruning Comparison
```bash
# Test all pruning methods on the same dataset
for method in none cost_complexity validation reduced_error min_impurity; do
    echo "Testing $method pruning:"
    ./venv/bin/python run_binary_classification.py \
        ./tests/resources/agaricus-lepiota.csv class \
        --pruning $method --lg_k 14 --max_depth 8 \
        --verbose 1 --sample_size 1500
    echo "---"
done
```

### Results Summary
| Method | Nodes | Reduction | Best For |
|--------|-------|-----------|----------|
| None | 17 | - | Baseline comparison |
| Cost-complexity | 9 | 47% reduction | **General use** ⭐ |
| Validation | 17 | 0% | Conservative accuracy |
| Min impurity | 15 | 12% | Minimal pruning |

## 📊 Dataset-Specific Examples

### Small Dataset (< 500 samples)
```bash
# Conservative approach for small datasets
./venv/bin/python run_binary_classification.py \
    small_dataset.csv target \
    --lg_k 12 --max_depth 6 \
    --pruning min_impurity --min_impurity_decrease 0.005 \
    --verbose 1
```

### Medium Dataset (500-2000 samples)
```bash
# Balanced pruning for medium datasets
./venv/bin/python run_binary_classification.py \
    medium_dataset.csv target \
    --lg_k 14 --max_depth 8 \
    --pruning cost_complexity \
    --verbose 1
```

### Large Dataset (> 2000 samples)
```bash
# Validation-based pruning for large datasets
./venv/bin/python run_binary_classification.py \
    large_dataset.csv target \
    --lg_k 16 --max_depth 10 \
    --pruning validation --validation_fraction 0.25 \
    --verbose 1
```

## 🛠 Advanced Usage Examples

### Model Persistence Workflow
```bash
# 1. Train and save a pruned model
./venv/bin/python run_binary_classification.py \
    ./tests/resources/agaricus-lepiota.csv class \
    --pruning cost_complexity --lg_k 14 \
    --save_model mushroom_pruned_model \
    --verbose 1

# 2. Load and inspect the saved model
./venv/bin/python run_binary_classification.py \
    dummy_file dummy_col \
    --model_info mushroom_pruned_model.pkl

# 3. Use saved model for predictions
./venv/bin/python run_binary_classification.py \
    new_mushroom_data.csv class \
    --load_model mushroom_pruned_model.pkl
```

### Parameter Tuning Examples
```bash
# Conservative pruning (removes very little)
./venv/bin/python run_binary_classification.py \
    data.csv target --pruning min_impurity \
    --min_impurity_decrease 0.001

# Moderate pruning (balanced approach)
./venv/bin/python run_binary_classification.py \
    data.csv target --pruning cost_complexity \
    --lg_k 14 --max_depth 8

# Aggressive pruning (maximum generalization)
./venv/bin/python run_binary_classification.py \
    data.csv target --pruning validation \
    --validation_fraction 0.3 --max_depth 12
```

## 🐍 Python API Examples

### Basic Python Workflow
```python
#!/usr/bin/env python3
import pandas as pd
from tests.test_binary_classification_sketches import (
    create_binary_classification_sketches,
    create_binary_classification_feature_mapping
)
from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Create theta sketches
sketches = create_binary_classification_sketches(df, lg_k=14)
mapping = create_binary_classification_feature_mapping(sketches)

# Create classifier with pruning
clf = ThetaSketchDecisionTreeClassifier(
    criterion='gini',
    max_depth=8,
    pruning='cost_complexity',
    verbose=1
)

# Train the model
clf.fit(sketches, mapping)

# Get results
print(f"Tree nodes: {clf._count_tree_nodes()}")
print(f"Tree depth: {clf.tree_.depth}")
print(f"Top features: {clf.get_top_features(5)}")
```

### Pruning Comparison Script
```python
#!/usr/bin/env python3
"""Compare all pruning methods systematically."""

import pandas as pd
from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier

def run_pruning_comparison(csv_file, target_col, sample_size=1000):
    """Run comprehensive pruning comparison."""

    # Load and prepare data
    df = pd.read_csv(csv_file)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # Create sketches
    from tests.test_binary_classification_sketches import (
        create_binary_classification_sketches,
        create_binary_classification_feature_mapping
    )

    sketches = create_binary_classification_sketches(df, lg_k=14)
    mapping = create_binary_classification_feature_mapping(sketches)

    # Test all methods
    methods = {
        'none': 'No pruning (baseline)',
        'cost_complexity': 'Cost-complexity pruning',
        'validation': 'Validation-based pruning',
        'min_impurity': 'Minimum impurity pruning'
    }

    results = []

    for method, description in methods.items():
        clf = ThetaSketchDecisionTreeClassifier(
            criterion='gini',
            max_depth=8,
            pruning=method,
            min_impurity_decrease=0.01 if method == 'min_impurity' else 0.0,
            verbose=0
        )

        clf.fit(sketches, mapping)

        results.append({
            'Method': method,
            'Description': description,
            'Nodes': clf._count_tree_nodes(),
            'Leaves': clf._count_tree_leaves(),
            'Depth': clf.tree_.depth
        })

    # Create comparison DataFrame
    df_results = pd.DataFrame(results)
    baseline_nodes = df_results[df_results['Method'] == 'none']['Nodes'].iloc[0]
    df_results['Reduction'] = df_results['Nodes'].apply(
        lambda x: f"-{baseline_nodes - x} ({((baseline_nodes - x) / baseline_nodes * 100):.0f}%)"
        if x < baseline_nodes else "None"
    )

    return df_results

# Usage
if __name__ == "__main__":
    results = run_pruning_comparison(
        './tests/resources/agaricus-lepiota.csv',
        'class',
        sample_size=1500
    )

    print("🍄 Mushroom Dataset Pruning Comparison")
    print("=" * 50)
    print(results[['Method', 'Nodes', 'Reduction', 'Description']].to_string(index=False))
```

### Validation-Based Pruning with Custom Data Split
```python
#!/usr/bin/env python3
"""Advanced example with custom validation split."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier

def train_with_validation_pruning(csv_file, target_col):
    """Train with validation-based pruning using custom data split."""

    # Load dataset
    df = pd.read_csv(csv_file)

    # Split for validation
    train_df, val_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df[target_col]
    )

    print(f"Training: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")

    # Create sketches from training data only
    from tests.test_binary_classification_sketches import (
        create_binary_classification_sketches,
        create_binary_classification_feature_mapping
    )

    sketches = create_binary_classification_sketches(train_df, lg_k=14)
    mapping = create_binary_classification_feature_mapping(sketches)

    # Prepare validation data
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    # Convert to binary features (simplified for example)
    X_val_binary = np.zeros((len(X_val), len(mapping)))
    for i, (_, row) in enumerate(X_val.iterrows()):
        for feature_name, col_idx in mapping.items():
            if '=' in feature_name:
                base_feature, value = feature_name.split('=')
                if base_feature in row.index:
                    X_val_binary[i, col_idx] = 1 if str(row[base_feature]) == value else 0

    # Train with validation pruning
    clf = ThetaSketchDecisionTreeClassifier(
        criterion='gini',
        max_depth=10,
        pruning='validation',
        validation_fraction=0.25,  # This is ignored when X_val provided
        verbose=1
    )

    # Fit with explicit validation data
    clf.fit(sketches, mapping, X_val=X_val_binary, y_val=y_val.values)

    print(f"\\nFinal tree: {clf._count_tree_nodes()} nodes, depth {clf.tree_.depth}")
    return clf

# Usage
if __name__ == "__main__":
    model = train_with_validation_pruning(
        './tests/resources/agaricus-lepiota.csv',
        'class'
    )
```

## 🔧 Troubleshooting Examples

### Issue: No Pruning Occurs
```bash
# If no pruning happens, try more aggressive settings
./venv/bin/python run_binary_classification.py \
    data.csv target \
    --pruning min_impurity --min_impurity_decrease 0.05  # Higher threshold

# Or try different method
./venv/bin/python run_binary_classification.py \
    data.csv target \
    --pruning cost_complexity --max_depth 12  # Deeper tree first
```

### Issue: Over-Pruning (Tree becomes too small)
```bash
# Use more conservative settings
./venv/bin/python run_binary_classification.py \
    data.csv target \
    --pruning min_impurity --min_impurity_decrease 0.001  # Lower threshold

# Or use reduced error pruning
./venv/bin/python run_binary_classification.py \
    data.csv target \
    --pruning reduced_error --validation_fraction 0.2
```

## 📋 Best Practices Checklist

- ✅ **Start with `cost_complexity`** for general use
- ✅ **Use `validation`** when accuracy is critical
- ✅ **Try `min_impurity`** for conservative pruning
- ✅ **Compare multiple methods** on your specific dataset
- ✅ **Save pruned models** for production use
- ✅ **Monitor compression ratios** (aim for 20-50% reduction)
- ✅ **Validate on held-out test data** after pruning

## 🚀 Production Deployment Example

```bash
# 1. Train best model with validation
./venv/bin/python run_binary_classification.py \
    training_data.csv target \
    --pruning cost_complexity --lg_k 16 \
    --save_model production_model --verbose 1

# 2. Test on validation data
./venv/bin/python run_binary_classification.py \
    validation_data.csv target \
    --load_model production_model.pkl

# 3. Deploy for inference
./venv/bin/python run_binary_classification.py \
    new_data.csv target \
    --load_model production_model.pkl
```

This completes the practical examples. All scripts are ready to run and demonstrate real-world usage patterns.