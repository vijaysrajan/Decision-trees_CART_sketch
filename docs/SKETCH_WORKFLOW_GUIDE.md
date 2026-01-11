# 🎯 Complete Sketch-Based Workflow Guide

This guide covers the complete end-to-end workflow for building, training, serializing, and deploying theta sketch decision trees starting from CSV files containing pre-computed theta sketches.

## 📋 Overview

The theta sketch workflow has three main phases:
1. **Training Phase**: Load sketch CSV files → Build tree → Save model
2. **Inference Phase**: Load saved model → Predict on new data
3. **Model Management**: Inspect, validate, and deploy models

## 🗂️ Required Input Files

### 1. Sketch CSV Files (Pre-computed from Big Data)

You need **2 CSV files** in one of these modes:

#### **Dual-Class Mode** (Recommended for Balanced Datasets)
```
positive_class.csv    # Sketches from positive class samples
negative_class.csv    # Sketches from negative class samples
```

#### **One-vs-All Mode** (Recommended for Imbalanced Datasets)
```
positive_class.csv    # Sketches from positive class samples
total_population.csv  # Sketches from ENTIRE dataset (all classes)
```

### 2. CSV Format Specification

**Required 3-column format:**
```csv
identifier,sketch_feature_present,sketch_feature_absent
total,c2FtcGxlX3ByZXNlbnRfc2tldGNo,c2FtcGxlX2Fic2VudF9za2V0Y2g=
age>30,YWdlXzMwX3ByZXNlbnQ=,YWdlXzMwX2Fic2VudA==
income>50k,aW5jb21lXzUwa19wcmVzZW50,aW5jb21lXzUwa19hYnNlbnQ=
```

- **identifier**: Feature name or 'total' for population sketch
- **sketch_feature_present**: Base64-encoded theta sketch for samples where feature=1
- **sketch_feature_absent**: Base64-encoded theta sketch for samples where feature=0

### 3. Configuration File (Optional)

**config.yaml** - Auto-generated if not provided:
```yaml
feature_mapping:
  "age>30": 0
  "income>50k": 1
  "has_diabetes": 2

classification_mode: "dual_class"  # or "one_vs_all"
lg_k: 12  # Sketch size parameter
```

## 🚀 Complete Workflow Examples

### Example 1: Dual-Class Mode Workflow

```bash
# Step 1: Prepare your sketch CSV files
# positive_class.csv - sketches from positive samples
# negative_class.csv - sketches from negative samples

# Step 2: Train model with pruning and save
./venv/bin/python run_binary_classification.py \
    positive_class.csv negative_class.csv \
    --lg_k 14 --max_depth 8 \
    --pruning cost_complexity \
    --save_model my_production_model \
    --verbose 1

# Output:
# Building decision tree...
# Applying cost_complexity pruning...
# Pruning complete: 8 nodes removed
# Model saved: my_production_model.pkl

# Step 3: Load model for inference
./venv/bin/python run_binary_classification.py \
    new_data.csv target_column \
    --load_model my_production_model.pkl

# Step 4: Inspect saved model
./venv/bin/python run_binary_classification.py \
    dummy_file dummy_col \
    --model_info my_production_model.pkl
```

### Example 2: One-vs-All Mode Workflow

```bash
# For imbalanced datasets (rare events)
# positive_class.csv - sketches from positive samples
# total_population.csv - sketches from ALL samples

./venv/bin/python run_binary_classification.py \
    positive_class.csv total_population.csv \
    --lg_k 16 --max_depth 10 \
    --pruning validation --validation_fraction 0.2 \
    --save_model rare_event_model \
    --verbose 2
```

## 🐍 Python API Workflow

### Complete Training and Inference Script

```python
#!/usr/bin/env python3
"""Complete sketch-based workflow example."""

import pandas as pd
import numpy as np
from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier

def train_model_from_sketches():
    """Train model from sketch CSV files."""

    # Method 1: Direct fit_from_csv (One-step training)
    clf = ThetaSketchDecisionTreeClassifier.fit_from_csv(
        positive_csv='positive_class.csv',
        negative_csv='negative_class.csv',  # or total_population.csv for one-vs-all
        config_path='config.yaml',  # Optional - auto-generated if missing
        criterion='gini',
        max_depth=10,
        pruning='cost_complexity',
        verbose=1
    )

    # Save the trained model
    clf.save_model('production_model', include_sketches=False)

    print(f"Model trained and saved!")
    print(f"Tree nodes: {clf._count_tree_nodes()}")
    print(f"Tree depth: {clf.tree_.depth}")

    return clf

def load_and_predict():
    """Load saved model and make predictions."""

    # Load the saved model
    clf = ThetaSketchDecisionTreeClassifier.load_model('production_model.pkl')

    # Prepare inference data (binary features)
    # X should be binary matrix: rows=samples, cols=features
    X_test = np.array([
        [1, 0, 1, 0],  # Sample 1: features 0,2 present
        [0, 1, 0, 1],  # Sample 2: features 1,3 present
        [1, 1, 0, 0]   # Sample 3: features 0,1 present
    ])

    # Make predictions
    predictions = clf.predict(X_test)
    probabilities = clf.predict_proba(X_test)

    print(f"Predictions: {predictions}")
    print(f"Probabilities: {probabilities}")

    # Get feature importance
    top_features = clf.get_top_features(5)
    print(f"Top features: {top_features}")

    return predictions, probabilities

def inspect_model():
    """Inspect a saved model."""

    # Load and inspect model metadata
    clf = ThetaSketchDecisionTreeClassifier.load_model('production_model.pkl')
    info = clf.get_model_info()

    print("Model Information:")
    print(f"  Features: {info['n_features']}")
    print(f"  Tree nodes: {info['tree_nodes']}")
    print(f"  Tree depth: {info['tree_depth']}")
    print(f"  Hyperparameters: {info['hyperparameters']}")

    return info

if __name__ == "__main__":
    # Complete workflow
    print("1. Training model from sketches...")
    model = train_model_from_sketches()

    print("\\n2. Loading model and making predictions...")
    pred, prob = load_and_predict()

    print("\\n3. Inspecting saved model...")
    info = inspect_model()
```

### Advanced Workflow with Custom Data Processing

```python
#!/usr/bin/env python3
"""Advanced sketch workflow with custom processing."""

from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier
from theta_sketch_tree import load_sketches, load_config
import pandas as pd
import numpy as np

def advanced_training_workflow():
    """Advanced training with custom sketch loading."""

    # Method 2: Manual sketch loading with more control

    # Load sketch data manually
    sketch_data = load_sketches(
        positive_csv='positive_class.csv',
        negative_csv='negative_class.csv'
    )

    # Load or create configuration
    try:
        config = load_config('config.yaml')
        feature_mapping = config['feature_mapping']
    except FileNotFoundError:
        # Auto-generate feature mapping
        feature_names = list(sketch_data['positive'].keys())
        feature_names.remove('total')  # Remove total sketch
        feature_mapping = {name: idx for idx, name in enumerate(feature_names)}

    print(f"Feature mapping: {feature_mapping}")
    print(f"Number of features: {len(feature_mapping)}")

    # Create classifier with advanced settings
    clf = ThetaSketchDecisionTreeClassifier(
        criterion='gini',
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        pruning='validation',
        validation_fraction=0.25,
        min_impurity_decrease=0.01,
        verbose=2
    )

    # Train with manual data
    clf.fit(sketch_data, feature_mapping)

    # Advanced model saving with metadata
    model_metadata = {
        'dataset_name': 'production_dataset_v2',
        'training_date': '2024-01-15',
        'data_version': '2.1',
        'features_count': len(feature_mapping),
        'pruning_applied': clf.pruning,
        'performance_notes': 'Optimized for high-volume inference'
    }

    # Save with custom metadata
    clf.save_model('advanced_model', include_sketches=False)

    return clf, feature_mapping

def batch_inference_workflow(model_path, inference_csv):
    """Batch inference on CSV file."""

    # Load model
    clf = ThetaSketchDecisionTreeClassifier.load_model(model_path)

    # Load inference data
    df = pd.read_csv(inference_csv)

    # Convert to binary matrix (assumes CSV has binary 0/1 features)
    feature_columns = [col for col in df.columns if col not in ['id', 'timestamp']]
    X = df[feature_columns].values.astype(int)

    print(f"Processing {len(df)} samples with {X.shape[1]} features...")

    # Batch predictions
    predictions = clf.predict(X)
    probabilities = clf.predict_proba(X)

    # Create results DataFrame
    results = df.copy()
    results['prediction'] = predictions
    results['probability_class_0'] = probabilities[:, 0]
    results['probability_class_1'] = probabilities[:, 1]
    results['confidence'] = np.max(probabilities, axis=1)

    # Save results
    output_file = inference_csv.replace('.csv', '_predictions.csv')
    results.to_csv(output_file, index=False)

    print(f"Results saved to: {output_file}")
    print(f"Prediction distribution: {pd.Series(predictions).value_counts().to_dict()}")

    return results

if __name__ == "__main__":
    # Advanced workflow example
    model, mapping = advanced_training_workflow()

    # Batch inference example
    # results = batch_inference_workflow('advanced_model.pkl', 'new_data.csv')
```

## 📊 Data Preparation Guidelines

### Creating Sketch CSV Files

Your big data pipeline should generate sketch CSV files with this structure:

```python
# Example of what your big data pipeline should output
# This is typically done in Spark, Hadoop, or other big data frameworks

def create_sketch_csvs_from_bigdata():
    """
    Example: How sketch CSV files should be created from big data.
    This would typically run in your big data pipeline (Spark, etc.)
    """

    # Pseudocode for big data sketch generation:
    #
    # FOR each feature in features:
    #     positive_present_sketch = create_sketch(bigdata.filter(class==1 AND feature==1))
    #     positive_absent_sketch = create_sketch(bigdata.filter(class==1 AND feature==0))
    #
    #     negative_present_sketch = create_sketch(bigdata.filter(class==0 AND feature==1))
    #     negative_absent_sketch = create_sketch(bigdata.filter(class==0 AND feature==0))
    #
    #     save_to_csv(positive_class.csv, feature, positive_present_sketch, positive_absent_sketch)
    #     save_to_csv(negative_class.csv, feature, negative_present_sketch, negative_absent_sketch)

    pass  # Implementation depends on your big data platform
```

### Validation Checklist

Before training, verify your sketch CSV files:

```bash
# 1. Check file format
head -3 positive_class.csv
# Should show: identifier,sketch_feature_present,sketch_feature_absent

# 2. Verify sketch count
wc -l positive_class.csv negative_class.csv
# Should be equal (same number of features)

# 3. Check for 'total' sketch
grep "^total," positive_class.csv
# Should exist for population estimates

# 4. Validate base64 encoding
python -c "import base64; print('Valid' if base64.b64decode('c2FtcGxl') else 'Invalid')"
```

## 🚀 Production Deployment

### Production Training Pipeline

```bash
#!/bin/bash
# production_training.sh

set -e

echo "🚀 Production Model Training Pipeline"

# 1. Validate input files
echo "📋 Validating sketch files..."
python validate_sketches.py positive_class.csv negative_class.csv

# 2. Train with optimal settings
echo "🤖 Training production model..."
./venv/bin/python run_binary_classification.py \
    positive_class.csv negative_class.csv \
    --lg_k 16 \
    --max_depth 10 \
    --criterion gini \
    --pruning cost_complexity \
    --save_model production_model_$(date +%Y%m%d) \
    --verbose 1

# 3. Validate trained model
echo "✅ Validating trained model..."
./venv/bin/python run_binary_classification.py \
    dummy_file dummy_col \
    --model_info production_model_$(date +%Y%m%d).pkl

echo "🎉 Production model ready!"
```

### Production Inference Pipeline

```bash
#!/bin/bash
# production_inference.sh

MODEL_PATH="production_model_20240115.pkl"
INPUT_DATA="daily_inference_data.csv"
OUTPUT_PATH="predictions_$(date +%Y%m%d_%H%M).csv"

echo "🔮 Production Inference Pipeline"

# 1. Load model and process data
python batch_inference.py \
    --model $MODEL_PATH \
    --input $INPUT_DATA \
    --output $OUTPUT_PATH \
    --batch_size 10000

echo "📊 Inference complete: $OUTPUT_PATH"
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue: "Sketch data format error"
```bash
# Verify CSV format
head -1 positive_class.csv
# Expected: identifier,sketch_feature_present,sketch_feature_absent

# Check for BOM or encoding issues
file positive_class.csv
hexdump -C positive_class.csv | head -1
```

#### Issue: "Feature mapping mismatch"
```bash
# Compare feature counts
wc -l positive_class.csv negative_class.csv
grep -v "^total," positive_class.csv | wc -l

# Verify feature names match
cut -d',' -f1 positive_class.csv | sort > pos_features.txt
cut -d',' -f1 negative_class.csv | sort > neg_features.txt
diff pos_features.txt neg_features.txt
```

#### Issue: "Base64 decode error"
```python
# Test sketch decoding
import base64
with open('positive_class.csv') as f:
    for i, line in enumerate(f):
        if i == 0: continue  # Skip header
        parts = line.strip().split(',')
        try:
            base64.b64decode(parts[1])  # Test present sketch
            base64.b64decode(parts[2])  # Test absent sketch
        except Exception as e:
            print(f"Line {i}: {e}")
```

## 📚 Additional Resources

- **[Data Formats](docs/04_data_formats.md)** - Detailed CSV format specifications
- **[User Guide](docs/user_guide.md)** - Complete API documentation
- **[Pruning Guide](PRUNING_GUIDE.md)** - Overfitting prevention methods
- **[Examples](EXAMPLES.md)** - Real-world usage examples

## 🎯 Best Practices Summary

✅ **Always validate sketch CSV format** before training
✅ **Use cost_complexity pruning** for production models
✅ **Save models without sketches** for faster deployment
✅ **Verify feature mapping consistency** across files
✅ **Use one-vs-all mode** for imbalanced datasets
✅ **Test inference pipeline** before production deployment
✅ **Monitor model performance** in production
✅ **Version control your models** with timestamps

This completes the comprehensive sketch-based workflow guide!