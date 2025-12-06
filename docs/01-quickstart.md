# Quick Start Guide

## Overview

The Theta Sketch Decision Tree is a production-ready CART classifier that **trains on theta sketches** but **predicts on raw binary data**. This unique architecture enables privacy-preserving machine learning on large datasets while maintaining sklearn compatibility.

**Status**: ✅ **Production Ready** - 420 tests passing with 97% coverage

## 5-Minute Setup

### Prerequisites
- Python 3.8+
- Apache DataSketches Python library

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/theta-sketch-tree.git
cd Decision-trees_CART_sketch

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Verify Installation

```bash
# Run quick test
./venv/bin/python -c "from theta_sketch_tree import ThetaSketchDecisionTreeClassifier; print('✅ Installation successful')"
```

## Basic Usage

### Method 1: Pre-computed Sketches

```python
from theta_sketch_tree import ThetaSketchDecisionTreeClassifier, load_sketches, load_config
import numpy as np

# Load pre-computed sketch data
sketch_data = load_sketches('positive_class.csv', 'negative_class.csv')
config = load_config('config.yaml')

# Train classifier
clf = ThetaSketchDecisionTreeClassifier(criterion='gini', max_depth=10)
clf.fit(sketch_data, config['feature_mapping'])

# Make predictions on raw binary data
X_test = np.array([[1, 0, 1], [0, 1, 0]])  # Binary features: 0/1 values
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

print(f"Predictions: {predictions}")
print(f"Top features: {clf.get_top_features(top_k=3)}")
```

### Method 2: One-Step Training

```python
# Direct training from CSV files
clf = ThetaSketchDecisionTreeClassifier.fit_from_csv(
    positive_csv='positive_class.csv',
    negative_csv='negative_class.csv',
    feature_mapping_json='feature_mapping.json',
    criterion='gini',
    max_depth=10
)

# Ready for predictions
predictions = clf.predict(X_test)
```

## Command Line Interface

### Training from Raw CSV Data

```bash
# Train on mushroom dataset (generates sketches automatically)
./venv/bin/python run_binary_classification.py \
    ./tests/resources/agaricus-lepiota.csv class \
    --lg_k 14 --max_depth 8 --criterion gini \
    --verbose 1
```

### Complete Sketch-Based Workflow (Recommended)

Train from any CSV dataset using the 3-step sketch-based workflow:

```bash
# Step 1: Generate 2-column sketches from raw CSV
./venv/bin/python tools/create_2col_sketches.py data.csv target_column --lg_k 16

# Step 2: Convert to 3-column training format
./venv/bin/python tools/simple_convert_to_3col.py \
    examples/sketches/dataset_sketches/dataset_positive_2col_sketches_lg_k_16.csv \
    examples/sketches/dataset_sketches/dataset_negative_2col_sketches_lg_k_16.csv \
    examples/sketches/dataset_sketches/dataset_feature_mapping.json \
    examples/sketches/dataset_sketches/

# Step 3: Train from 3-column sketches
./venv/bin/python tools/train_from_3col_sketches.py \
    examples/sketches/dataset_sketches/dataset_3col_sketches.csv \
    examples/sketches/dataset_sketches/dataset_feature_mapping.json \
    configs/training_config.yaml
```

**Complete Example - Mushroom Dataset:**
```bash
# Generate mushroom sketches
./venv/bin/python tools/create_2col_sketches.py \
    tests/resources/agaricus-lepiota.csv class --lg_k 12

# Convert to training format
./venv/bin/python tools/simple_convert_to_3col.py \
    examples/sketches/agaricus_lepiota_sketches/agaricus_lepiota_positive_2col_sketches_lg_k_19.csv \
    examples/sketches/agaricus_lepiota_sketches/agaricus_lepiota_negative_2col_sketches_lg_k_19.csv \
    examples/sketches/agaricus_lepiota_sketches/agaricus_lepiota_feature_mapping.json \
    examples/sketches/agaricus_lepiota_sketches/

# Train model
./venv/bin/python tools/train_from_3col_sketches.py \
    examples/sketches/agaricus_lepiota_sketches/agaricus_lepiota_3col_sketches.csv \
    examples/sketches/agaricus_lepiota_sketches/agaricus_lepiota_feature_mapping.json \
    configs/mushroom_training_config.yaml
```

### Legacy Sketch Training

```bash
# Train directly from legacy sketch CSV files
./venv/bin/python tools/train_from_sketches.py \
    positive.csv negative.csv config.yaml
```

**Benefits of Sketch-Based Workflow:**
- 🔄 **Reproducible**: Same sketches can be reused for multiple training runs
- ⚡ **Fast**: Skip sketch generation for repeated experiments
- 🔒 **Privacy**: Share sketches without exposing raw data
- 📦 **Portable**: Sketches are much smaller than original datasets
- 🎛️ **Flexible**: Works with any binary classification CSV dataset
- 🏭 **Production Ready**: Ideal for enterprise deployment scenarios

### With Pruning (Recommended)

```bash
# Cost-complexity pruning for better generalization
./venv/bin/python run_binary_classification.py \
    ./tests/resources/agaricus-lepiota.csv class \
    --lg_k 14 --max_depth 8 --criterion gini \
    --pruning cost_complexity --verbose 1
```

## Key Features

- **🎯 Sketch-based Training**: Memory-efficient learning from theta sketches
- **📊 Standard Inference**: sklearn-compatible predictions on raw binary data
- **🌳 Advanced Pruning**: 4 pruning methods (cost-complexity, validation, reduced-error, min-impurity)
- **⚖️ Multiple Criteria**: Gini, Entropy, Gain Ratio, Binomial, Chi-Square
- **💾 Model Persistence**: Save/load trained models with hyperparameters
- **🔧 Missing Values**: Robust handling with majority-vote strategy
- **📈 Feature Importance**: Built-in weighted impurity decrease calculation
- **⚡ High Performance**: >400K predictions/second with linear scaling

## Expected Data Formats

### Sketch CSV Files
Your training data should be pre-computed theta sketches in CSV format:

```
feature_name,present_sketch,absent_sketch
age>30,<base64_encoded_sketch>,<base64_encoded_sketch>
income>50k,<base64_encoded_sketch>,<base64_encoded_sketch>
education=grad,<base64_encoded_sketch>,<base64_encoded_sketch>
```

### Feature Mapping JSON
```json
{
  "age>30": 0,
  "income>50k": 1,
  "education=grad": 2
}
```

### Prediction Data
Binary matrix with 0/1 values:
```python
X_test = np.array([
    [1, 0, 1],  # age>30=yes, income>50k=no, education=grad=yes
    [0, 1, 0],  # age>30=no, income>50k=yes, education=grad=no
])
```

## Performance Benchmarks

| Metric | Performance |
|--------|-------------|
| Training Speed | ~1.5s for 100 features |
| Prediction Throughput | >400K samples/sec |
| Memory Usage | Linear scaling |
| Test Coverage | 97% (420 tests) |

## Testing

```bash
# Quick validation
./test-unit.sh

# Full test suite
./test-all.sh
```

## Next Steps

1. **Detailed Usage**: See [User Guide](02-user-guide.md) for comprehensive examples
2. **Architecture**: Read [Architecture Overview](03-architecture.md) to understand the system
3. **Performance**: Review [Performance Guide](06-performance.md) for optimization
4. **Production**: Check [Deployment Guide](11-deployment.md) for production strategies

## Common Issues

- **Import Error**: Verify virtual environment activation and dependencies
- **Sketch Format Error**: Ensure CSV files match expected format (see [Data Formats](09-data-formats.md))
- **Performance Issues**: Review [Performance Guide](06-performance.md) for optimization tips

## Support

- 📖 **Documentation**: [docs/](./README.md)
- 🐛 **Issues**: GitHub Issues
- 💬 **Discussions**: GitHub Discussions

---

**Ready to dive deeper?** Continue with the [User Guide](02-user-guide.md) for comprehensive examples and advanced features.