# Theta Sketch Decision Tree Classifier

A production-ready CART Decision Tree that trains on theta sketches but performs inference on raw tabular data. Designed for large-scale machine learning with privacy-preserving data structures and full sklearn compatibility.

**Status**: ✅ **Production Ready** - 420 tests passing with 97% coverage

## Quick Start

```python
from theta_sketch_tree import ThetaSketchDecisionTreeClassifier, load_sketches, load_config
import numpy as np

# Load pre-computed sketch data
sketch_data = load_sketches('positive_class.csv', 'negative_class.csv')
config = load_config('config.yaml')

# Train classifier (Random Forest compatible)
clf = ThetaSketchDecisionTreeClassifier(
    criterion='gini',
    max_depth=10,
    max_features='sqrt',  # Feature subsampling for ensemble methods
    random_state=42
)
clf.fit(sketch_data, config['feature_mapping'])

# Make predictions on raw binary data
X_test = np.array([[1, 0, 1], [0, 1, 0]])  # Binary features: 0/1 values
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

# Analyze results
print(f"Predictions: {predictions}")
print(f"Top features: {clf.get_top_features(top_k=3)}")
```

## Training Options

### Option 1: Train from Raw CSV Data (Auto-generates sketches)
```bash
python run_binary_classification.py data.csv target_column --lg_k 16 --max_depth 6
```

### Option 2: Complete Sketch-Based Workflow (Recommended for Production)

Train from any CSV dataset using the 3-step sketch-based workflow:

```bash
# Step 1: Generate 2-column sketches from raw CSV
python tools/create_2col_sketches.py data.csv target_column --lg_k 16

# Step 2: Convert to 3-column training format
python tools/simple_convert_to_3col.py \
    examples/sketches/dataset_sketches/dataset_positive_2col_sketches_lg_k_16.csv \
    examples/sketches/dataset_sketches/dataset_negative_2col_sketches_lg_k_16.csv \
    examples/sketches/dataset_sketches/dataset_feature_mapping.json \
    examples/sketches/dataset_sketches/

# Step 3: Train from 3-column sketches
python tools/train_from_3col_sketches.py \
    examples/sketches/dataset_sketches/dataset_3col_sketches.csv \
    examples/sketches/dataset_sketches/dataset_feature_mapping.json \
    configs/training_config.yaml
```

## 🚀 Production Pipeline Scripts

For production deployments and customer demos, use these complete pipeline scripts:

### Script 1: Complete CSV → Model Pipeline
Transforms raw CSV data through the entire pipeline to a trained model:

```bash
# Complete pipeline: CSV → 2-col sketches → 3-col sketches → model
./du_pipeline.sh

# Uses DU dataset, outputs to DU_output/ directory
# Includes decision rules extraction for validation
```

### Script 2: 2-Column Sketches → Model Pipeline
Starts from existing 2-column sketches:

```bash
# From 2-column sketches to trained model
./pipeline_2col_to_model.sh <2col_sketches.csv> <lg_k> <output_dir> <target_column> [config.yaml]

# Examples:
./pipeline_2col_to_model.sh DU_output/DU_raw_2col_sketches_lg_k_16.csv 16 models/ tripOutcome
./pipeline_2col_to_model.sh mushroom_2col_sketches.csv 12 mushroom_models/ edible configs/mushroom_config.yaml
```

### Script 3: 3-Column Sketches → Model Training
Direct model training from prepared sketches:

```bash
# Train model from 3-column sketches
./pipeline_3col_to_model.sh <positive_3col.csv> <negative_3col.csv> <lg_k> <output_dir> [config.yaml]

# Examples:
./pipeline_3col_to_model.sh positive_sketches.csv negative_sketches.csv 16 models/
./pipeline_3col_to_model.sh DU_output/positive_3col_sketches_lg_k_16.csv DU_output/negative_3col_sketches_lg_k_16.csv 16 production_models/ configs/production_config.yaml
```

### Script 4: Advanced Classification Strategies Demo
Showcases business-ready classification strategies:

```bash
# Complete advanced classification demo
./demo_advanced_classification.sh [model.pkl] [test_data.csv] [strategy] [output_dir]

# Examples:
./demo_advanced_classification.sh                                    # Use defaults (DU model)
./demo_advanced_classification.sh model.pkl data.csv class_weightage demo/
./demo_advanced_classification.sh model.pkl data.csv all output/     # Test all strategies
```

## 🌲 Random Forest Compatibility

**Feature Subsampling** for ensemble methods (bootstrap sampling not possible with pre-aggregated sketches):

### Supported Subsampling Strategies

```python
from theta_sketch_tree import ThetaSketchDecisionTreeClassifier

# Different feature subsampling strategies
trees = [
    ThetaSketchDecisionTreeClassifier(max_features='sqrt', random_state=i),    # √n features
    ThetaSketchDecisionTreeClassifier(max_features='log2', random_state=i),    # log₂(n) features
    ThetaSketchDecisionTreeClassifier(max_features=0.3, random_state=i),       # 30% of features
    ThetaSketchDecisionTreeClassifier(max_features=15, random_state=i),        # Exactly 15 features
    ThetaSketchDecisionTreeClassifier(max_features=None, random_state=i),      # All features
]

# Train ensemble with different feature subsets
for i, tree in enumerate(trees):
    tree.fit(sketch_data, feature_mapping)
    print(f"Tree {i}: {tree.n_features_in_} total, ~{tree.max_features} subsampled")
```

### Random Forest Simulation

```python
# Simulate Random Forest with sketch-based trees
class SketchRandomForest:
    def __init__(self, n_estimators=100, max_features='sqrt', random_state=None):
        self.trees = [
            ThetaSketchDecisionTreeClassifier(
                max_features=max_features,
                random_state=random_state + i if random_state else i
            ) for i in range(n_estimators)
        ]

    def fit(self, sketch_data, feature_mapping):
        for tree in self.trees:
            tree.fit(sketch_data, feature_mapping)

    def predict(self, X):
        # Majority voting across trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.round(np.mean(predictions, axis=0)).astype(int)

# Usage
rf = SketchRandomForest(n_estimators=50, max_features='sqrt', random_state=42)
rf.fit(sketch_data, feature_mapping)
ensemble_predictions = rf.predict(X_test)
```

### Why Feature Subsampling Only?

🔍 **Technical Limitation**: Bootstrap sampling requires access to individual data rows, but theta sketches are pre-aggregated summary statistics. The original data rows are no longer available.

✅ **Solution**: Feature subsampling provides the diversity needed for ensemble methods while working within sketch constraints.

🎯 **Benefits**:
- Reduces overfitting through feature diversity
- Enables ensemble methods (Random Forest, Extra Trees)
- Maintains computational efficiency
- Full sklearn API compatibility

### Demo

```bash
# See comprehensive feature subsampling demo
python demo_feature_subsampling.py
```

## 🎯 Advanced Classification Strategies

Post-processing classification with multiple business strategies:

### Available Strategies

| Strategy | Use Case | Key Feature |
|----------|----------|-------------|
| **Class Weightage** | Cost-sensitive decisions | Adjusts threshold based on business costs |
| **Binomial P-value** | Statistical rigor | Ensures significance vs. population baseline |
| **Ratio Threshold** | Simple rules | Interpretable percentage cutoffs |
| **Confidence Intervals** | Uncertainty-aware | Wilson score intervals for reliability |

### Business Scenarios

```yaml
# Customer Retention (False negatives expensive)
strategy: "class_weightage"
positive_class_weight: 5.0  # Losing customers costs 5x more
negative_class_weight: 1.0
min_collateral_damage: 100  # Need ≥100 safe samples

# Fraud Detection (Need statistical confidence)
strategy: "binomial_pvalue"
population_positive_rate: 0.001  # Baseline fraud rate
significance_level: 0.01         # 99% confidence required

# Marketing Campaign (Simple + reliable)
strategy: "ratio_threshold"
ratio_threshold: 0.25           # 25% positive rate required
min_samples_for_prediction: 50  # Minimum sample size

# Medical Diagnosis (Uncertainty-aware)
strategy: "confidence_intervals"
confidence_level: 0.99          # 99% confidence intervals
decision_threshold: 0.10        # Conservative threshold
```

### Quick Demo
```bash
# See all strategies in action
./venv/bin/python demo_advanced_strategies.py

# Test with your model
./venv/bin/python tools/advanced_classifier.py \
    --model your_model.pkl \
    --data your_data.csv \
    --config configs/classification_strategies.yaml \
    --summary
```

**Real Example - Mushroom Dataset:**
```bash
# Generate mushroom sketches
python tools/create_2col_sketches.py tests/resources/agaricus-lepiota.csv class --lg_k 12

# Convert to training format
python tools/simple_convert_to_3col.py \
    examples/sketches/agaricus_lepiota_sketches/agaricus_lepiota_positive_2col_sketches_lg_k_19.csv \
    examples/sketches/agaricus_lepiota_sketches/agaricus_lepiota_negative_2col_sketches_lg_k_19.csv \
    examples/sketches/agaricus_lepiota_sketches/agaricus_lepiota_feature_mapping.json \
    examples/sketches/agaricus_lepiota_sketches/

# Train model
python tools/train_from_3col_sketches.py \
    examples/sketches/agaricus_lepiota_sketches/agaricus_lepiota_3col_sketches.csv \
    examples/sketches/agaricus_lepiota_sketches/agaricus_lepiota_feature_mapping.json \
    configs/mushroom_training_config.yaml
```

### Option 3: Legacy Sketch Training
```bash
# Train from pre-existing sketch files (legacy format)
python tools/train_from_sketches.py positive.csv negative.csv config.yaml
```

**Benefits of Sketch-Based Workflow:**
- 🔄 **Reproducible**: Same sketches can be reused for multiple training runs
- ⚡ **Fast**: Skip sketch generation for repeated experiments
- 🔒 **Privacy**: Share sketches without exposing raw data
- 📦 **Portable**: Sketches are much smaller than original datasets
- 🎛️ **Flexible**: Works with any binary classification CSV dataset

**Specialized Tools**: See `tools/tools-guide.md` for sketch generation, conversion, and tree comparison utilities.

## Key Features

- **🎯 Sketch-based Training**: Memory-efficient learning from theta sketches
- **📊 Standard Inference**: sklearn-compatible predictions on raw binary data
- **🌳 Advanced Pruning**: 4 pruning methods (cost-complexity, validation, reduced-error, min-impurity)
- **⚖️ Multiple Criteria**: Gini, Entropy, Gain Ratio, Binomial, Chi-Square
- **💾 Model Persistence**: Save/load trained models with hyperparameters
- **🔧 Missing Values**: Robust handling with majority-vote strategy
- **📈 Feature Importance**: Built-in weighted impurity decrease calculation
- **⚡ High Performance**: >400K predictions/second with linear scaling

## Installation

```bash
# Clone and setup
git clone https://github.com/your-org/theta-sketch-tree.git
cd Decision-trees_CART_sketch

# Install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Pruning Examples

Prevent overfitting with built-in pruning methods:

```bash
# Cost-complexity pruning (recommended)
python run_binary_classification.py data.csv target --pruning cost_complexity

# Validation-based pruning
python run_binary_classification.py data.csv target --pruning validation
```

```python
# Python API
clf = ThetaSketchDecisionTreeClassifier(
    criterion='gini',
    pruning='cost_complexity',
    min_impurity_decrease=0.01
)
```

## Performance

| Metric | Performance |
|--------|-------------|
| Training Speed | ~1.5s for 100 features |
| Prediction Throughput | >400K samples/sec |
| Memory Usage | Linear scaling |
| Test Coverage | 97% (420 tests) |

## Documentation

📖 **Comprehensive documentation in [`docs/`](docs/index.md)**:

### Essential Documentation
1. **[Quick Start Guide](docs/01-quickstart.md)** - 5-minute setup and basic usage
2. **[User Guide](docs/02-user-guide.md)** - Complete API usage with examples
3. **[Architecture Overview](docs/03-architecture.md)** - System design and components
4. **[Algorithm Reference](docs/04-algorithms.md)** - Mathematical foundations and pseudocode
5. **[API Reference](docs/05-api-reference.md)** - Complete method specifications

### Specialized Guides
6. **[Performance Guide](docs/06-performance.md)** - Optimization and benchmarking
7. **[Testing Guide](docs/07-testing.md)** - Validation strategies
8. **[Troubleshooting Guide](docs/08-troubleshooting.md)** - Common issues and solutions

### Advanced Topics
9. **[Data Formats](docs/09-data-formats.md)** - Input requirements and validation
10. **[Hyperparameter Tuning](docs/10-hyperparameter-tuning.md)** - Model optimization
11. **[Deployment Guide](docs/11-deployment.md)** - Production deployment strategies

**Start here**: New users should begin with the [Quick Start Guide](docs/01-quickstart.md)

## Testing

```bash
# Unit tests with coverage requirements
./test-unit.sh

# Integration tests (mushroom examples) without coverage pressure
./test-integration.sh

# All tests with coverage capture
./test-all.sh
```

## Requirements

- Python 3.8+
- Apache DataSketches Python library
- scikit-learn, NumPy, SciPy, Pandas

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`./test-unit.sh`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push and open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- 📖 **Documentation**: [docs/](docs/index.md)
- 🐛 **Issues**: GitHub Issues
- 💬 **Discussions**: GitHub Discussions

---

**Enterprise Ready**: Production-quality implementation with professional logging, comprehensive validation, and clean sklearn-compatible interfaces.