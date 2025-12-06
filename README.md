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

# Train classifier
clf = ThetaSketchDecisionTreeClassifier(criterion='gini', max_depth=10)
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