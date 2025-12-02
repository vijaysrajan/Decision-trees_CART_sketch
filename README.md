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

📖 **Complete documentation in [`docs/`](docs/)**:

- **[User Guide](docs/user_guide.md)** - Comprehensive usage examples
- **[Architecture Overview](docs/architecture-overview.md)** - System design and components
- **[Algorithm Details](docs/03_algorithms.md)** - Mathematical foundations
- **[API Reference](docs/05_api_design.md)** - Method signatures and examples
- **[Performance Analysis](docs/performance_analysis.md)** - Benchmarks and optimization
- **[Testing Strategy](TESTING.md)** - Unit vs integration test approach

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

- 📖 **Documentation**: [docs/](docs/)
- 🐛 **Issues**: GitHub Issues
- 💬 **Discussions**: GitHub Discussions

---

**Enterprise Ready**: Production-quality implementation with professional logging, comprehensive validation, and clean sklearn-compatible interfaces.