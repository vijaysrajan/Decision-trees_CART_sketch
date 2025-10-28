# Theta Sketch Decision Tree Classifier

A production-grade CART Decision Tree Classifier that trains on theta sketches but performs inference on raw tabular data. Designed for medical/market basket analysis scenarios with sklearn compatibility.

## Project Status

🚧 **In Development** - Design Phase Complete

## Key Features

- ✅ Trains on theta sketches (privacy-preserving set summaries)
- ✅ Infers on binary tabular data (sklearn-compatible)
- ✅ Multiple split criteria (Gini, Entropy, Gain Ratio, Binomial, Chi-Square)
- ✅ Missing value handling (majority path method)
- ✅ Class weighting for imbalanced data
- ✅ Pre and post-pruning support
- ✅ Feature importance computation
- ✅ ROC/AUC metrics built-in

## Architecture

### Training Phase
- **Input**: CSV file with serialized theta sketches + YAML config
- **Process**: Build decision tree using sketch set operations
- **Output**: Trained tree structure

### Inference Phase
- **Input**: Binary feature matrix (0/1 values)
- **Process**: Tree traversal with missing value handling
- **Output**: Class predictions and probabilities

## Design Documents

All design documents are in the `docs/` directory:

1. [High-Level Architecture](docs/01_high_level_architecture.md)
2. [Low-Level Design](docs/02_low_level_design.md)
3. [Algorithm Pseudocode](docs/03_algorithms.md)
4. [Data Format Specifications](docs/04_data_formats.md)
5. [API Design](docs/05_api_design.md)
6. [Module Structure](docs/06_module_structure.md)
7. [Testing Strategy](docs/07_testing_strategy.md)
8. [Implementation Roadmap](docs/08_implementation_roadmap.md)
9. [Design Corrections](docs/CORRECTIONS.md)

## Installation (Coming Soon)

```bash
# Not yet available - in development
pip install theta-sketch-tree
```

## Usage Example (Planned API)

```python
from theta_sketch_tree import ThetaSketchDecisionTreeClassifier
import numpy as np

# Training
clf = ThetaSketchDecisionTreeClassifier(
    criterion='gini',
    max_depth=10,
    verbose=1
)
clf.fit('sketches.csv', 'config.yaml')

# Inference on binary data
X_test = np.array([
    [1, 0, 1],  # Binary features
    [0, 1, 0],
])
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
```

## Development

### Requirements
- Python 3.8+
- Apache DataSketches Python library
- scikit-learn
- NumPy, SciPy, Pandas

### Project Structure

```
Decision-trees_CART_sketch/
├── docs/                    # Design documentation
├── theta_sketch_tree/       # Main package (to be implemented)
├── tests/                   # Test suite (to be implemented)
├── examples/                # Example scripts (to be implemented)
└── benchmarks/              # Performance benchmarks (to be implemented)
```

## Timeline

- **Phase 1 (Complete)**: Design documentation
- **Phase 2 (Weeks 1-3)**: Core implementation
- **Phase 3 (Weeks 4-5)**: Advanced features
- **Phase 4 (Week 6)**: Testing and polish
- **Target**: v0.1.0 release in 6 weeks

## Contributing

This is currently a personal project. Contributions will be accepted after v0.1.0 release.

## License

To be determined.

## Contact

[Your contact information]

---

**Note**: This project is in active development. The API and features described above are planned but not yet implemented. See `docs/08_implementation_roadmap.md` for the complete development plan.
