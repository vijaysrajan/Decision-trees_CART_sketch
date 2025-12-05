# Hyperparameter Tuning Guide

## Overview

This guide provides comprehensive strategies for optimizing Theta Sketch Decision Tree hyperparameters to achieve optimal performance across different datasets and use cases.

## Key Hyperparameters

### Core Tree Parameters

#### **max_depth** (Most Important)
- **Range**: 3-20 (typical), None for unlimited
- **Impact**: Controls model complexity and overfitting
- **Tuning Strategy**:
  ```python
  # Start with dataset-appropriate depth
  depth_suggestions = {
      'small_dataset': range(8, 15),      # <10K samples
      'medium_dataset': range(6, 12),     # 10K-100K samples
      'large_dataset': range(4, 10),      # >100K samples
  }
  ```

#### **criterion** (Performance vs Accuracy)
- **Options**: `'gini'`, `'entropy'`, `'gain_ratio'`, `'binomial'`, `'chi_square'`
- **Recommendations**:
  ```python
  criterion_guide = {
      'balanced_data': 'gini',           # Fast, reliable
      'imbalanced_data': 'chi_square',   # Statistical significance
      'categorical_heavy': 'gain_ratio',  # Handles bias
      'high_accuracy': 'entropy',        # Information theory
      'small_samples': 'binomial'        # Statistical testing
  }
  ```

#### **min_samples_split**
- **Range**: 2-50
- **Purpose**: Prevents overfitting by requiring minimum samples to split
- **Strategy**: Start with 2, increase for noisy data

#### **min_impurity_decrease**
- **Range**: 0.0-0.1
- **Purpose**: Pre-pruning based on information gain threshold
- **Strategy**: Start with 0.0, increase to 0.01-0.05 for aggressive pruning

### Pruning Parameters

#### **pruning**
- **Options**: `None`, `'cost_complexity'`, `'validation'`, `'reduced_error'`
- **Recommendations**:
  ```python
  pruning_strategy = {
      'production': 'cost_complexity',    # Best generalization
      'small_data': 'validation',         # Cross-validation
      'interpretability': 'reduced_error', # Simplest tree
      'development': None                 # No pruning for debugging
  }
  ```

## Systematic Tuning Approach

### 1. Grid Search Implementation

```python
from sklearn.model_selection import ParameterGrid
from theta_sketch_tree import ThetaSketchDecisionTreeClassifier

def tune_hyperparameters(sketch_data, feature_mapping, validation_func):
    """Systematic hyperparameter tuning with grid search."""

    # Define parameter grid
    param_grid = {
        'criterion': ['gini', 'entropy', 'chi_square'],
        'max_depth': [5, 8, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_impurity_decrease': [0.0, 0.01, 0.05],
        'pruning': [None, 'cost_complexity']
    }

    best_score = 0
    best_params = None
    results = []

    for params in ParameterGrid(param_grid):
        # Train model
        clf = ThetaSketchDecisionTreeClassifier(**params, verbose=0)
        clf.fit(sketch_data, feature_mapping)

        # Evaluate performance
        score = validation_func(clf)
        results.append({'params': params, 'score': score})

        # Track best
        if score > best_score:
            best_score = score
            best_params = params

        print(f"Score: {score:.4f}, Params: {params}")

    return best_params, best_score, results

# Example usage
def validation_accuracy(clf):
    """Custom validation function."""
    # Implement your validation logic
    X_val, y_val = load_validation_data()
    predictions = clf.predict(X_val)
    return accuracy_score(y_val, predictions)

best_params, score, all_results = tune_hyperparameters(
    sketch_data, feature_mapping, validation_accuracy
)
```

### 2. Bayesian Optimization

```python
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

def bayesian_optimization(sketch_data, feature_mapping, n_calls=50):
    """Bayesian optimization for efficient hyperparameter search."""

    # Define search space
    space = [
        Categorical(['gini', 'entropy', 'chi_square'], name='criterion'),
        Integer(3, 20, name='max_depth'),
        Integer(2, 20, name='min_samples_split'),
        Real(0.0, 0.1, name='min_impurity_decrease'),
        Categorical([None, 'cost_complexity'], name='pruning')
    ]

    def objective(params):
        """Objective function to minimize (negative accuracy)."""
        criterion, max_depth, min_samples_split, min_impurity_decrease, pruning = params

        clf = ThetaSketchDecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            pruning=pruning,
            verbose=0
        )

        clf.fit(sketch_data, feature_mapping)
        score = validation_accuracy(clf)

        return -score  # Minimize negative accuracy

    # Run optimization
    result = gp_minimize(objective, space, n_calls=n_calls, random_state=42)

    # Extract best parameters
    best_params = {
        'criterion': result.x[0],
        'max_depth': result.x[1],
        'min_samples_split': result.x[2],
        'min_impurity_decrease': result.x[3],
        'pruning': result.x[4]
    }

    return best_params, -result.fun
```

### 3. Multi-Objective Optimization

```python
def multi_objective_tuning(sketch_data, feature_mapping):
    """Balance accuracy, speed, and interpretability."""

    def evaluate_model(params):
        """Evaluate multiple objectives."""
        clf = ThetaSketchDecisionTreeClassifier(**params, verbose=0)

        # Measure training time
        start_time = time.time()
        clf.fit(sketch_data, feature_mapping)
        training_time = time.time() - start_time

        # Measure accuracy
        accuracy = validation_accuracy(clf)

        # Measure interpretability (inverse of tree complexity)
        interpretability = 1.0 / (clf.get_n_nodes() + 1)

        # Measure prediction speed
        X_test = generate_test_data(1000)
        start_time = time.time()
        clf.predict(X_test)
        prediction_speed = 1000 / (time.time() - start_time)  # samples/sec

        return {
            'accuracy': accuracy,
            'training_time': training_time,
            'interpretability': interpretability,
            'prediction_speed': prediction_speed
        }

    # Define weights for objectives
    weights = {
        'accuracy': 0.5,
        'training_time': -0.2,      # Negative for minimization
        'interpretability': 0.2,
        'prediction_speed': 0.1
    }

    def combined_score(metrics):
        """Weighted combination of objectives."""
        return sum(weights[key] * value for key, value in metrics.items())

    # Test parameter combinations
    best_score = float('-inf')
    best_params = None

    for params in parameter_combinations:
        metrics = evaluate_model(params)
        score = combined_score(metrics)

        if score > best_score:
            best_score = score
            best_params = params

        print(f"Score: {score:.4f}, Metrics: {metrics}")

    return best_params, best_score
```

## Dataset-Specific Strategies

### Small Datasets (<1K samples)

```python
small_dataset_config = {
    'max_depth': 8,                    # Prevent overfitting
    'min_samples_split': 5,            # Require more samples
    'min_impurity_decrease': 0.01,     # Early stopping
    'pruning': 'validation',           # Cross-validation pruning
    'criterion': 'chi_square'          # Statistical significance
}
```

### Medium Datasets (1K-100K samples)

```python
medium_dataset_config = {
    'max_depth': 12,                   # Balanced complexity
    'min_samples_split': 2,            # Default splitting
    'min_impurity_decrease': 0.0,      # Let algorithm decide
    'pruning': 'cost_complexity',      # Best generalization
    'criterion': 'gini'                # Fast and effective
}
```

### Large Datasets (>100K samples)

```python
large_dataset_config = {
    'max_depth': 8,                    # Control training time
    'min_samples_split': 10,           # Aggressive pruning
    'min_impurity_decrease': 0.005,    # Moderate threshold
    'pruning': 'cost_complexity',      # Prevent overfitting
    'criterion': 'gini'                # Fastest criterion
}
```

## Advanced Tuning Techniques

### 1. Feature Subset Selection

```python
def feature_subset_tuning(sketch_data, feature_mapping, max_features=100):
    """Tune feature selection alongside hyperparameters."""

    all_features = list(feature_mapping.keys())

    # Score features by discriminative power
    feature_scores = {}
    for feature in all_features:
        if feature != 'total':
            pos_present = sketch_data['positive'][feature][0].get_estimate()
            neg_present = sketch_data['negative'][feature][0].get_estimate()

            # Calculate relative difference
            total_pos = sketch_data['positive']['total'].get_estimate()
            total_neg = sketch_data['negative']['total'].get_estimate()

            pos_rate = pos_present / total_pos if total_pos > 0 else 0
            neg_rate = neg_present / total_neg if total_neg > 0 else 0

            feature_scores[feature] = abs(pos_rate - neg_rate)

    # Try different feature subset sizes
    best_score = 0
    best_config = None

    for n_features in [25, 50, 100, 200]:
        if n_features > len(feature_scores):
            continue

        # Select top features
        top_features = sorted(feature_scores.items(),
                            key=lambda x: x[1], reverse=True)[:n_features]

        selected_mapping = {name: idx for idx, (name, _) in enumerate(top_features)}

        # Tune hyperparameters for this feature subset
        config, score, _ = tune_hyperparameters(
            sketch_data, selected_mapping, validation_accuracy
        )

        if score > best_score:
            best_score = score
            best_config = {
                'hyperparams': config,
                'feature_mapping': selected_mapping,
                'n_features': n_features,
                'score': score
            }

        print(f"Features: {n_features}, Score: {score:.4f}")

    return best_config
```

### 2. Ensemble Configuration

```python
def ensemble_hyperparameter_tuning(sketch_data, feature_mapping):
    """Tune parameters for ensemble of diverse trees."""

    # Generate diverse configurations
    diverse_configs = [
        {'criterion': 'gini', 'max_depth': 8, 'min_samples_split': 2},
        {'criterion': 'entropy', 'max_depth': 12, 'min_samples_split': 5},
        {'criterion': 'chi_square', 'max_depth': 6, 'min_samples_split': 10},
        {'criterion': 'gain_ratio', 'max_depth': 10, 'min_samples_split': 3},
    ]

    ensemble_classifiers = []

    for config in diverse_configs:
        clf = ThetaSketchDecisionTreeClassifier(**config, verbose=0)
        clf.fit(sketch_data, feature_mapping)
        ensemble_classifiers.append(clf)

    # Evaluate ensemble performance
    def ensemble_predict(X):
        """Majority vote prediction."""
        predictions = np.array([clf.predict(X) for clf in ensemble_classifiers])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, predictions)

    # Test ensemble
    X_val, y_val = load_validation_data()
    ensemble_predictions = ensemble_predict(X_val)
    ensemble_accuracy = accuracy_score(y_val, ensemble_predictions)

    return ensemble_classifiers, ensemble_accuracy
```

## Performance vs Accuracy Tradeoffs

### Speed-Optimized Configuration

```python
speed_config = {
    'criterion': 'gini',              # Fastest criterion
    'max_depth': 6,                   # Shallow tree
    'min_samples_split': 10,          # Large splits only
    'min_impurity_decrease': 0.05,    # Aggressive early stopping
    'pruning': None                   # No post-processing
}
```

### Accuracy-Optimized Configuration

```python
accuracy_config = {
    'criterion': 'entropy',           # Best information gain
    'max_depth': 15,                  # Deep tree
    'min_samples_split': 2,           # Fine-grained splits
    'min_impurity_decrease': 0.0,     # No early stopping
    'pruning': 'cost_complexity'      # Best generalization
}
```

### Memory-Optimized Configuration

```python
memory_config = {
    'criterion': 'gini',              # Efficient computation
    'max_depth': 8,                   # Limited tree size
    'min_samples_split': 20,          # Fewer nodes
    'min_impurity_decrease': 0.02,    # Early stopping
    'pruning': 'reduced_error'        # Aggressive pruning
}
```

## Validation Strategies

### Cross-Validation for Sketch Data

```python
def sketch_cross_validation(sketch_data, feature_mapping, cv_folds=5):
    """Custom cross-validation for sketch-based training."""

    # Since we can't easily split sketch data, we use bootstrap sampling
    n_bootstrap = cv_folds
    scores = []

    for fold in range(n_bootstrap):
        # Create bootstrap sample of features
        feature_sample = random.sample(
            list(feature_mapping.keys()),
            k=int(0.8 * len(feature_mapping))
        )

        sample_mapping = {feat: idx for idx, feat in enumerate(feature_sample)}

        # Train on sample
        clf = ThetaSketchDecisionTreeClassifier(verbose=0)
        clf.fit(sketch_data, sample_mapping)

        # Evaluate on validation set
        score = validation_accuracy(clf)
        scores.append(score)

        print(f"Fold {fold+1}: {score:.4f}")

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    print(f"CV Score: {mean_score:.4f} ± {std_score:.4f}")
    return mean_score, std_score
```

## Production Deployment Configuration

### Recommended Production Settings

```python
production_config = {
    # Core parameters
    'criterion': 'gini',              # Reliable and fast
    'max_depth': 10,                  # Balanced complexity
    'min_samples_split': 5,           # Moderate pruning
    'min_impurity_decrease': 0.01,    # Prevent overfitting

    # Pruning for generalization
    'pruning': 'cost_complexity',     # Best practice

    # Performance settings
    'verbose': 0,                     # Silent operation

    # Memory efficiency
    # (Implemented through feature selection if needed)
}

# Example deployment
def deploy_optimized_model(sketch_data, feature_mapping):
    """Deploy production-optimized model."""

    # Apply feature selection if needed
    if len(feature_mapping) > 200:
        optimized_mapping = select_top_features(sketch_data, feature_mapping, 150)
    else:
        optimized_mapping = feature_mapping

    # Train with production config
    clf = ThetaSketchDecisionTreeClassifier(**production_config)
    clf.fit(sketch_data, optimized_mapping)

    # Validate performance meets requirements
    validation_score = validate_model_performance(clf)

    if validation_score < 0.85:  # Minimum accuracy threshold
        raise ValueError(f"Model accuracy {validation_score:.3f} below threshold")

    # Save optimized model
    from theta_sketch_tree.model_persistence import ModelPersistence
    ModelPersistence.save_model(clf, 'production_model.pkl')

    return clf, validation_score
```

---

## Next Steps

- **Performance**: See [Performance Guide](06-performance.md) for optimization techniques
- **User Guide**: Review [User Guide](02-user-guide.md) for API usage
- **Testing**: Check [Testing Guide](07-testing.md) for validation strategies
- **Troubleshooting**: See [Troubleshooting Guide](08-troubleshooting.md) for common issues