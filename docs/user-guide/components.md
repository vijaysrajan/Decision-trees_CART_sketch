# Component Guide
*User-focused guide: When to use which components and how they work together*

## Overview

The Theta Sketch Decision Tree system is built from modular components that you can use independently or together. This guide explains **when to use which components** and **how to combine them** for different use cases.

## Core User-Facing Components

### 🎯 ThetaSketchDecisionTreeClassifier
**What it does**: Main API for training and prediction
**When to use**: This is your primary interface - use for all standard classification tasks
**How to use**:

```python
from theta_sketch_tree import ThetaSketchDecisionTreeClassifier

# Standard usage
clf = ThetaSketchDecisionTreeClassifier(
    criterion='gini',
    max_depth=10,
    pruning='cost_complexity'
)

# Train from sketch data
clf.fit(sketch_data, feature_mapping)

# Make predictions on raw data
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
```

**Key features**:
- ✅ sklearn-compatible API
- ✅ Built-in validation and error handling
- ✅ Comprehensive logging
- ✅ Model persistence support

---

### 📊 Data Loading Components

#### SketchLoader
**What it does**: Loads theta sketches from CSV files
**When to use**: When you have pre-computed sketches in CSV format
**How to use**:

```python
from theta_sketch_tree import load_sketches

# Dual CSV mode (separate positive/negative files)
sketch_data = load_sketches('positive_class.csv', 'negative_class.csv')

# Single CSV mode (one-vs-all)
sketch_data = load_sketches(
    csv_path='all_data.csv',
    target_positive='class_yes',
    target_negative='class_no'
)
```

**Use cases**:
- Loading sketches computed from big data pipelines
- Supporting both dual-class and one-vs-all scenarios
- Automatic sketch format validation

#### ConfigParser
**What it does**: Loads hyperparameters and feature mappings from YAML
**When to use**: When you want to externalize configuration
**How to use**:

```python
from theta_sketch_tree import load_config

config = load_config('model_config.yaml')
# Returns: {'hyperparameters': {...}, 'feature_mapping': {...}, 'targets': {...}}

# Use in classifier
clf = ThetaSketchDecisionTreeClassifier(**config['hyperparameters'])
clf.fit(sketch_data, config['feature_mapping'])
```

---

### 🔧 Utility Components

#### ClassifierUtils
**What it does**: Convenience methods for common tasks
**When to use**: For quick model setup, analysis, and workflow automation
**How to use**:

```python
from theta_sketch_tree.classifier_utils import ClassifierUtils

# One-line training from files
clf = ClassifierUtils.fit_from_csv(
    positive_csv='pos.csv',
    negative_csv='neg.csv',
    config_path='config.yaml'
)

# Feature importance analysis
importance_dict = ClassifierUtils.get_feature_importance_dict(clf)
top_features = ClassifierUtils.get_top_features(clf, top_k=5)

# Model persistence
ClassifierUtils.save_model(clf, 'my_model.pkl', include_sketches=False)
loaded_clf = ClassifierUtils.load_model('my_model.pkl')
```

**Use cases**:
- Rapid prototyping
- One-liner model training
- Feature importance analysis
- Model management

#### ModelPersistence
**What it does**: Save and load trained models
**When to use**: For production deployment and model versioning
**How to use**:

```python
from theta_sketch_tree.model_persistence import ModelPersistence

# Save model
ModelPersistence.save_model(clf, 'production_model.pkl')

# Load model
clf = ModelPersistence.load_model('production_model.pkl')

# Get model metadata
info = ModelPersistence.get_model_info(clf)
print(f"Tree depth: {info['tree_depth']}")
print(f"Number of features: {info['n_features']}")
```

---

### ⚙️ Advanced Components

#### Split Criteria
**What it does**: Different algorithms for evaluating split quality
**When to use**: When you need specialized criteria for your domain
**Available criteria**:
- `'gini'` - General purpose (recommended)
- `'entropy'` - Information theory approach
- `'gain_ratio'` - Handles bias toward multi-valued features
- `'binomial'` - Statistical significance testing
- `'chi_square'` - Chi-square test for independence

**How to use**:
```python
# Specify criterion in classifier
clf = ThetaSketchDecisionTreeClassifier(criterion='entropy')

# Or create custom criterion
from theta_sketch_tree.criteria import SplitCriterion

class CustomCriterion(SplitCriterion):
    def evaluate_split(self, left_counts, right_counts, parent_counts):
        # Your custom logic here
        return score
```

#### Pruning Methods
**What it does**: Reduces overfitting by removing unnecessary branches
**When to use**: When your trees are overfitting (high training, low validation accuracy)
**Available methods**:
- `'cost_complexity'` - Balanced approach (recommended)
- `'validation'` - Uses validation data for pruning decisions
- `'reduced_error'` - Conservative accuracy-preserving pruning
- `'min_impurity'` - Removes splits with minimal benefit

**How to use**:
```python
clf = ThetaSketchDecisionTreeClassifier(
    pruning='cost_complexity',
    min_impurity_decrease=0.01,
    validation_fraction=0.2
)
```

---

## Usage Patterns

### Pattern 1: Quick Start (Simplest)
**Use case**: You have sketch CSVs and want predictions fast

```python
from theta_sketch_tree.classifier_utils import ClassifierUtils

# One line training
clf = ClassifierUtils.fit_from_csv('pos.csv', 'neg.csv', 'config.yaml')

# Make predictions
predictions = clf.predict(X_test)
```

### Pattern 2: Standard Workflow (Most Common)
**Use case**: Production ML pipeline with configuration control

```python
from theta_sketch_tree import ThetaSketchDecisionTreeClassifier, load_sketches, load_config

# Load data and config
sketch_data = load_sketches('positive.csv', 'negative.csv')
config = load_config('production_config.yaml')

# Configure classifier
clf = ThetaSketchDecisionTreeClassifier(
    criterion='gini',
    max_depth=config['hyperparameters']['max_depth'],
    pruning='cost_complexity',
    verbose=1  # Enable logging
)

# Train and predict
clf.fit(sketch_data, config['feature_mapping'])
predictions = clf.predict(X_test)

# Analyze results
feature_importance = clf.feature_importances_
top_features = clf.get_top_features(top_k=10)
```

### Pattern 3: Advanced Customization
**Use case**: Research or specialized domains requiring custom criteria/pruning

```python
from theta_sketch_tree import ThetaSketchDecisionTreeClassifier
from theta_sketch_tree.criteria import register_criterion
from theta_sketch_tree.pruning import custom_prune

# Register custom criterion
@register_criterion('domain_specific')
class DomainSpecificCriterion:
    def evaluate_split(self, left_counts, right_counts, parent_counts):
        # Domain-specific split evaluation
        return score

# Use custom components
clf = ThetaSketchDecisionTreeClassifier(
    criterion='domain_specific',  # Use custom criterion
    pruning='none',               # Handle pruning manually
    verbose=2                     # Detailed logging
)

clf.fit(sketch_data, feature_mapping)

# Custom post-training pruning
clf.tree_ = custom_prune(clf.tree_, validation_data, alpha=0.05)
```

### Pattern 4: Production Deployment
**Use case**: Deployed model serving predictions at scale

```python
import joblib
from theta_sketch_tree.model_persistence import ModelPersistence

# Training (done offline)
clf = train_model()  # Your training process
ModelPersistence.save_model(clf, 'production_model_v1.2.pkl')

# Deployment (load once, predict many times)
clf = ModelPersistence.load_model('production_model_v1.2.pkl')

def predict_endpoint(features):
    """Fast prediction endpoint"""
    # Input validation happens automatically
    predictions = clf.predict(features)
    probabilities = clf.predict_proba(features)

    return {
        'predictions': predictions.tolist(),
        'probabilities': probabilities.tolist(),
        'model_version': '1.2'
    }
```

---

## Component Dependencies

Understanding how components depend on each other helps with debugging:

```
ThetaSketchDecisionTreeClassifier
├── TreeOrchestrator (coordinates building)
│   ├── SplitFinder (finds best splits)
│   │   ├── Criteria (evaluates split quality)
│   │   └── ValidationUtils (validates inputs)
│   ├── TreeBuilder (creates tree structure)
│   └── Pruning (removes overfitting)
├── TreeTraverser (handles prediction)
│   └── ValidationUtils (validates inputs)
├── FeatureImportance (calculates importances)
└── TreeLogger (structured logging)

SketchLoader
├── ConfigParser (loads YAML configs)
└── ValidationUtils (validates sketch data)

ModelPersistence
├── TreeNode serialization
└── Hyperparameter preservation
```

## Debugging Guide

### Common Issues and Solutions

1. **"Sketch format not recognized"**
   - **Problem**: CSV doesn't match expected format
   - **Solution**: Check CSV has columns: `identifier,sketch_present,sketch_absent`
   - **Component**: SketchLoader

2. **"Feature mapping mismatch"**
   - **Problem**: Feature names in sketch don't match mapping
   - **Solution**: Ensure config.yaml feature_mapping matches CSV identifiers
   - **Component**: ConfigParser, ThetaSketchDecisionTreeClassifier

3. **"Low prediction accuracy"**
   - **Problem**: Overfitting or poor hyperparameters
   - **Solution**: Try different pruning methods or split criteria
   - **Component**: Pruning, Criteria

4. **"Slow training"**
   - **Problem**: Too many features or deep trees
   - **Solution**: Set max_depth, min_samples_split limits
   - **Component**: TreeOrchestrator configuration

Use `verbose=2` in the classifier for detailed debugging information from all components.

This modular design lets you mix and match components based on your specific needs, from simple one-liner usage to advanced custom implementations.