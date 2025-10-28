# Data Format Specifications
## Theta Sketch Decision Tree Classifier

---

## Table of Contents
1. [CSV Sketch Format](#1-csv-sketch-format)
2. [Config File Format (YAML)](#2-config-file-format-yaml)
3. [Inference Data Format](#3-inference-data-format)
4. [Model Persistence Format](#4-model-persistence-format)
5. [Examples](#5-examples)

---

## 1. CSV Sketch Format

### Format Specification

```
<identifier>, <sketch_bytes>
```

**Columns**:
- **Column 1: Identifier** (string)
  - `<total>` or `""` (empty string): Population sketch (all records for this target class)
  - `<feature_name>`: Feature condition sketch (e.g., "age>30", "income>50k")
  - `<target_name>`: Target class sketch (e.g., "target_yes", "target_no")

- **Column 2: Sketch Bytes** (base64-encoded or hex-encoded string)
  - Serialized Apache DataSketches theta sketch
  - Encoding: base64 (default) or hex

### Rules

1. **One CSV file per dataset** containing sketches for all classes and features
2. **Target identification**: Target sketch names must match config file specification
3. **Feature sketches**: Must be present for both positive and negative target classes
4. **Total sketch**: Required for each target class (identifier: "total" or "")

### Example CSV Structure

```csv
identifier,sketch
,<base64_sketch_total_yes>
target_yes,<base64_sketch_target_yes>
age>30,<base64_sketch_age_yes>
income>50k,<base64_sketch_income_yes>
has_diabetes,<base64_sketch_diabetes_yes>
,<base64_sketch_total_no>
target_no,<base64_sketch_target_no>
age>30,<base64_sketch_age_no>
income>50k,<base64_sketch_income_no>
has_diabetes,<base64_sketch_diabetes_no>
```

**Note**: The order of rows doesn't matter, but grouping by target class aids readability.

### Detailed Format by Row Type

#### 1. Total/Population Sketch
```
<total or empty>, <sketch_bytes>
```
- **Purpose**: Represents all records for a target class
- **Identifier options**: "total", "" (empty string), or "<total>"
- **Example**: `,AgMDAAAazJMCAAAAAACAPwAAAAAAAAAA...`

#### 2. Feature Condition Sketch
```
<feature_name>, <sketch_bytes>
```
- **Purpose**: Represents records where feature condition is TRUE
- **Format**: `dimension=value`, `dimension>value`, `dimension<value`, or simple `item_name`
- **Examples**:
  - `age>30,AgMDAAAazJMCAAAAAACAPwAAAAAAAAAA...`
  - `income>50000,AgMDAAAazJMCAAAAAACAPwAAAAAAAAAA...`
  - `region=East,AgMDAAAazJMCAAAAAACAPwAAAAAAAAAA...`
  - `item_12345,AgMDAAAazJMCAAAAAACAPwAAAAAAAAAA...`

#### 3. Target Class Sketch
```
<target_name>, <sketch_bytes>
```
- **Purpose**: Identifies which records belong to this target class
- **Must match config file** target specification
- **Examples**:
  - `target_yes,AgMDAAAazJMCAAAAAACAPwAAAAAAAAAA...`
  - `target_no,AgMDAAAazJMCAAAAAACAPwAAAAAAAAAA...`
  - `positive,AgMDAAAazJMCAAAAAACAPwAAAAAAAAAA...`
  - `negative,AgMDAAAazJMCAAAAAACAPwAAAAAAAAAA...`

### Sketch Bytes Encoding

#### Base64 Encoding (Default)
```python
import base64
from datasketches import compact_theta_sketch

# Serialize sketch to bytes
sketch_bytes = sketch.serialize()

# Encode to base64 string for CSV
sketch_str = base64.b64encode(sketch_bytes).decode('ascii')

# Example output:
# "AgMDAAAazJMCAAAAAACAPwAAAAAAAAAA..."
```

#### Hex Encoding (Alternative)
```python
# Serialize sketch to bytes
sketch_bytes = sketch.serialize()

# Encode to hex string for CSV
sketch_str = sketch_bytes.hex()

# Example output:
# "020303000001accd3020000000000..."
```

### Deserialization

```python
import base64
from datasketches import compact_theta_sketch

# Read from CSV
sketch_str = row['sketch']

# Decode from base64
sketch_bytes = base64.b64decode(sketch_str)

# Deserialize to ThetaSketch object
sketch = compact_theta_sketch.deserialize(sketch_bytes)

# Get cardinality estimate
count = sketch.get_estimate()
```

### CSV Parsing Requirements

1. **Header**: Optional. If present, must be: `identifier,sketch` or `<identifier>,<sketch>`
2. **Delimiter**: Comma (`,`)
3. **Quoting**: Optional for identifiers, required if identifier contains commas
4. **Line endings**: Unix (`\n`), Windows (`\r\n`), or Mac (`\r`) - all supported
5. **Encoding**: UTF-8

### Validation Rules

The loader must validate:
- ✅ Each row has exactly 2 columns
- ✅ Sketch bytes can be decoded (base64 or hex)
- ✅ Sketch bytes can be deserialized to ThetaSketch
- ✅ Target sketches specified in config are present
- ✅ Feature sketches are present for both target classes
- ✅ Total sketch is present for each target class

---

## 2. Config File Format (YAML)

### Complete Specification

```yaml
# =============================================================================
# Target Class Specification
# =============================================================================
targets:
  positive: "target_yes"    # Name of positive class sketch in CSV
  negative: "target_no"     # Name of negative class sketch in CSV

# =============================================================================
# Tree Hyperparameters
# =============================================================================
hyperparameters:
  # --- Split Criteria ---
  criterion: "gini"
  # Options: "gini", "entropy", "gain_ratio", "binomial", "binomial_chi"
  # Default: "gini"

  # --- Tree Structure ---
  splitter: "best"
  # Options: "best", "random"
  # Default: "best"

  max_depth: 10
  # Maximum tree depth (null = unlimited)
  # Type: int or null
  # Default: null

  min_samples_split: 20
  # Minimum samples required to split an internal node
  # Type: int
  # Default: 2

  min_samples_leaf: 10
  # Minimum samples required in a leaf node
  # Type: int
  # Default: 1

  max_features: null
  # Max features to consider for splitting
  # Options: int (number), float (fraction), "sqrt", "log2", null (all)
  # Default: null

  # --- Statistical Tests ---
  min_pvalue: 0.05
  # Significance threshold for binomial/chi-square criteria
  # Type: float
  # Default: 0.05

  use_bonferroni: true
  # Apply Bonferroni correction for multiple testing
  # Type: bool
  # Default: true

  # --- Imbalanced Data Handling ---
  class_weight: "balanced"
  # Options: "balanced", null, or dict {0: weight0, 1: weight1}
  # Default: null

  use_weighted_gini: true
  # Use class weights in Gini impurity calculation
  # Type: bool
  # Default: true

  # --- Missing Value Handling ---
  missing_value_strategy: "majority"
  # Options: "majority", "zero", "error"
  # Default: "majority"

  # --- Pruning ---
  pruning: "both"
  # Options: null, "pre", "post", "both"
  # Default: null

  min_impurity_decrease: 0.001
  # Minimum impurity decrease required for split (pre-pruning)
  # Type: float
  # Default: 0.0

  ccp_alpha: 0.01
  # Cost-complexity parameter for post-pruning
  # Type: float
  # Default: 0.0

  # --- Regularization ---
  max_leaf_nodes: null
  # Maximum number of leaf nodes
  # Type: int or null
  # Default: null

  min_weight_fraction_leaf: 0.0
  # Minimum weighted fraction of total samples in a leaf
  # Type: float
  # Default: 0.0

  # --- Performance ---
  use_cache: true
  # Cache sketch operations during training
  # Type: bool
  # Default: true

  cache_size_mb: 100
  # Maximum cache size in megabytes
  # Type: int
  # Default: 100

  # --- Other ---
  random_state: 42
  # Random seed for reproducibility
  # Type: int or null
  # Default: null

  verbose: 1
  # Verbosity level (0: silent, 1: progress, 2: debug)
  # Type: int
  # Default: 0

# =============================================================================
# Feature Mapping (for Inference)
# =============================================================================
feature_mapping:
  # Format: <feature_name>: {column_index, operator, threshold}

  "age>30":
    column_index: 0       # Column in raw data (0-indexed)
    operator: ">"         # Comparison operator
    threshold: 30         # Threshold value
    # Generates: lambda x: x > 30

  "income>50k":
    column_index: 1
    operator: ">"
    threshold: 50000
    # Generates: lambda x: x > 50000

  "has_diabetes":
    column_index: 2
    operator: "=="
    threshold: 1
    # Generates: lambda x: x == 1

  "age<=65":
    column_index: 0
    operator: "<="
    threshold: 65
    # Generates: lambda x: x <= 65

  "income_bracket=high":
    column_index: 1
    operator: ">="
    threshold: 100000
    # Generates: lambda x: x >= 100000

# =============================================================================
# Notes
# =============================================================================
# 1. Feature names in feature_mapping must match feature names in CSV
# 2. Supported operators: ">", ">=", "<", "<=", "==", "!="
# 3. Threshold types: int, float, or string (for equality checks)
# 4. Column indices are 0-based and refer to raw inference data
```

### Minimal Config Example

```yaml
targets:
  positive: "target_yes"
  negative: "target_no"

hyperparameters:
  criterion: "gini"
  max_depth: 10

feature_mapping:
  "age>30":
    column_index: 0
    operator: ">"
    threshold: 30
  "income>50k":
    column_index: 1
    operator: ">"
    threshold: 50000
```

### Config Validation Rules

The parser must validate:
- ✅ Required top-level keys: `targets`, `hyperparameters`, `feature_mapping`
- ✅ `targets` has `positive` and `negative` keys
- ✅ `hyperparameters` values are valid types and ranges
- ✅ `feature_mapping` entries have required keys: `column_index`, `operator`, `threshold`
- ✅ Operators are in allowed set: {">", ">=", "<", "<=", "==", "!="}
- ✅ Column indices are non-negative integers
- ✅ Criterion is in allowed set: {"gini", "entropy", "gain_ratio", "binomial", "binomial_chi"}

### JSON Alternative

For users who prefer JSON:

```json
{
  "targets": {
    "positive": "target_yes",
    "negative": "target_no"
  },
  "hyperparameters": {
    "criterion": "gini",
    "max_depth": 10,
    "min_samples_split": 20,
    "min_samples_leaf": 10
  },
  "feature_mapping": {
    "age>30": {
      "column_index": 0,
      "operator": ">",
      "threshold": 30
    },
    "income>50k": {
      "column_index": 1,
      "operator": ">",
      "threshold": 50000
    }
  }
}
```

Both YAML and JSON formats are supported by the ConfigParser.

---

## 3. Inference Data Format

### Raw Tabular Data

**Format**: NumPy array or Pandas DataFrame

#### NumPy Array
```python
import numpy as np

# Shape: (n_samples, n_features)
X = np.array([
    [35, 60000, 1],    # Sample 1: age=35, income=60000, diabetes=1
    [25, 45000, 0],    # Sample 2: age=25, income=45000, diabetes=0
    [55, 120000, 1],   # Sample 3: age=55, income=120000, diabetes=1
    [42, np.nan, 0],   # Sample 4: age=42, income=missing, diabetes=0
], dtype=np.float64)

# Column order must match column_index in feature_mapping
# Example:
#   Column 0: age
#   Column 1: income
#   Column 2: diabetes
```

#### Pandas DataFrame
```python
import pandas as pd

# Column names are informational only
# Actual mapping uses column_index from config
X = pd.DataFrame({
    'age': [35, 25, 55, 42],
    'income': [60000, 45000, 120000, np.nan],
    'diabetes': [1, 0, 1, 0]
})

# Internally converted to numpy array:
X_array = X.values  # Shape: (4, 3)
```

### Binary Feature Matrix (Internal)

After transformation using feature_mapping:

```python
# Shape: (n_samples, n_binary_features)
# dtype: bool (or float with NaN for missing)

X_binary = np.array([
    [True, True, True],      # age>30: True, income>50k: True, diabetes==1: True
    [False, False, False],   # age>30: False, income>50k: False, diabetes==1: False
    [True, True, True],      # age>30: True, income>50k: True, diabetes==1: True
    [True, np.nan, False],   # age>30: True, income>50k: NaN (missing), diabetes==1: False
])
```

### Missing Value Handling

**Supported missing value representations**:
- `np.nan` (NumPy NaN)
- `None` (Python None)
- `pd.NA` (Pandas NA)

**Treatment during inference**:
- Missing values preserved during transformation
- Handled according to `missing_value_strategy`:
  - `"majority"`: Follow majority path at each node
  - `"zero"`: Treat as False (condition not met)
  - `"error"`: Raise ValueError

### Prediction Output

```python
# predict() returns class labels
predictions = clf.predict(X)
# Output: np.array([1, 0, 1, 0])  # shape: (n_samples,), dtype: int

# predict_proba() returns class probabilities
probabilities = clf.predict_proba(X)
# Output: np.array([
#     [0.2, 0.8],  # Sample 1: P(class=0)=0.2, P(class=1)=0.8
#     [0.9, 0.1],  # Sample 2: P(class=0)=0.9, P(class=1)=0.1
#     [0.1, 0.9],  # Sample 3: P(class=0)=0.1, P(class=1)=0.9
#     [0.7, 0.3],  # Sample 4: P(class=0)=0.7, P(class=1)=0.3
# ])  # shape: (n_samples, 2), dtype: float64
```

---

## 4. Model Persistence Format

### Pickle Format (Primary)

```python
import pickle

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Load model
with open('model.pkl', 'rb') as f:
    clf_loaded = pickle.load(f)
```

**Saved Components**:
- All hyperparameters
- Trained tree structure (Tree object with all TreeNodes)
- Feature mapping
- Feature names
- sklearn attributes (classes_, n_classes_, etc.)

### JSON Format (Export for Interpretability)

For human-readable tree inspection:

```python
# Export tree structure to JSON
tree_json = clf.export_tree_json()

# Example output:
{
  "tree_metadata": {
    "n_nodes": 15,
    "n_leaves": 8,
    "max_depth": 4,
    "criterion": "gini"
  },
  "feature_names": ["age>30", "income>50k", "has_diabetes"],
  "tree_structure": {
    "type": "internal",
    "depth": 0,
    "n_samples": 1000,
    "feature_idx": 0,
    "feature_name": "age>30",
    "impurity": 0.5,
    "missing_direction": "right",
    "left": {
      "type": "leaf",
      "depth": 1,
      "n_samples": 400,
      "prediction": 0,
      "probabilities": [0.9, 0.1],
      "class_counts": [360, 40]
    },
    "right": {
      "type": "internal",
      "depth": 1,
      "n_samples": 600,
      "feature_idx": 1,
      "feature_name": "income>50k",
      "impurity": 0.48,
      "missing_direction": "left",
      "left": {...},
      "right": {...}
    }
  }
}
```

**JSON Export Use Cases**:
- Model interpretability and visualization
- Audit trails for regulated industries
- Integration with web-based decision tree visualizers
- Debugging tree structure

**Note**: JSON export does NOT include:
- Feature mapping lambda functions (not serializable)
- Sketch cache
- Original sketch data

To restore full functionality, use pickle format.

---

## 5. Examples

### Complete End-to-End Example

#### 1. CSV Sketch File (`sketches.csv`)

```csv
identifier,sketch
,AgMDAAAazJMCAAAAAACAPwAAAAAAAAAA...
target_yes,AgMDAAAazJMCAAAAAACAPwAAAAAAAAAA...
age>30,AgMDAAAazJMCAAAAAACAPwAAAAAAAAAA...
income>50k,AgMDAAAazJMCAAAAAACAPwAAAAAAAAAA...
has_diabetes,AgMDAAAazJMCAAAAAACAPwAAAAAAAAAA...
,AgMDAAAazJMCAAAAAACAPwAAAAAAAAAA...
target_no,AgMDAAAazJMCAAAAAACAPwAAAAAAAAAA...
age>30,AgMDAAAazJMCAAAAAACAPwAAAAAAAAAA...
income>50k,AgMDAAAazJMCAAAAAACAPwAAAAAAAAAA...
has_diabetes,AgMDAAAazJMCAAAAAACAPwAAAAAAAAAA...
```

#### 2. Config File (`config.yaml`)

```yaml
targets:
  positive: "target_yes"
  negative: "target_no"

hyperparameters:
  criterion: "gini"
  max_depth: 5
  min_samples_split: 10
  min_samples_leaf: 5
  use_cache: true
  verbose: 1

feature_mapping:
  "age>30":
    column_index: 0
    operator: ">"
    threshold: 30
  "income>50k":
    column_index: 1
    operator: ">"
    threshold: 50000
  "has_diabetes":
    column_index: 2
    operator: "=="
    threshold: 1
```

#### 3. Training

```python
from theta_sketch_tree import ThetaSketchDecisionTreeClassifier

# Create classifier
clf = ThetaSketchDecisionTreeClassifier()

# Train on sketches
clf.fit(
    csv_path='sketches.csv',
    config_path='config.yaml'
)

print(f"Tree built: {clf.tree_.n_nodes} nodes, {clf.tree_.n_leaves} leaves")
```

#### 4. Inference

```python
import numpy as np

# Raw test data
X_test = np.array([
    [35, 60000, 1],     # age=35, income=60k, diabetes=1
    [25, 45000, 0],     # age=25, income=45k, diabetes=0
    [55, 120000, 1],    # age=55, income=120k, diabetes=1
    [42, np.nan, 0],    # age=42, income=missing, diabetes=0
])

# Predictions
y_pred = clf.predict(X_test)
print(f"Predictions: {y_pred}")
# Output: [1, 0, 1, 0]

# Probabilities
y_proba = clf.predict_proba(X_test)
print(f"Probabilities:\n{y_proba}")
# Output:
# [[0.2, 0.8],
#  [0.9, 0.1],
#  [0.1, 0.9],
#  [0.7, 0.3]]
```

#### 5. Model Persistence

```python
# Save
clf.save_model('diabetes_model.pkl')

# Load
clf_loaded = ThetaSketchDecisionTreeClassifier.load_model('diabetes_model.pkl')

# Use loaded model
predictions = clf_loaded.predict(X_test)
```

---

### Medical Use Case Example

#### CSV Sketch File (`patient_outcomes.csv`)

Features for predicting hospital readmission:

```csv
identifier,sketch
,<total_positive_sketch>
readmitted_yes,<readmitted_yes_sketch>
age>65,<age_65_positive_sketch>
diabetes_diagnosis,<diabetes_positive_sketch>
emergency_admission,<emergency_positive_sketch>
length_of_stay>7,<los_7_positive_sketch>
num_medications>5,<med_5_positive_sketch>
,<total_negative_sketch>
readmitted_no,<readmitted_no_sketch>
age>65,<age_65_negative_sketch>
diabetes_diagnosis,<diabetes_negative_sketch>
emergency_admission,<emergency_negative_sketch>
length_of_stay>7,<los_7_negative_sketch>
num_medications>5,<med_5_negative_sketch>
```

#### Config File (`patient_config.yaml`)

```yaml
targets:
  positive: "readmitted_yes"
  negative: "readmitted_no"

hyperparameters:
  criterion: "binomial"      # Statistical significance testing
  min_pvalue: 0.01           # Strict significance threshold
  use_bonferroni: true       # Multiple testing correction
  max_depth: 8
  min_samples_split: 50      # Conservative splitting
  min_samples_leaf: 20
  class_weight: "balanced"   # Handle imbalanced readmission rates
  missing_value_strategy: "majority"
  pruning: "both"
  min_impurity_decrease: 0.01
  use_cache: true
  verbose: 1

feature_mapping:
  "age>65":
    column_index: 0
    operator: ">"
    threshold: 65
  "diabetes_diagnosis":
    column_index: 1
    operator: "=="
    threshold: 1
  "emergency_admission":
    column_index: 2
    operator: "=="
    threshold: 1
  "length_of_stay>7":
    column_index: 3
    operator: ">"
    threshold: 7
  "num_medications>5":
    column_index: 4
    operator: ">"
    threshold: 5
```

#### Inference Data

```python
# Patient records: [age, diabetes, emergency, los, num_meds]
X_patients = np.array([
    [72, 1, 1, 10, 8],    # High risk patient
    [45, 0, 0, 3, 2],     # Low risk patient
    [68, 1, 0, 5, 6],     # Medium risk patient
])

# Predict readmission risk
risk_scores = clf.predict_proba(X_patients)[:, 1]
print(f"Readmission risk scores: {risk_scores}")
# Output: [0.85, 0.12, 0.43]
```

---

## Summary

This specification defines:

✅ **CSV Sketch Format**: Complete specification for sketch serialization
✅ **Config File Format**: YAML/JSON configuration for targets, hyperparameters, and feature mapping
✅ **Inference Format**: NumPy/Pandas format for raw tabular data
✅ **Model Persistence**: Pickle for full model save/load, JSON for interpretability
✅ **Examples**: Complete end-to-end workflows for medical use cases

**Key Points**:
- Base64 encoding for sketch bytes (human-readable, CSV-safe)
- YAML config for easy hyperparameter tuning
- sklearn-compatible data formats (numpy/pandas)
- Pickle for production deployment
- JSON export for model interpretability

**Validation**:
- All formats include strict validation rules
- Parsers must validate structure, types, and ranges
- Clear error messages for invalid inputs
