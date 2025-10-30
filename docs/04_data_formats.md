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

### Two Input Modes

The classifier supports two input modes for sketch data:

**Mode 1: Single CSV with Intersections**
- All sketches (features and targets) in one CSV file
- Loader performs intersection operations (target ∩ feature)
- Simpler setup but sketch operations compound error

**Mode 2: Dual CSV Pre-Intersected (RECOMMENDED)**
- Two separate CSV files: `target_yes.csv` and `target_no.csv`
- Sketches pre-computed in big data (already intersected)
- Better accuracy (no sketch operation error compounding)
- Faster loading (no runtime intersections)

### Format Specification

```
<identifier>, <sketch_bytes>
```

**Columns**:
- **Column 1: Identifier** (string)
  - `total`: Population sketch for this class
  - `<feature_name>`: Feature condition sketch (e.g., "age>30", "income>50k")
  - **Mode 1 only**: `<target_name>` (e.g., "target_yes", "target_no")

- **Column 2: Sketch Bytes** (base64-encoded or hex-encoded string)
  - Serialized Apache DataSketches theta sketch
  - Encoding: base64 (default) or hex

### Mode 1: Single CSV Example

```csv
identifier,sketch
total,<base64_all_records>
target_yes,<base64_positive_class>
target_no,<base64_negative_class>
age>30,<base64_anyone_age>30>
income>50k,<base64_anyone_income>50k>
has_diabetes,<base64_anyone_diabetes>
```

**Loading**: Loader intersects `target_yes ∩ age>30`, `target_no ∩ age>30`, etc.

### Mode 2: Dual CSV Example (RECOMMENDED)

**target_yes.csv** (positive class, pre-intersected):
```csv
identifier,sketch
total,<base64_positive_class_total>
age>30,<base64_positive_AND_age>30>
income>50k,<base64_positive_AND_income>50k>
has_diabetes,<base64_positive_AND_diabetes>
```

**target_no.csv** (negative class, pre-intersected):
```csv
identifier,sketch
total,<base64_negative_class_total>
age>30,<base64_negative_AND_age>30>
income>50k,<base64_negative_AND_income>50k>
has_diabetes,<base64_negative_AND_diabetes>
```

**Loading**: Sketches already intersected, no operations needed.

**Note**: Feature names must match between both files.

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
  # Simple format: <feature_name>: <column_index>
  # Maps feature names (used in tree splits) to column indices in binary inference data

  "age>30": 0           # Column 0 contains binary age>30 values (0/1)
  "income>50k": 1       # Column 1 contains binary income>50k values (0/1)
  "has_diabetes": 2     # Column 2 contains binary diabetes indicator (0/1)
  "age<=65": 3          # Column 3 contains binary age<=65 values (0/1)
  "city=NY": 4          # Column 4 contains binary city=NY indicator (0/1)

# =============================================================================
# Notes
# =============================================================================
# 1. Feature names in feature_mapping must match feature names in CSV sketches
# 2. All inference data must be PRE-TRANSFORMED to binary (0/1) values
# 3. Feature transformations (age > 30, city == 'NY', etc.) happen BEFORE inference
# 4. Column indices are 0-based and refer to binary inference data columns
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
  "age>30": 0        # Simple column index mapping
  "income>50k": 1
  "city=NY": 2
```

### Config Validation Rules

The parser must validate:
- ✅ Required top-level keys: `targets`, `hyperparameters`, `feature_mapping`
- ✅ `targets` has `positive` and `negative` keys
- ✅ `hyperparameters` values are valid types and ranges
- ✅ `feature_mapping` values are non-negative integers (column indices)
- ✅ Feature names are strings
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
    "age>30": 0,
    "income>50k": 1,
    "city=NY": 2
  }
}
```

Both YAML and JSON formats are supported by the ConfigParser.

---

## 3. Inference Data Format

### Binary Tabular Data

**Format**: NumPy array or Pandas DataFrame with binary (0/1) features

**IMPORTANT**: All features must be PRE-TRANSFORMED to binary values before inference.

#### NumPy Array
```python
import numpy as np

# Shape: (n_samples, n_features)
# All values are binary: 0 or 1 (False or True)
X = np.array([
    [1, 1, 0, 1],    # Sample 1: age>30=Yes, income>50k=Yes, city=NY=No, diabetes=Yes
    [0, 0, 1, 0],    # Sample 2: age>30=No, income>50k=No, city=NY=Yes, diabetes=No
    [1, 1, 0, 1],    # Sample 3: age>30=Yes, income>50k=Yes, city=NY=No, diabetes=Yes
    [1, np.nan, 0, 0], # Sample 4: age>30=Yes, income>50k=MISSING, city=NY=No, diabetes=No
], dtype=np.float64)

# Column order must match feature_mapping column indices
# Example feature_mapping: {"age>30": 0, "income>50k": 1, "city=NY": 2, "diabetes": 3}
#   Column 0: age>30 (binary: 0=No, 1=Yes)
#   Column 1: income>50k (binary: 0=No, 1=Yes)
#   Column 2: city=NY (binary: 0=No, 1=Yes)
#   Column 3: diabetes (binary: 0=No, 1=Yes)
```

#### Pandas DataFrame
```python
import pandas as pd

# Columns contain pre-computed binary features
# Column order must match feature_mapping
X = pd.DataFrame({
    'age_over_30': [1, 0, 1, 1],        # Binary: age > 30
    'income_over_50k': [1, 0, 1, np.nan],  # Binary: income > 50k (with missing)
    'city_NY': [0, 1, 0, 0],            # Binary: city == 'NY'
    'diabetes': [1, 0, 1, 0]            # Binary: has diabetes
})

# Internally converted to numpy array:
X_array = X.values  # Shape: (4, 4)
```

**Note**: DataFrame column names are informational only. The classifier uses positional indices from feature_mapping.

### Supported Data Types

```python
# All are valid binary representations:
X_bool = np.array([[True, False], [False, True]])  # Boolean
X_int = np.array([[1, 0], [0, 1]])                  # Integer 0/1
X_float = np.array([[1.0, 0.0], [0.0, 1.0]])        # Float 0.0/1.0
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
  "age>30": 0        # Column 0 contains binary age>30 values
  "income>50k": 1    # Column 1 contains binary income>50k values
  "has_diabetes": 2  # Column 2 contains binary diabetes values
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

# Binary test data (features already transformed to 0/1)
X_test = np.array([
    [1, 1, 1],       # age>30=1, income>50k=1, diabetes=1
    [0, 0, 0],       # age>30=0, income>50k=0, diabetes=0
    [1, 1, 1],       # age>30=1, income>50k=1, diabetes=1
    [1, np.nan, 0],  # age>30=1, income>50k=MISSING, diabetes=0
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

**Note**: Data must be pre-transformed to binary before calling predict(). Feature transformations (age > 30, income > 50000, etc.) happen externally.

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
  "age>65": 0               # Binary: patient age > 65
  "diabetes_diagnosis": 1   # Binary: has diabetes diagnosis
  "emergency_admission": 2  # Binary: admitted via emergency
  "length_of_stay>7": 3     # Binary: length of stay > 7 days
  "num_medications>5": 4    # Binary: taking > 5 medications
```

#### Inference Data

```python
# Patient records (binary features: age>65, diabetes, emergency, los>7, meds>5)
X_patients = np.array([
    [1, 1, 1, 1, 1],    # High risk: all risk factors present
    [0, 0, 0, 0, 0],    # Low risk: no risk factors
    [1, 1, 0, 0, 1],    # Medium risk: some risk factors
])

# Predict readmission risk
risk_scores = clf.predict_proba(X_patients)[:, 1]
print(f"Readmission risk scores: {risk_scores}")
# Output: [0.85, 0.12, 0.43]
```

**Note**: All features must be pre-computed as binary before inference (e.g., age > 65 becomes 1 or 0).

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
