# Data Formats Guide

## Overview

This guide describes the data formats required for the Theta Sketch Decision Tree classifier, including CSV sketch formats, feature mapping files, and validation procedures.

## Sketch CSV Format

### Required Structure

All sketch CSV files must follow this exact 3-column format:

```csv
feature_name,present_sketch,absent_sketch
age>30,<base64_encoded_sketch_1>,<base64_encoded_sketch_2>
income>50k,<base64_encoded_sketch_3>,<base64_encoded_sketch_4>
education=grad,<base64_encoded_sketch_5>,<base64_encoded_sketch_6>
```

### Column Specifications

- **`feature_name`**: String identifier for the binary feature
- **`present_sketch`**: Base64-encoded theta sketch for samples where feature=1
- **`absent_sketch`**: Base64-encoded theta sketch for samples where feature=0

### Training Modes

#### Dual-Class Mode (Recommended for Balanced Datasets)
```
positive_class.csv    # Sketches from positive class samples
negative_class.csv    # Sketches from negative class samples
```

#### One-vs-All Mode (Recommended for Imbalanced Datasets)
```
positive_class.csv    # Sketches from positive class samples
total_population.csv  # Sketches from ENTIRE dataset (all classes)
```

## Feature Mapping Format

### JSON Structure
```json
{
  "age>30": 0,
  "income>50k": 1,
  "education=grad": 2,
  "marital_status=married": 3
}
```

**Requirements:**
- Maps feature names to column indices in prediction data
- Must include all features present in sketch CSV files
- Indices must be contiguous (0, 1, 2, ..., n-1)

## Configuration Files

### YAML Configuration
```yaml
model_parameters:
  criterion: "gini"
  max_depth: 10
  min_samples_split: 2
  pruning: "cost_complexity"

data_sources:
  positive_csv: "positive_class.csv"
  negative_csv: "negative_class.csv"
  feature_mapping_json: "feature_mapping.json"

training_parameters:
  lg_k: 12
  verbose: 1
```

## Validation and Quality Checks

### Sketch Validation
```python
import base64
from datasketches import theta_sketch

def validate_sketch_data(csv_path):
    """Validate sketch CSV format and integrity."""
    df = pd.read_csv(csv_path)

    # Check column format
    expected_cols = ['feature_name', 'present_sketch', 'absent_sketch']
    assert list(df.columns) == expected_cols

    # Check sketch deserialization
    for idx, row in df.iterrows():
        try:
            present_bytes = base64.b64decode(row['present_sketch'])
            absent_bytes = base64.b64decode(row['absent_sketch'])

            present_sketch = theta_sketch.deserialize(present_bytes)
            absent_sketch = theta_sketch.deserialize(absent_bytes)

            print(f"✅ {row['feature_name']}: {present_sketch.get_estimate():.0f} + {absent_sketch.get_estimate():.0f}")
        except Exception as e:
            print(f"❌ {row['feature_name']}: {e}")
```

## Common Format Issues

### Issue 1: Incorrect Column Names
```
❌ Bad: feature,present,absent
✅ Good: feature_name,present_sketch,absent_sketch
```

### Issue 2: Invalid Base64 Encoding
```
❌ Bad: Sketch contains invalid characters
✅ Good: Proper base64 string: "rO0ABX..."
```

### Issue 3: lg_k Parameter Mismatch
```
❌ Bad: Mixing sketches with different lg_k values
✅ Good: All sketches created with same lg_k=12
```

---

## Next Steps

- **User Guide**: See [User Guide](02-user-guide.md) for usage examples
- **Troubleshooting**: Check [Troubleshooting Guide](08-troubleshooting.md) for format issues
- **Performance**: Review [Performance Guide](06-performance.md) for optimization
- **API Reference**: See [API Reference](05-api-reference.md) for data loading methods