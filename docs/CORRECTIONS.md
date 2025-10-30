# Design Corrections Based on User Feedback
## Simplified Architecture - Binary Features Only

---

## Key Changes

### 1. **Dual CSV Input Mode Added** ✅
**Reason**: Pre-computing class-conditional sketches in big data avoids sketch operation error compounding.

**Mode 1 - Single CSV** (original):
```python
# All sketches in one file, intersections done during loading
clf.fit(csv_path='features.csv', config_path='config.yaml')
# Loader computes: target_yes ∩ age>30, target_no ∩ age>30, etc.
# Error compounds with each intersection operation
```

**Mode 2 - Dual CSV** (RECOMMENDED):
```python
# Pre-intersected sketches from big data
clf.fit(
    positive_csv='target_yes.csv',  # Contains target_yes AND age>30, etc.
    negative_csv='target_no.csv',   # Contains target_no AND age>30, etc.
    config_path='config.yaml'
)
# No intersection operations needed - better accuracy!
```

**Benefits of Mode 2**:
- ✅ Better accuracy (no sketch operation error compounding)
- ✅ Faster loading (no runtime intersections)
- ✅ Sketches built directly from raw data in big data system
- ✅ Cleaner separation: heavy lifting in big data, lightweight training here

---

### 2. **Removed FeatureTransformer** ❌
**Reason**: Inference data is already binary. No transformation needed.

**Old Design** (WRONG):
```python
# Complex transformation from continuous to binary
feature_mapping = {
    "age>30": (0, lambda x: x > 30),  # Transform continuous age
}
X_raw = np.array([[35, 60000, 1]])  # Continuous values
X_binary = transformer.transform(X_raw)  # Transform to binary
```

**New Design** (CORRECT):
```python
# Simple column index mapping
feature_mapping = {
    "age>30": 0,           # Column 0 is already binary (age>30?)
    "income>50k": 1,       # Column 1 is already binary (income>50k?)
}
X_binary = np.array([[1, 1, 1]])  # Already binary!
```

---

### 2. **Simplified Config File Format**

**Old Config** (WRONG):
```yaml
feature_mapping:
  "age>30":
    column_index: 0
    operator: ">"      # NOT NEEDED
    threshold: 30      # NOT NEEDED
```

**New Config** (CORRECT):
```yaml
feature_mapping:
  "age>30": 0              # Just column index
  "city=NewYork": 1
  "gender=M": 2
  "income>50k": 3
```

---

### 3. **Simplified Classifier Attributes**

**Removed Attributes**:
```python
# OLD (removed)
_feature_transformer: FeatureTransformer  # NOT NEEDED
_feature_mapping: Dict[str, Tuple[int, Callable]]  # Too complex
```

**New Attributes**:
```python
# NEW (simplified)
_feature_mapping: Dict[str, int]  # Just feature_name -> column_index
```

---

### 4. **Inference Data Format**

**All features must be BINARY** (0/1, True/False):

```python
# CORRECT: Binary features
X_test = np.array([
    [1, 0, 1, 0],  # age>30=Yes, city=NY=No, gender=M=Yes, income>50k=No
    [0, 1, 0, 1],  # age>30=No, city=NY=Yes, gender=M=No, income>50k=Yes
    [1, 1, 1, 1],  # All features = Yes
])

# Column 0: age>30 (binary: 0=No, 1=Yes)
# Column 1: city=NewYork (binary: 0=No, 1=Yes)
# Column 2: gender=M (binary: 0=No, 1=Yes)
# Column 3: income>50k (binary: 0=No, 1=Yes)
```

**Missing Values**:
```python
# Supported missing representations
X_with_missing = np.array([
    [1, 0, np.nan, 1],     # Missing gender
    [0, None, 1, 0],       # Missing city (Python None)
    [1, pd.NA, 0, 1],      # Missing city (Pandas NA)
])

# Also support empty string in DataFrames
df = pd.DataFrame({
    'age>30': [1, 0, 1],
    'city=NY': ['', 1, 0],  # Empty string = missing
})
```

---

### 5. **Simplified predict() Method**

**Old (WRONG - too complex)**:
```python
def predict(self, X):
    check_is_fitted(self)
    X = check_array(X)

    # Transform raw to binary (NOT NEEDED!)
    X_binary = self._feature_transformer.transform(X)

    # Traverse tree
    predictions = self._tree_traverser.predict(X_binary)
    return predictions
```

**New (CORRECT - simplified)**:
```python
def predict(self, X):
    """
    Predict on binary feature data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Binary feature matrix (0/1 values, with NaN for missing)
    """
    check_is_fitted(self)
    X = check_array(X, force_all_finite='allow-nan')  # Allow NaN

    # X is already binary - just traverse tree
    predictions = self._tree_traverser.predict(X, self.tree_.root)
    return predictions
```

---

### 6. **Simplified TreeTraverser**

**No feature mapping needed during traversal**:

```python
def _traverse_to_leaf(self, sample, node):
    """
    Traverse tree for single binary sample.

    Parameters
    ----------
    sample : array of shape (n_features,)
        Binary feature values (0/1, with NaN for missing)
    node : TreeNode
        Current node
    """
    if node.is_leaf:
        return node

    # Get feature value (already binary!)
    feature_value = sample[node.feature_idx]

    # Handle missing
    if is_missing(feature_value):  # NaN, None, "", etc.
        if self.missing_value_strategy == 'error':
            raise ValueError(f"Missing value at feature {node.feature_name}")
        elif self.missing_value_strategy == 'zero':
            feature_value = 0  # Treat as False
        elif self.missing_value_strategy == 'majority':
            # Follow majority direction
            if node.missing_direction == 'left':
                return self._traverse_to_leaf(sample, node.left)
            else:
                return self._traverse_to_leaf(sample, node.right)

    # Binary split: 0 goes left, 1 goes right
    if feature_value == 0 or feature_value == False:
        return self._traverse_to_leaf(sample, node.left)
    else:  # feature_value == 1 or True
        return self._traverse_to_leaf(sample, node.right)
```

---

### 7. **Missing Value Detection**

```python
def is_missing(value):
    """
    Check if value is missing.

    Handles: NaN, None, pd.NA, empty string
    """
    if value is None:
        return True
    if isinstance(value, str) and value == "":
        return True
    if pd.isna(value):  # Handles np.nan and pd.NA
        return True
    return False
```

---

### 8. **Updated Test Fixtures**

**Old (WRONG - continuous data)**:
```python
@pytest.fixture
def sample_raw_data():
    return np.array([
        [35, 60000],    # Continuous age and income - WRONG!
        [25, 45000],
    ])
```

**New (CORRECT - binary data)**:
```python
@pytest.fixture
def sample_binary_data():
    """
    Binary feature data for inference testing.

    Features:
      - Column 0: age>30 (0/1)
      - Column 1: city=NewYork (0/1)
      - Column 2: gender=M (0/1)
      - Column 3: income>50k (0/1)
    """
    return np.array([
        [1, 0, 1, 1],      # age>30, not NY, male, high income
        [0, 1, 0, 0],      # age≤30, NY, female, low income
        [1, 1, 1, 1],      # All features = 1
        [1, 0, np.nan, 1], # Missing gender
    ])

@pytest.fixture
def sample_binary_dataframe():
    """Binary features as DataFrame (tests empty string handling)."""
    return pd.DataFrame({
        'age>30': [1, 0, 1, 1],
        'city=NY': [0, 1, '', 0],      # Empty string = missing
        'gender=M': [1, 0, 1, None],   # None = missing
        'income>50k': [1, 0, 1, pd.NA] # pd.NA = missing
    })
```

---

### 9. **Updated Config Parser**

```python
class ConfigParser:
    def parse_feature_mapping(self, feature_mapping_config):
        """
        Parse simplified feature mapping.

        Input:
        ------
        {
            "age>30": 0,
            "city=NY": 1,
            "gender=M": 2
        }

        Returns:
        --------
        Dict[str, int]: feature_name -> column_index
        """
        # Validate all values are integers
        for feature_name, col_idx in feature_mapping_config.items():
            if not isinstance(col_idx, int):
                raise ValueError(
                    f"Feature mapping value must be int column index, "
                    f"got {type(col_idx)} for '{feature_name}'"
                )
            if col_idx < 0:
                raise ValueError(
                    f"Column index must be >= 0, got {col_idx} for '{feature_name}'"
                )

        return feature_mapping_config
```

---

### 10. **Updated fit() Signature**

**Simplified - no feature_mapping needed as parameter**:

```python
def fit(self, csv_path, config_path):
    """
    Build decision tree from theta sketches.

    Parameters
    ----------
    csv_path : str
        Path to CSV file with theta sketches
    config_path : str
        Path to YAML config file with:
        - targets: positive/negative class names
        - hyperparameters: tree parameters
        - feature_mapping: feature_name -> column_index

    Returns
    -------
    self : ThetaSketchDecisionTreeClassifier
    """
    # Load config
    config = ConfigParser().load(config_path)

    # Extract feature mapping (just column indices)
    self._feature_mapping = config['feature_mapping']  # Dict[str, int]

    # Load sketches
    sketches_pos, sketches_neg = SketchLoader().load(
        csv_path,
        target_positive=config['targets']['positive'],
        target_negative=config['targets']['negative']
    )

    # Build tree
    root = TreeBuilder(...).build_tree(
        sketch_dict_pos=sketches_pos,
        sketch_dict_neg=sketches_neg,
        feature_names=list(self._feature_mapping.keys()),
        depth=0
    )

    # Set sklearn attributes
    self.classes_ = np.array([0, 1])  # Binary classification
    self.n_classes_ = 2
    self.n_features_in_ = len(self._feature_mapping)
    self.feature_names_in_ = np.array(list(self._feature_mapping.keys()))
    self.tree_ = Tree(root)

    return self
```

---

## Summary of Removals

❌ **Removed Classes**:
- `FeatureTransformer` (not needed)

❌ **Removed Methods**:
- `_create_comparison_lambda()` (not needed)
- `ConfigParser.parse_feature_mapping()` now much simpler

❌ **Removed Config Fields**:
- `operator` (not needed)
- `threshold` (not needed)

---

## Summary of Simplifications

✅ **Simplified**:
1. `feature_mapping`: Now just `Dict[str, int]` (feature_name -> column_index)
2. `config.yaml`: Just map feature names to column indices
3. `predict()`: No transformation, direct tree traversal
4. Test fixtures: Use binary data from the start
5. Missing values: Handle `""`, `None`, `np.nan`, `pd.NA`

---

## Updated Module Structure

**Modules NO LONGER NEEDED**:
- ~~`feature_transformer.py`~~ (removed)

**Simplified Modules**:
- `config_parser.py`: Much simpler (no lambda generation)
- `tree_traverser.py`: Simpler (no transformation)
- `classifier.py`: Simpler (fewer internal components)

**New Estimated Lines of Code**: ~3,000-3,500 (down from ~4,000-5,000)

---

## Updated Example

```python
# ========== Training ==========
clf = ThetaSketchDecisionTreeClassifier(
    criterion='gini',
    max_depth=10,
    verbose=1
)

# Config file (simplified):
# feature_mapping:
#   "age>30": 0
#   "city=NY": 1
#   "gender=M": 2

clf.fit('sketches.csv', 'config.yaml')

# ========== Inference (Binary Data) ==========
X_test = np.array([
    [1, 0, 1],  # age>30=Yes, city=NY=No, gender=M=Yes
    [0, 1, 0],  # age>30=No, city=NY=Yes, gender=M=No
])

predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
```

---

## Impact on Roadmap

**Time Savings**: ~1 week (Week 4 now much simpler)

**Week 4 Updated** (was: Inference & Missing Values):
- ~~Day 1-2: Feature Transformer~~ (REMOVED)
- Day 1-2: Tree Traverser (simpler now)
- Day 3: Missing Value Handler (add empty string support)
- Day 4-7: Main Classifier (simpler integration)

**New Timeline**: 5 weeks instead of 6

---

## Questions for You

1. ✅ **Confirm**: All inference data will be pre-computed binary features?
2. ✅ **Confirm**: Feature names in CSV match feature names in inference data schema?
3. ✅ **Confirm**: Missing values can be: `np.nan`, `None`, `pd.NA`, `""` (empty string)?
4. ❓ **Question**: Should we support feature names with special characters? (e.g., `"age>30"`, `"city=New York"`)
5. ❓ **Question**: For tree nodes, should `feature_idx` reference the column index directly, or map through `feature_mapping`?

---

This simplified design is **much cleaner** and **easier to implement**. Thank you for catching these issues! 🎉
