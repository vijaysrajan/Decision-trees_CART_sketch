# Hyperparameter Tuning Guide
## Theta Sketch Decision Tree Classifier

---

## Table of Contents
1. [Quick Start: Recommended Configurations](#1-quick-start-recommended-configurations)
2. [Sketch Size (k) Selection](#2-sketch-size-k-selection)
3. [Tree Depth Selection](#3-tree-depth-selection)
4. [Split Criterion Selection](#4-split-criterion-selection)
5. [Class Imbalance Handling](#5-class-imbalance-handling)
6. [Pruning Parameters](#6-pruning-parameters)
7. [Performance Tuning](#7-performance-tuning)
8. [Complete Configuration Templates](#8-complete-configuration-templates)

---

## 1. Quick Start: Recommended Configurations

### Choose Your Use Case

| Use Case | Config Template | Key Settings |
|----------|----------------|--------------|
| **CTR Prediction (1% positive)** | [CTR Template](#ctr-prediction-1-positive-rate) | k=4096, depth=5, binomial |
| **Fraud Detection (<0.1%)** | [Fraud Template](#fraud-detection-01-positive-rate) | k=8192, depth=5, binomial |
| **Customer Churn (10-15%)** | [Churn Template](#customer-churn-10-15-positive-rate) | k=4096, depth=6, gini |
| **Medical Diagnosis (5%)** | [Medical Template](#medical-diagnosis-balanced-or-imbalanced) | k=8192, depth=4, binomial |
| **Stakeholder Analysis** | [Analysis Template](#stakeholder-analysis-any-use-case) | k=4096, depth=3, gini |
| **General Purpose** | [Default Template](#general-purpose-balanced-data) | k=4096, depth=5, entropy |

---

## 2. Sketch Size (k) Selection

### Understanding the Trade-off

**Sketch size (k)** controls the accuracy-storage-speed trade-off:

```
Larger k → Better accuracy, More storage, Slower
Smaller k → Worse accuracy, Less storage, Faster
```

### Error vs Sketch Size

From Apache DataSketches formula: **RSE = 1/√k**

| log₂(k) | k Value | Base RSE | Error @ Depth 3 | Error @ Depth 5 | Storage per Sketch |
|---------|---------|----------|-----------------|-----------------|-------------------|
| 10 | 1,024 | 3.13% | 12.3% | 31.3% | ~8 KB |
| 11 | 2,048 | 2.21% | 8.7% | 22.1% | ~16 KB |
| **12** | **4,096** | **1.56%** | **6.2%** | **15.6%** | **~32 KB** ← Default |
| 13 | 8,192 | 1.10% | 4.4% | 11.0% | ~64 KB |
| 14 | 16,384 | 0.78% | 3.1% | 7.8% | ~128 KB |
| 15 | 32,768 | 0.55% | 2.2% | 5.5% | ~256 KB |

**Note**: Errors calculated assuming F≈2.5 per intersection with feature-absent sketches.

### Decision Matrix for k Selection

#### By Dataset Size

| Records | Features | Recommended k | log₂(k) | Rationale |
|---------|----------|---------------|---------|-----------|
| <100M | <100 | 2,048 | 11 | Small data, base error is acceptable |
| 100M-1B | 100-1000 | **4,096** | **12** | **Sweet spot for most use cases** |
| 1B-10B | 1000-5000 | **4,096** | **12** | **Default (storage-efficient)** |
| 1B-10B | 1000-5000 (critical) | 8,192 | 13 | Better accuracy if storage allows |
| >10B | >5000 | 8,192-16,384 | 13-14 | Large scale, can afford storage |

#### By Accuracy Requirements

| Acceptable Error | Max Depth | Recommended k | log₂(k) | Use Case |
|------------------|-----------|---------------|---------|----------|
| >15% | 3-5 | 2,048 | 11 | Exploratory analysis |
| 10-15% | 5 | **4,096** | **12** | **Production (standard)** |
| 5-10% | 5 | 8,192 | 13 | High-accuracy production |
| <5% | 3-4 | 8,192-16,384 | 13-14 | Medical, financial, critical |

#### By Storage Constraints

For 5,000 features with Mode 2 (3-column CSV with feature-absent sketches):

| Storage Budget | Max k | log₂(k) | Total Storage | Trade-off |
|----------------|-------|---------|---------------|-----------|
| <200 MB | 2,048 | 11 | 160 MB | ⚠️ Low accuracy |
| 200-500 MB | **4,096** | **12** | **320 MB** | ✅ **Recommended** |
| 500 MB-1 GB | 8,192 | 13 | 640 MB | ✅ Better accuracy |
| 1-2 GB | 16,384 | 14 | 1.28 GB | ⚠️ Diminishing returns |
| >2 GB | 32,768 | 15 | 2.56 GB | ❌ Overkill |

### Recommendation by Use Case

```yaml
# CTR, Conversion (1-5% positive rate)
sketch_size_k: 4096  # Sweet spot for imbalanced data

# Fraud, Anomaly (<1% positive rate)
sketch_size_k: 8192  # Need better accuracy for rare events

# Churn, Medical (balanced or 5-20% positive)
sketch_size_k: 4096  # Standard accuracy sufficient

# Critical applications (medical, financial)
sketch_size_k: 8192  # Higher accuracy for high-stakes decisions

# Exploratory analysis (cost-sensitive)
sketch_size_k: 2048  # Lower accuracy acceptable
```

### ROI Analysis: Is Larger k Worth It?

**k=4096 → k=8192 (2× storage)**:
- Error reduction: 29% (from 1.56% to 1.10% base RSE)
- At depth 5: 15.6% → 11.0% (30% improvement)
- Storage increase: 2× (320 MB → 640 MB)
- Training time: +40%

**Verdict**: ✅ Worth it for critical applications, ⚠️ Overkill for most use cases

**k=8192 → k=16,384 (2× storage)**:
- Error reduction: 29% (from 1.10% to 0.78% base RSE)
- At depth 5: 11.0% → 7.8% (29% improvement)
- Storage increase: 2× (640 MB → 1.28 GB)

**Verdict**: ❌ Diminishing returns, not recommended

---

## 3. Tree Depth Selection

### The Critical Parameter for Sketch-Based Trees

**Tree depth directly affects error compounding**: Each level multiplies error by √F

### Error by Depth (k=4096, with feature-absent sketches)

| Max Depth | Typical Error | Split Quality | Interpretability | Recommendation |
|-----------|---------------|---------------|------------------|----------------|
| 1 | 1.56% | Excellent | Very Simple | ⚠️ Too simple (underfits) |
| 2 | 3.1% | Very Good | Simple | ⚠️ Simple (may underfit) |
| **3** | **6.2%** | **Good** | **Visualizable** | ✅ **Stakeholder analysis** |
| 4 | 9.3% | Fair | Complex | ⚠️ Borderline for visualization |
| **5** | **12.4%** | **Acceptable** | **Not visualizable** | ✅ **Production deployment** |
| 6 | 15.5% | Marginal | Very Complex | ⚠️ High error, use cautiously |
| 7 | 19.4% | Poor | Too Complex | ❌ Not recommended |
| 8 | 24.3% | Very Poor | Too Complex | ❌ Avoid |
| 10 | 38.7% | Unusable | Too Complex | ❌ Never use |

**Assumptions**: F≈2.5 per level (moderate feature correlation), k=4096

### Sweet Spot: Depth 5 for Production

**Why depth 5 is optimal**:
1. **Error is acceptable**: 12-15% for most ML tasks (CTR, churn, fraud)
2. **Balances bias-variance**: Not too shallow (underfit), not too deep (overfit)
3. **Computational efficiency**: Fast training and inference
4. **Generalization**: Deeper trees often overfit on sketch noise

**When to use depth 3**:
- Stakeholder-facing analysis
- Need interpretable trees
- Visualization required
- High-stakes decisions needing explainability

**When to use depth 6**:
- Large, clean datasets
- Complex patterns
- Can tolerate 15% error
- Have k=8192 (better accuracy)

### Decision Matrix for max_depth

#### By Use Case

| Use Case | Positive Rate | Recommended Depth | Rationale |
|----------|---------------|-------------------|-----------|
| **CTR Prediction** | 1-5% | **5** | Balances accuracy and overfitting |
| **Fraud Detection** | <0.1% | **5** | Need depth for rare patterns |
| **Conversion** | 2-10% | **5** | Standard depth |
| **Churn** | 10-20% | **5-6** | Can go deeper (more balanced) |
| **Medical Diagnosis** | Varies | **4** | Prioritize interpretability |
| **Stakeholder Analysis** | Any | **3** | Must be visualizable |
| **Exploratory** | Any | **3-4** | Fast iteration |

#### By Accuracy Requirements

| Target Accuracy | Max Depth | Expected Error (k=4096) | Notes |
|-----------------|-----------|------------------------|-------|
| >75% (acceptable) | 3-5 | 6-12% | Good for most applications |
| >80% (good) | 5-6 | 12-16% | Production standard |
| >85% (excellent) | 6-7 | 16-20% | Need k=8192 or ensemble |

#### By Data Characteristics

| Feature Correlation | Recommended Depth | Rationale |
|---------------------|-------------------|-----------|
| High (F≈1.5-2.0) | 5-6 | Lower error growth, can go deeper |
| Medium (F≈2.0-3.0) | **5** | **Standard case** |
| Low (F≈3.0-5.0) | 3-4 | High error growth, stay shallow |

### Depth Tuning Strategy

```python
# Step 1: Try depth=5 first (standard)
clf_d5 = train_model(max_depth=5)
acc_d5 = evaluate(clf_d5)  # e.g., 82%

# Step 2: If underfitting (accuracy <75%), try depth=6
if acc_d5 < 0.75:
    clf_d6 = train_model(max_depth=6)
    acc_d6 = evaluate(clf_d6)

    # Use depth=6 if it improves significantly
    if acc_d6 > acc_d5 + 0.03:  # >3% improvement
        best_depth = 6
    else:
        best_depth = 5  # Not worth the extra complexity

# Step 3: If depth=5 works well, try depth=3 for stakeholders
clf_d3 = train_model(max_depth=3)
acc_d3 = evaluate(clf_d3)

# Use depth=3 if accuracy loss is acceptable (<5%)
if acc_d3 > acc_d5 - 0.05:
    stakeholder_depth = 3
else:
    stakeholder_depth = 5  # Need full tree for accuracy
```

---

## 4. Split Criterion Selection

### Available Criteria

| Criterion | Best For | Advantages | Disadvantages |
|-----------|----------|------------|---------------|
| **Gini** | Balanced data | Fast, well-understood | Not optimized for imbalance |
| **Entropy** | Balanced data | Information-theoretic | Slightly slower than Gini |
| **Gain Ratio** | High-cardinality | Handles many-valued splits | More complex |
| **Binomial** | Imbalanced data | Statistical significance | Slower, requires more samples |
| **Chi-Square** | Imbalanced data | Multi-class friendly | Requires more samples |

### Recommendation by Use Case

```yaml
# CTR, Fraud, Anomaly (imbalanced: <5% positive)
criterion: "binomial"
min_pvalue: 0.001              # Strict threshold
use_bonferroni: true           # Multiple testing correction

# Churn, Conversion (mild imbalance: 5-20% positive)
criterion: "gini"              # Simpler, faster
class_weight: "balanced"       # Handle imbalance via weights

# Balanced data (40-60% positive)
criterion: "entropy"           # or "gini", both work well

# Stakeholder analysis (any imbalance)
criterion: "gini"              # Most familiar to business
```

### Statistical Criteria (Binomial, Chi-Square)

**When to use**:
- ✅ Imbalanced data (positive rate <10%)
- ✅ High-stakes decisions (medical, financial)
- ✅ Need statistical rigor
- ✅ Many features (reduce false discoveries)

**Hyperparameters**:
```yaml
criterion: "binomial"
min_pvalue: 0.001              # 99.9% confidence
use_bonferroni: true           # Adjust for multiple comparisons
```

**Trade-offs**:
- **Slower**: Statistical tests at each split
- **Conservative**: Fewer splits (simpler trees)
- **Robust**: Less likely to overfit on noise

---

## 5. Class Imbalance Handling

### The Three-Pillar Approach

For imbalanced datasets (CTR, fraud, etc.), use ALL three:

```yaml
hyperparameters:
  # Pillar 1: Class weights
  class_weight: "balanced"         # Weight minority class higher
  use_weighted_gini: true          # Apply weights in impurity calc

  # Pillar 2: Statistical criterion
  criterion: "binomial"            # Statistical significance testing
  min_pvalue: 0.001                # Strict threshold

  # Pillar 3: Conservative splitting
  min_samples_split: 1000          # Require evidence
  min_samples_leaf: 500            # Avoid tiny minority leaves
```

### Class Weight Selection

| Positive Rate | class_weight | Manual Weights (if needed) |
|---------------|--------------|----------------------------|
| 1% (CTR) | "balanced" | {0: 1.0, 1: 99.0} |
| 0.1% (Fraud) | "balanced" | {0: 1.0, 1: 999.0} |
| 10% (Churn) | "balanced" | {0: 1.0, 1: 9.0} |
| 50% (Balanced) | null | Not needed |

**"balanced"** auto-computes: `weight = n_samples / (n_classes * n_class_samples)`

### Sample Size Requirements for Imbalanced Data

| Positive Rate | min_samples_split | min_samples_leaf | Rationale |
|---------------|-------------------|------------------|-----------|
| <1% (Fraud) | 5000 | 2500 | Need 50+ positive samples per leaf |
| 1-5% (CTR) | 1000 | 500 | Need 10-25 positive samples per leaf |
| 5-10% (Conversion) | 500 | 250 | Need 12-25 positive samples per leaf |
| 10-20% (Churn) | 200 | 100 | Need 10-20 positive samples per leaf |
| >20% (Balanced) | 100 | 50 | Standard settings |

**Rule of Thumb**: `min_samples_leaf` should ensure ≥10 minority class samples per leaf.

---

## 6. Pruning Parameters

### Pre-Pruning (Early Stopping)

**Controls**: Stop splitting early if not worthwhile

```yaml
# Aggressive (for stakeholder analysis)
min_impurity_decrease: 0.01        # Require 1% impurity reduction
min_samples_split: 100             # Need 100+ samples to split
min_samples_leaf: 50               # Leaves must have 50+ samples
max_leaf_nodes: 15                 # Limit total leaves

# Moderate (for production)
min_impurity_decrease: 0.002       # Require 0.2% improvement
min_samples_split: 500
min_samples_leaf: 250

# Light (maximize accuracy)
min_impurity_decrease: 0.0         # No early stopping
min_samples_split: 10
min_samples_leaf: 5
```

### Post-Pruning (Cost-Complexity Pruning)

**Controls**: Remove branches after tree is built

```yaml
# Aggressive (for stakeholder trees)
ccp_alpha: 0.005                   # Strong pruning

# Moderate (for production)
ccp_alpha: 0.001                   # Light pruning

# None (maximize accuracy)
ccp_alpha: 0.0                     # No post-pruning
```

### Pruning Strategy by Use Case

| Use Case | Pruning | ccp_alpha | max_leaf_nodes | Goal |
|----------|---------|-----------|----------------|------|
| **Stakeholder Analysis** | Aggressive | 0.005 | 15 | Simplicity |
| **Production (balanced)** | Light | 0.001 | null | Accuracy |
| **Production (imbalanced)** | Moderate | 0.002 | null | Balance |
| **Exploratory** | Aggressive | 0.01 | 20 | Fast iteration |

---

## 7. Performance Tuning

### Caching

**Always enable for training**:

```yaml
use_cache: true
cache_size_mb: 500                 # Adjust based on features

# Cache size guide:
# - 100 features: 100 MB
# - 1000 features: 200 MB
# - 5000 features: 500 MB
```

**Expected speedup**: 2-5× faster training

### Feature Sampling

For high-dimensional data (>1000 features):

```yaml
max_features: "sqrt"               # Sample √n features per split

# Options:
# - "sqrt": Good for most cases (like Random Forest)
# - "log2": More aggressive sampling
# - 0.5: Sample 50% of features
# - null: Use all features (slower, may overfit)
```

**Benefits**:
- Faster training
- Reduces overfitting
- More robust to irrelevant features

### Parallel Training (Future)

```yaml
n_jobs: 4                          # Use 4 CPU cores (if implemented)
```

---

## 8. Complete Configuration Templates

### CTR Prediction (1% Positive Rate)

```yaml
# config_ctr.yaml
targets:
  positive: "clicked"
  negative: "not_clicked"

hyperparameters:
  # Sketch and tree structure
  sketch_size_k: 4096              # Balance accuracy/storage
  max_depth: 5                     # Sweet spot for CTR

  # Imbalanced data handling
  criterion: "binomial"            # Statistical rigor
  min_pvalue: 0.001
  use_bonferroni: true
  class_weight: "balanced"         # Critical!
  use_weighted_gini: true

  # Sample requirements
  min_samples_split: 1000
  min_samples_leaf: 500

  # Pruning
  pruning: "both"
  min_impurity_decrease: 0.002
  ccp_alpha: 0.001

  # Performance
  use_cache: true
  cache_size_mb: 500
  max_features: "sqrt"             # For many features

  random_state: 42
  verbose: 1

feature_mapping:
  # All features (100-5000)
  "mobile": 0
  "weekend": 1
  # ... all features
```

**Expected Results**:
- Accuracy: 80-85%
- ROC AUC: 0.75-0.80
- Error @ depth 5: ~12%
- Training time: 10-60 min (with cache)

---

### Fraud Detection (0.1% Positive Rate)

```yaml
# config_fraud.yaml
hyperparameters:
  # Higher accuracy for rare events
  sketch_size_k: 8192              # Better accuracy for rare fraud
  max_depth: 5

  # Very strict statistical criteria
  criterion: "binomial"
  min_pvalue: 0.0001               # 99.99% confidence
  use_bonferroni: true
  class_weight: "balanced"

  # Very conservative splitting
  min_samples_split: 5000          # Need strong evidence
  min_samples_leaf: 2500           # Ensure 2-3 fraud cases per leaf

  # Light pruning (preserve rare patterns)
  pruning: "pre"                   # Only pre-pruning
  min_impurity_decrease: 0.005

  use_cache: true
  cache_size_mb: 1000
  max_features: "sqrt"
```

**Expected Results**:
- Precision: 60-80% @ 80% recall
- ROC AUC: 0.80-0.85
- Error @ depth 5: ~9% (k=8192)

---

### Customer Churn (10-15% Positive Rate)

```yaml
# config_churn.yaml
hyperparameters:
  sketch_size_k: 4096
  max_depth: 6                     # Can go deeper (less imbalanced)

  # Standard criteria work well
  criterion: "entropy"             # or "gini"
  class_weight: "balanced"         # Still helpful

  # Moderate sample requirements
  min_samples_split: 500
  min_samples_leaf: 250

  # Moderate pruning
  pruning: "both"
  min_impurity_decrease: 0.003
  ccp_alpha: 0.002

  use_cache: true
  cache_size_mb: 200
```

**Expected Results**:
- Accuracy: 75-85%
- ROC AUC: 0.70-0.80
- Error @ depth 6: ~15%

---

### Medical Diagnosis (Balanced or Imbalanced)

```yaml
# config_medical.yaml
hyperparameters:
  # High accuracy requirement
  sketch_size_k: 8192              # Medical = high stakes
  max_depth: 4                     # Limit for interpretability

  # Statistical rigor
  criterion: "binomial"
  min_pvalue: 0.001
  use_bonferroni: true
  class_weight: "balanced"

  # Conservative (avoid false positives/negatives)
  min_samples_split: 200
  min_samples_leaf: 100

  # Light pruning (preserve patterns)
  pruning: "both"
  min_impurity_decrease: 0.005
  ccp_alpha: 0.001

  use_cache: true
```

**Expected Results**:
- Accuracy: 80-90%
- Error @ depth 4: ~9% (k=8192)
- Interpretable tree for clinicians

---

### Stakeholder Analysis (Any Use Case)

```yaml
# config_analysis.yaml
hyperparameters:
  sketch_size_k: 4096
  max_depth: 3                     # Must be visualizable!

  # Simple criterion
  criterion: "gini"                # Familiar to business
  class_weight: "balanced"

  # Conservative
  min_samples_split: 100
  min_samples_leaf: 50

  # Aggressive pruning for simplicity
  pruning: "both"
  min_impurity_decrease: 0.01
  ccp_alpha: 0.005
  max_leaf_nodes: 15               # Limit complexity

  use_cache: true
  cache_size_mb: 100
```

**Expected Results**:
- Accuracy: 75-80%
- Error @ depth 3: ~6%
- Tree fits on one page

---

### General Purpose (Balanced Data)

```yaml
# config_default.yaml
hyperparameters:
  sketch_size_k: 4096              # Standard
  max_depth: 5                     # Standard
  criterion: "entropy"             # or "gini"

  min_samples_split: 100
  min_samples_leaf: 50

  pruning: "both"
  min_impurity_decrease: 0.005
  ccp_alpha: 0.002

  use_cache: true
  cache_size_mb: 200
  random_state: 42
```

---

## Quick Reference Cheat Sheet

### By Priority

**1. Most Important Hyperparameters** (tune these first):
- `sketch_size_k`: 4096 (default) or 8192 (high-accuracy)
- `max_depth`: 5 (production) or 3 (analysis)
- `criterion`: "binomial" (imbalanced) or "gini" (balanced)
- `class_weight`: "balanced" (always for imbalanced data)

**2. Secondary Hyperparameters** (tune if needed):
- `min_samples_split`: 100-1000 (higher for imbalanced)
- `min_samples_leaf`: 50-500 (higher for imbalanced)
- `ccp_alpha`: 0.001-0.005 (higher for simpler trees)

**3. Performance Hyperparameters**:
- `use_cache`: true (always)
- `cache_size_mb`: 100-500 (more for more features)
- `max_features`: "sqrt" (for >1000 features)

### By Use Case (One-Liner)

```yaml
# CTR (1% positive)
{k: 4096, depth: 5, criterion: "binomial", class_weight: "balanced", min_samples_split: 1000}

# Fraud (<0.1% positive)
{k: 8192, depth: 5, criterion: "binomial", class_weight: "balanced", min_samples_split: 5000}

# Churn (10% positive)
{k: 4096, depth: 6, criterion: "gini", class_weight: "balanced", min_samples_split: 500}

# Stakeholder Analysis
{k: 4096, depth: 3, criterion: "gini", max_leaf_nodes: 15, ccp_alpha: 0.005}
```

---

## Tuning Process Workflow

### Step 1: Start with Template

Choose template based on use case → [CTR, Fraud, Churn, Medical, Analysis]

### Step 2: Tune Key Parameters

1. **Fix k=4096** initially (change later if needed)
2. **Tune max_depth**: Try 3, 5, 6 → pick best on validation set
3. **Tune criterion**: Try "gini", "binomial" → compare AUC

### Step 3: Tune Sample Requirements

For imbalanced data:
1. Start with `min_samples_split = 1000 / positive_rate`
2. Set `min_samples_leaf = min_samples_split / 2`
3. Adjust if needed

### Step 4: Tune Pruning

1. Train without pruning (`ccp_alpha=0`)
2. Evaluate accuracy
3. If overfitting, increase `ccp_alpha` gradually (0.001 → 0.002 → 0.005)
4. Stop when validation accuracy stabilizes

### Step 5: Evaluate on Test Set

- Accuracy
- ROC AUC
- Precision/Recall at operating point
- Online A/B test (if production)

### Step 6: Consider Increasing k if Needed

If accuracy is insufficient:
1. Try k=8192 (2× storage)
2. Re-evaluate
3. If improvement <3%, revert to k=4096

---

**Key Takeaway**: Start with recommended templates, tune max_depth and criterion, then fine-tune sample requirements and pruning. Only increase k if accuracy is still insufficient.
