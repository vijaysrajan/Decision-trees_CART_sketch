# Deployment Strategies: Analysis vs Production
## Theta Sketch Decision Tree Classifier

---

## Table of Contents
1. [Two Distinct Use Cases](#1-two-distinct-use-cases)
2. [Strategy 1: Stakeholder Analysis & Business Insights](#2-strategy-1-stakeholder-analysis--business-insights)
3. [Strategy 2: Production Deployment & Runtime Inference](#3-strategy-2-production-deployment--runtime-inference)
4. [Hybrid Strategy: Best of Both Worlds](#4-hybrid-strategy-best-of-both-worlds)
5. [Migration Path: Analysis → Deployment](#5-migration-path-analysis--deployment)
6. [Summary Comparison](#6-summary-comparison)

---

## 1. Two Distinct Use Cases

The Theta Sketch Decision Tree Classifier serves **two fundamentally different purposes** with different optimization goals:

### Use Case A: Stakeholder Analysis (Business Intelligence)

**Primary Goal**: **Interpretability** - Help stakeholders understand data patterns

**Audience**:
- Business stakeholders (marketing, product, operations)
- Data analysts
- Domain experts (medical, financial)
- Executives making strategic decisions

**Key Requirements**:
- ✅ Simple, visualizable tree structure
- ✅ Clear, actionable decision rules
- ✅ High accuracy for trust and credibility
- ✅ Export to human-readable formats (JSON, diagrams)
- ❌ Inference speed not critical (batch analysis)
- ❌ Model size not critical (one-time analysis)

**Example Questions to Answer**:
- "What factors most influence customer churn?"
- "Which ad combinations drive highest CTR?"
- "What patient characteristics predict readmission risk?"
- "How do different demographics respond to our product?"

**Deliverables**:
- Visual tree diagrams
- Decision rule explanations
- Feature importance rankings
- Segment analysis reports

---

### Use Case B: Production Deployment (Real-Time Inference)

**Primary Goal**: **Predictive Performance** - Maximize accuracy for real-time decisions

**Audience**:
- ML engineers
- Production systems
- Real-time recommendation engines
- Automated decision systems

**Key Requirements**:
- ✅ Best possible prediction accuracy
- ✅ Fast inference (<1ms per sample)
- ✅ Low memory footprint
- ✅ Robust to missing values
- ✅ Stable, reproducible predictions
- ❌ Interpretability less important (black box acceptable)
- ❌ Tree complexity less constrained

**Example Use Cases**:
- Real-time ad serving (CTR prediction)
- Fraud detection
- Recommendation systems
- Risk scoring APIs
- Healthcare decision support systems

**Deliverables**:
- Pickled model for fast loading
- REST API for predictions
- Performance benchmarks
- A/B test results

---

## 2. Strategy 1: Stakeholder Analysis & Business Insights

### Recommended Configuration

```yaml
# config_analysis.yaml
targets:
  positive: "target_yes"
  negative: "target_no"

hyperparameters:
  # OPTIMIZE FOR INTERPRETABILITY

  # Simple criterion (easier to explain)
  criterion: "gini"              # or "entropy" (both well-understood)

  # SHALLOW TREE for simplicity
  max_depth: 3                   # 3-5 levels maximum (visualizable)
  min_samples_split: 100         # Conservative (avoid overfitting)
  min_samples_leaf: 50

  # Strong pruning for simplicity
  pruning: "both"
  min_impurity_decrease: 0.01    # Require meaningful splits
  ccp_alpha: 0.005               # Aggressive post-pruning

  # Limit tree size
  max_leaf_nodes: 15             # Keep tree small and readable

  # Class weights for imbalanced data
  class_weight: "balanced"

  # Performance (less critical)
  use_cache: true
  cache_size_mb: 100

  random_state: 42               # Reproducibility critical for reports
  verbose: 1

feature_mapping:
  # Use descriptive names that stakeholders understand
  "mobile_device": 0
  "weekend": 1
  "ad_position=top": 2
  "user_engaged": 3
  # ... limit to 10-20 most important features
```

### Key Principles for Analysis Trees

#### 1. Shallow Depth (max_depth=3)

**Why**: Stakeholders can understand 3-level trees visually

**Benefits**:
- Error at depth 3: **6.2%** (k=4096) → Very good accuracy ✓
- Tree fits on one slide/page
- Decision rules are simple: "IF mobile AND weekend AND top THEN high CTR"

**Example Tree Structure**:
```
Root: mobile_device?
├── False (Desktop)
│   └── Leaf: CTR = 0.5%
└── True (Mobile)
    ├── weekend?
    │   ├── False (Weekday): CTR = 1.2%
    │   └── True (Weekend)
    │       ├── ad_position=top?
    │       │   ├── False: CTR = 2.1%
    │       │   └── True: CTR = 4.3%
```

**Stakeholder Insight**: "Mobile users on weekends with top ad placement have 4.3% CTR (4× baseline)"

#### 2. Feature Selection (10-20 features)

**Why**: Too many features confuse stakeholders

**Strategy**:
- Use feature importance from initial run
- Select top 10-20 most important features
- Re-train with filtered feature set

```python
# Initial training with all features
clf_full = ThetaSketchDecisionTreeClassifier()
clf_full.fit(csv_path='sketches.csv', config_path='config.yaml')

# Get feature importance
importances = clf_full.feature_importances_
top_features = np.argsort(importances)[-20:]  # Top 20

# Re-train with top features only
config_filtered = update_feature_mapping(config, top_features)
clf_analysis = ThetaSketchDecisionTreeClassifier()
clf_analysis.fit(csv_path='sketches_filtered.csv', config_path='config_filtered.yaml')
```

#### 3. Strong Pruning (max_leaf_nodes=15)

**Why**: Simpler trees are easier to explain and visualize

**Trade-off**:
- Accuracy loss: 2-5% vs unpruned tree
- Interpretability gain: Tree fits on one page, stakeholders can understand every split

**Example Pruning Impact**:
```
Unpruned tree: 47 nodes, 24 leaves, test accuracy 82%
Pruned tree:   15 nodes, 8 leaves,  test accuracy 79%

Business value: 3% accuracy loss is acceptable for clear stakeholder communication
```

#### 4. Use Gini or Entropy (Not Binomial)

**Why**: Gini and entropy are familiar to business audiences

**Avoid**:
- Binomial/chi-square criteria (harder to explain)
- Statistical p-values (require statistical background)

**Stakeholder Communication**:
- ✅ "We split on mobile_device because it reduced impurity by 0.15 (high gain)"
- ❌ "We split on mobile_device because p-value was 0.001 (binomial test rejected null)"

### Visualization Strategies

#### 1. Tree Diagram Export

```python
# Export tree to JSON for visualization
tree_json = clf.export_tree_json()

# Save for web-based visualizer
with open('tree_viz.json', 'w') as f:
    json.dump(tree_json, f, indent=2)

# Or use graphviz (if implemented)
clf.plot_tree(output_path='tree_diagram.png', max_depth=3)
```

**Best Practices**:
- Color-code by predicted class (green=positive, red=negative)
- Show sample counts and class probabilities at each node
- Highlight decision path for example records
- Use descriptive feature names (not column indices)

#### 2. Feature Importance Chart

```python
import matplotlib.pyplot as plt

# Get feature importance
importances = clf.feature_importances_
feature_names = list(config['feature_mapping'].keys())

# Plot horizontal bar chart
plt.barh(feature_names, importances)
plt.xlabel('Importance (Gini decrease)')
plt.title('Top Features Influencing CTR')
plt.tight_layout()
plt.savefig('feature_importance.png')
```

#### 3. Decision Rule Extraction

```python
# Extract human-readable rules
rules = clf.export_decision_rules()

# Example output:
# Rule 1: IF mobile=True AND weekend=True AND ad_position=top THEN CTR=4.3% (confidence: 85%)
# Rule 2: IF mobile=False THEN CTR=0.5% (confidence: 92%)
# Rule 3: IF mobile=True AND weekend=False THEN CTR=1.2% (confidence: 78%)

# Save to stakeholder report
with open('business_rules.md', 'w') as f:
    for rule in rules:
        f.write(f"- {rule}\n")
```

#### 4. Segment Analysis Report

```python
# Analyze key segments
segments = {
    'Mobile Weekend Top': [1, 1, 1],
    'Desktop Weekday': [0, 0, 0],
    'Mobile Weekday Bottom': [1, 0, 0],
}

for segment_name, features in segments.items():
    prob = clf.predict_proba([features])[0, 1]
    print(f"{segment_name}: {prob:.1%} CTR")

# Generate report
# Mobile Weekend Top: 4.3% CTR (4.3× baseline)
# Desktop Weekday: 0.5% CTR (0.5× baseline)
# Mobile Weekday Bottom: 1.2% CTR (1.2× baseline)
```

### Expected Outcomes for Analysis Use Case

| Metric | Target | Actual (Typical) |
|--------|--------|------------------|
| **Tree Depth** | ≤3 | 3 |
| **Leaf Nodes** | 8-15 | 12 |
| **Accuracy** | >75% | 78-82% |
| **Error (k=4096, depth=3)** | <10% | 6.2% |
| **Feature Count** | 10-20 | 15 |
| **Visualization Time** | <5 min | 2 min |
| **Stakeholder Satisfaction** | High | ✅ |

---

## 3. Strategy 2: Production Deployment & Runtime Inference

### Recommended Configuration

```yaml
# config_production.yaml
targets:
  positive: "target_yes"
  negative: "target_no"

hyperparameters:
  # OPTIMIZE FOR ACCURACY

  # Statistical criterion for robustness
  criterion: "binomial"          # More rigorous than gini
  min_pvalue: 0.001              # Strict significance
  use_bonferroni: true

  # DEEPER TREE for accuracy
  max_depth: 5                   # 5-6 levels (accuracy sweet spot)
  min_samples_split: 500         # Balance accuracy and generalization
  min_samples_leaf: 250

  # Moderate pruning (preserve accuracy)
  pruning: "both"
  min_impurity_decrease: 0.002   # Less aggressive than analysis
  ccp_alpha: 0.0005              # Light post-pruning

  # Class weights for imbalanced data
  class_weight: "balanced"
  use_weighted_gini: true

  # Feature sampling for robustness
  max_features: "sqrt"           # Like Random Forest

  # Missing value handling
  missing_value_strategy: "majority"

  # Performance optimization
  use_cache: true
  cache_size_mb: 500             # Larger cache for training speed

  random_state: 42
  verbose: 1

feature_mapping:
  # ALL available features (100-5000)
  "feature_1": 0
  "feature_2": 1
  # ... all features
  "feature_5000": 4999
```

### Key Principles for Production Trees

#### 1. Deeper Depth (max_depth=5)

**Why**: Maximize accuracy for automated decisions

**Trade-off**:
- Accuracy gain: 5-10% vs shallow tree
- Error at depth 5: **12.2%** (k=4096, with feature-absent sketches) → Acceptable for CTR ✓
- Interpretability loss: Acceptable (model is "black box" for ML engineers)

**Production Requirements**:
- Test accuracy > 80% (CTR)
- ROC AUC > 0.75
- Precision/Recall balanced for business metric

#### 2. All Available Features (100-5000)

**Why**: Let tree discover complex patterns

**Strategy**:
- Include all features (no manual selection)
- Use max_features="sqrt" for random sampling (prevents overfitting)
- Feature importance still tracked for monitoring

**Benefits**:
- Captures rare but important feature interactions
- No human bias in feature selection
- Easier to maintain (no manual curation)

#### 3. Statistical Criterion (Binomial)

**Why**: More robust to noise than gini/entropy

**Benefits**:
- Splits are statistically significant (p<0.001)
- Less likely to overfit on noise
- Bonferroni correction for multiple testing
- Better for imbalanced data (CTR, fraud)

**Production Advantage**:
- Stable predictions across data drift
- Fewer false positive splits
- Better generalization to new data

#### 4. Light Pruning (ccp_alpha=0.0005)

**Why**: Preserve predictive accuracy

**Trade-off**:
- Tree complexity: 50-100 nodes (not visualizable, but accurate)
- Accuracy: Maximized
- Size: 1-5 MB pickled model (acceptable)

### Production Deployment Best Practices

#### 1. Model Serialization

```python
import pickle

# Train model
clf = ThetaSketchDecisionTreeClassifier()
clf.fit(csv_path='sketches.csv', config_path='config_production.yaml')

# Save model (pickle for speed)
with open('ctr_model_v1.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Metadata for version control
metadata = {
    'model_version': 'v1.0.0',
    'trained_date': '2025-01-15',
    'features': len(clf.feature_names_),
    'depth': clf.get_depth(),
    'test_accuracy': 0.823,
    'test_auc': 0.782,
    'sketch_size_k': 4096,
}

with open('ctr_model_v1_metadata.json', 'w') as f:
    json.dump(metadata, f)
```

#### 2. Fast Loading for API

```python
# Load model once at API startup (not per request!)
import pickle

class CTRPredictor:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded: {self.model.tree_.n_nodes} nodes")

    def predict_ctr(self, features):
        """Predict CTR for single impression (< 1ms)"""
        proba = self.model.predict_proba([features])[0, 1]
        return proba

# Initialize once
predictor = CTRPredictor('ctr_model_v1.pkl')

# Use in request handler (fast!)
@app.route('/predict_ctr', methods=['POST'])
def predict_ctr_api():
    features = request.json['features']  # Binary feature vector
    ctr = predictor.predict_ctr(features)
    return jsonify({'ctr': float(ctr)})
```

#### 3. Batch Inference Optimization

```python
# For batch scoring (millions of records)
import numpy as np

# Load test data
X_batch = np.load('impressions_batch.npy')  # Shape: (10M, 5000)

# Batch predict (vectorized, fast)
start_time = time.time()
ctr_scores = clf.predict_proba(X_batch)[:, 1]
elapsed = time.time() - start_time

print(f"Scored {len(X_batch):,} impressions in {elapsed:.2f}s")
print(f"Throughput: {len(X_batch)/elapsed:,.0f} samples/sec")

# Expected: 100K-1M samples/sec (depending on tree size and hardware)
```

#### 4. Missing Value Handling

```python
# Production data often has missing features
X_with_missing = np.array([
    [1, np.nan, 0, 1, np.nan],  # Some features missing
    [np.nan, np.nan, 1, 0, 1],  # More features missing
    [1, 0, 1, 1, 0],            # All features present
])

# Model handles missing values automatically (majority path)
predictions = clf.predict_proba(X_with_missing)[:, 1]

# No errors, graceful handling
```

#### 5. Monitoring & Retraining

```python
# Monitor model performance in production
from sklearn.metrics import roc_auc_score

# Weekly evaluation on new data
y_true = load_ground_truth_labels()  # Actual clicks
y_pred = clf.predict_proba(X_prod)[:, 1]

current_auc = roc_auc_score(y_true, y_pred)
print(f"Current AUC: {current_auc:.3f}")

# Alert if performance degrades
if current_auc < baseline_auc - 0.05:
    send_alert("Model performance degraded! Retrain needed.")

# Retrain monthly or when performance drops
if should_retrain():
    retrain_model(new_sketches_csv)
```

### Expected Outcomes for Production Use Case

| Metric | Target | Actual (Typical) |
|--------|--------|------------------|
| **Tree Depth** | 5-6 | 5 |
| **Leaf Nodes** | 30-60 | 45 |
| **Accuracy** | >80% | 82-85% |
| **ROC AUC** | >0.75 | 0.78-0.82 |
| **Inference Time** | <1ms | 0.3-0.8ms |
| **Model Size** | <10 MB | 2-5 MB |
| **Throughput** | >10K/sec | 100K-1M/sec |
| **Production Stability** | High | ✅ |

---

## 4. Hybrid Strategy: Best of Both Worlds

### The Challenge

**Problem**: Stakeholders want interpretable trees, but production needs accuracy

**Conflict**:
- Shallow trees (depth 3) → Interpretable but less accurate
- Deep trees (depth 5) → Accurate but not interpretable

### Solution: Dual Model Approach

#### Strategy A: Two Separate Trees

**Train two models with different goals**:

1. **Analysis Tree** (for stakeholders)
   - max_depth=3
   - max_leaf_nodes=15
   - Gini criterion
   - Top 20 features

2. **Production Tree** (for runtime)
   - max_depth=5
   - Binomial criterion
   - All features

**Workflow**:
```python
# Analysis tree (shallow, interpretable)
clf_analysis = ThetaSketchDecisionTreeClassifier()
clf_analysis.fit(
    csv_path='sketches.csv',
    config_path='config_analysis.yaml'  # max_depth=3
)
clf_analysis.export_tree_diagram('stakeholder_tree.png')
clf_analysis.export_decision_rules('business_rules.md')

# Production tree (deep, accurate)
clf_production = ThetaSketchDecisionTreeClassifier()
clf_production.fit(
    csv_path='sketches.csv',
    config_path='config_production.yaml'  # max_depth=5
)
clf_production.save_model('production_model.pkl')

# Stakeholders see analysis tree (simple)
# Production uses production tree (accurate)
```

**Benefits**:
- ✅ Stakeholders get simple, visualizable trees
- ✅ Production gets maximum accuracy
- ✅ No compromise on either goal

**Maintenance**:
- Both trees trained on same sketch data (consistent)
- Analysis tree updated quarterly (for reports)
- Production tree updated monthly (for accuracy)

---

#### Strategy B: Depth-Limited Explanation

**Use one production tree, but explain only top levels**:

```python
# Train full production tree (depth 5)
clf = ThetaSketchDecisionTreeClassifier()
clf.fit(csv_path='sketches.csv', config_path='config_production.yaml')

# Export only top 3 levels for stakeholders
tree_json_shallow = clf.export_tree_json(max_depth=3)
clf.plot_tree(output_path='stakeholder_view.png', max_depth=3)

# Use full tree for production inference
predictions = clf.predict_proba(X_prod)
```

**Benefits**:
- ✅ Single model to maintain
- ✅ Full accuracy for production
- ✅ Simplified view for stakeholders

**Trade-offs**:
- ⚠️ Stakeholder view is incomplete (missing deeper logic)
- ⚠️ May confuse stakeholders if predictions don't match top-3 levels

---

#### Strategy C: Rule Extraction from Deep Tree

**Extract simple rules from complex tree**:

```python
# Train complex tree
clf = ThetaSketchDecisionTreeClassifier()
clf.fit(csv_path='sketches.csv', config_path='config_production.yaml')  # depth=5

# Extract top K most important paths
important_paths = clf.extract_important_paths(top_k=10)

# Present to stakeholders as decision rules
for i, path in enumerate(important_paths, 1):
    print(f"Rule {i}: {path['conditions']} → CTR={path['ctr']:.2%} (covers {path['coverage']:.1%} of data)")

# Example output:
# Rule 1: mobile=True AND weekend=True AND ad_position=top → CTR=4.3% (covers 12% of data)
# Rule 2: mobile=False → CTR=0.5% (covers 40% of data)
# Rule 3: mobile=True AND user_engaged=True → CTR=2.8% (covers 18% of data)
```

**Benefits**:
- ✅ Stakeholders get actionable rules
- ✅ Production uses full tree
- ✅ Rules are data-driven (not manually simplified)

---

## 5. Migration Path: Analysis → Deployment

### Phase 1: Initial Analysis (Week 1-2)

**Goal**: Understand data, build stakeholder trust

**Steps**:
1. Train shallow analysis tree (max_depth=3)
2. Generate visualizations and reports
3. Present to stakeholders
4. Iterate based on feedback
5. Identify key features and patterns

**Deliverables**:
- Tree diagram
- Feature importance chart
- Business rules document
- Stakeholder presentation

---

### Phase 2: Production Prototype (Week 3-4)

**Goal**: Build accurate model for A/B testing

**Steps**:
1. Train deeper production tree (max_depth=5)
2. Evaluate on holdout set
3. Run offline backtests
4. Deploy to A/B test (5% traffic)
5. Monitor performance

**Metrics**:
- Test accuracy
- ROC AUC
- Online CTR lift (vs baseline)
- Latency (p50, p95, p99)

---

### Phase 3: Scale to Production (Week 5-6)

**Goal**: Roll out to 100% traffic

**Steps**:
1. Optimize inference speed
2. Set up monitoring and alerts
3. Ramp to 50% traffic
4. Monitor for issues
5. Ramp to 100% traffic

**Success Criteria**:
- Online CTR lift > 10%
- p99 latency < 5ms
- No error rate increase
- Stakeholder approval

---

### Phase 4: Ongoing Maintenance

**Analysis Tree** (Quarterly):
- Re-train on latest 6 months of data
- Update stakeholder reports
- Communicate new insights

**Production Tree** (Monthly):
- Re-train on latest 3 months of data
- Monitor performance drift
- A/B test before deploying new version

---

## 6. Summary Comparison

| Aspect | **Analysis/Stakeholder** | **Production/Deployment** |
|--------|--------------------------|---------------------------|
| **Primary Goal** | Interpretability | Accuracy |
| **Audience** | Business stakeholders | ML engineers, production systems |
| **Tree Depth** | 3 levels | 5-6 levels |
| **Leaf Nodes** | 8-15 | 30-60 |
| **Features** | Top 10-20 (curated) | All 100-5000 features |
| **Criterion** | Gini or Entropy | Binomial (statistical) |
| **Pruning** | Aggressive (simple tree) | Light (preserve accuracy) |
| **Accuracy** | 75-80% | 80-85% |
| **Error (k=4096)** | 6.2% @ depth 3 | 12.2% @ depth 5 |
| **Visualization** | Tree diagram, rules | JSON export only |
| **Inference Speed** | Not critical | <1ms per sample |
| **Model Size** | <1 MB | 2-5 MB |
| **Update Frequency** | Quarterly | Monthly |
| **Deployment** | Jupyter notebook, reports | REST API, batch scoring |
| **Monitoring** | Manual review | Automated metrics |
| **Success Metric** | Stakeholder satisfaction | Online lift, ROC AUC |

---

## Best Practices Summary

### For Analysis Use Case:
✅ **Use max_depth=3** for visualization
✅ **Limit to 10-20 top features** for clarity
✅ **Use Gini/Entropy** criterion (familiar to business)
✅ **Strong pruning** (max_leaf_nodes=15)
✅ **Export visualizations** (tree diagrams, rules)
✅ **Update quarterly** with new insights

### For Production Use Case:
✅ **Use max_depth=5** for accuracy
✅ **Include all features** (100-5000)
✅ **Use Binomial** criterion (statistically robust)
✅ **Light pruning** (preserve accuracy)
✅ **Optimize for inference speed** (<1ms)
✅ **Monitor and retrain monthly**

### For Hybrid Approach:
✅ **Train two separate models** (best option)
✅ **Analysis tree** for stakeholder communication
✅ **Production tree** for runtime inference
✅ **Consistent sketch data** for both
✅ **Extract simple rules** from complex tree for reporting

---

**Key Takeaway**: The same theta sketch infrastructure supports both use cases with different hyperparameter configurations. Choose the strategy that matches your organization's needs, or use a hybrid approach for maximum value.
