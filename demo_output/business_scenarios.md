# Business Classification Scenarios

## 1. Customer Retention (Class Weightage)
**Use Case**: Predicting customer churn
**Challenge**: False negatives (missing churners) cost $1000, false positives (unnecessary retention efforts) cost $50
**Strategy**: Class weightage with 20:1 ratio
**Config**: `positive_class_weight: 20.0, negative_class_weight: 1.0`

## 2. Fraud Detection (Binomial P-value)
**Use Case**: Credit card fraud detection
**Challenge**: Need statistical confidence that pattern is significantly different from normal
**Strategy**: Binomial testing against baseline fraud rate
**Config**: `population_positive_rate: 0.001, significance_level: 0.01`

## 3. Medical Diagnosis (Confidence Intervals)
**Use Case**: Disease screening
**Challenge**: Need uncertainty bounds on predictions
**Strategy**: Wilson score intervals with conservative thresholds
**Config**: `confidence_level: 0.99, decision_threshold: 0.10`

## 4. Marketing Campaign (Ratio Threshold + Collateral)
**Use Case**: Targeting high-value prospects
**Challenge**: Need reliable predictions and minimum campaign size
**Strategy**: Simple threshold with sample size requirements
**Config**: `ratio_threshold: 0.25, min_samples_for_prediction: 100, min_collateral_damage: 500`

## 5. Safety Critical (Multiple Constraints)
**Use Case**: Industrial equipment failure prediction
**Challenge**: High confidence required, minimum risk exposure
**Strategy**: Confidence intervals + collateral damage
**Config**: Multiple overlapping safety constraints
