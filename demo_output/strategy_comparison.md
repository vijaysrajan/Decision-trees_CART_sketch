# Classification Strategy Comparison

| Strategy | Best For | Key Parameter | Pros | Cons |
|----------|----------|---------------|------|------|
| **Class Weightage** | Imbalanced costs | `positive_class_weight` | Simple, direct cost optimization | Requires cost estimation |
| **Binomial P-value** | Statistical rigor | `significance_level` | Principled hypothesis testing | May be conservative |
| **Ratio Threshold** | Interpretable rules | `ratio_threshold` | Easy to explain, fast | Ignores uncertainty |
| **Confidence Intervals** | Uncertainty quantification | `confidence_level` | Accounts for sample size | More complex |

## When to Use Each Strategy

### Class Weightage
- Clear business costs for errors
- Imbalanced datasets
- Direct optimization goals

### Binomial P-value
- Regulatory requirements
- Scientific applications
- Need to prove significance

### Ratio Threshold
- Simple business rules
- Fast decision making
- Easy stakeholder communication

### Confidence Intervals
- Uncertainty matters
- Variable sample sizes
- Risk-sensitive applications

## Collateral Damage Constraint
Can be applied to ANY strategy as an additional safety constraint:
- Minimum number of "safe" samples required
- Prevents high-risk predictions on small populations
- Essential for safety-critical applications
