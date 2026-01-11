#!/bin/bash
# Pipeline Script 4: Advanced Classification Strategies Demo
# Usage: ./demo_advanced_classification.sh [model.pkl] [test_data.csv] [strategy] [output_dir]

set -e  # Exit on any error

# Parse arguments
MODEL_FILE="${1:-DU_output/du_model_lg_k_18/3col_sketches_lg_k_18_model_lg_k_18.pkl}"
TEST_DATA="${2:-tests/resources/DU_raw.csv}"
STRATEGY="${3:-all}"
OUTPUT_DIR="${4:-demo_output}"

echo "🎯 Advanced Classification Strategies Demo"
echo "=========================================="
echo "Model:       $MODEL_FILE"
echo "Test data:   $TEST_DATA"
echo "Strategy:    $STRATEGY"
echo "Output dir:  $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run strategy demo
run_strategy_demo() {
    local strategy_name="$1"
    local description="$2"
    local config_file="${OUTPUT_DIR}/${strategy_name}_config.yaml"
    local predictions_file="${OUTPUT_DIR}/${strategy_name}_predictions.json"

    echo "🔮 Testing Strategy: $strategy_name"
    echo "   Description: $description"

    # Create strategy-specific config
    case "$strategy_name" in
        "class_weightage")
            cat > "$config_file" << EOF
strategy: "class_weightage"
positive_class_weight: 3.0
negative_class_weight: 1.0
min_collateral_damage: 50
EOF
            ;;
        "binomial_pvalue")
            cat > "$config_file" << EOF
strategy: "binomial_pvalue"
population_positive_rate: 0.15
significance_level: 0.05
min_collateral_damage: 100
EOF
            ;;
        "ratio_threshold")
            cat > "$config_file" << EOF
strategy: "ratio_threshold"
ratio_threshold: 0.30
min_samples_for_prediction: 20
min_collateral_damage: 25
EOF
            ;;
        "confidence_intervals")
            cat > "$config_file" << EOF
strategy: "confidence_intervals"
confidence_level: 0.95
decision_threshold: 0.25
min_collateral_damage: 75
EOF
            ;;
    esac

    # Check if advanced classifier can run (might fail due to model format issues)
    if [ -f "$MODEL_FILE" ] && [ -f "$TEST_DATA" ]; then
        echo "   Running advanced classifier..."
        if ./venv/bin/python tools/advanced_classifier.py \
            --model "$MODEL_FILE" \
            --data "$TEST_DATA" \
            --config "$config_file" \
            --output "$predictions_file" \
            --summary 2>/dev/null; then
            echo "   ✅ Strategy completed successfully"
            echo "   Output: $predictions_file"
        else
            echo "   ⚠️  Advanced classifier failed (likely model format issue)"
            echo "   Config saved: $config_file"
        fi
    else
        echo "   ⚠️  Model or test data not found"
        echo "   Config saved: $config_file"
    fi
    echo ""
}

# Demo Section 1: Strategy Demonstrations
echo "📋 Part 1: Strategy Demonstration (Simulated)"
echo "=============================================="
./venv/bin/python demo_advanced_strategies.py
echo ""

# Demo Section 2: Configuration Examples
echo "📋 Part 2: Configuration Examples"
echo "=================================="

if [ "$STRATEGY" = "all" ] || [ "$STRATEGY" = "class_weightage" ]; then
    run_strategy_demo "class_weightage" "Business cost optimization - False negatives cost 3x more"
fi

if [ "$STRATEGY" = "all" ] || [ "$STRATEGY" = "binomial_pvalue" ]; then
    run_strategy_demo "binomial_pvalue" "Statistical significance testing"
fi

if [ "$STRATEGY" = "all" ] || [ "$STRATEGY" = "ratio_threshold" ]; then
    run_strategy_demo "ratio_threshold" "Simple threshold with minimum samples"
fi

if [ "$STRATEGY" = "all" ] || [ "$STRATEGY" = "confidence_intervals" ]; then
    run_strategy_demo "confidence_intervals" "Uncertainty-aware decisions using Wilson intervals"
fi

# Demo Section 3: Business Scenarios
echo "📋 Part 3: Business Scenarios"
echo "=============================="

cat > "${OUTPUT_DIR}/business_scenarios.md" << 'EOF'
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
EOF

echo "📄 Business scenarios documented: ${OUTPUT_DIR}/business_scenarios.md"

# Demo Section 4: Comparison Summary
echo ""
echo "📋 Part 4: Strategy Comparison Summary"
echo "====================================="

cat > "${OUTPUT_DIR}/strategy_comparison.md" << 'EOF'
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
EOF

echo "📄 Strategy comparison documented: ${OUTPUT_DIR}/strategy_comparison.md"

# Demo Section 5: Configuration Templates
echo ""
echo "📋 Part 5: Configuration Templates"
echo "=================================="

# Copy the main config template
cp configs/classification_strategies.yaml "${OUTPUT_DIR}/template_config.yaml"

echo "📄 Configuration templates available:"
echo "   Main template: ${OUTPUT_DIR}/template_config.yaml"
echo "   Strategy configs: ${OUTPUT_DIR}/*_config.yaml"

echo ""
echo "🎉 Advanced Classification Demo Completed!"
echo "=========================================="
echo ""
echo "📁 Demo outputs available in: $OUTPUT_DIR"
echo ""
echo "📋 Summary of deliverables:"
echo "   • Strategy demonstration (console output above)"
echo "   • Business scenarios: ${OUTPUT_DIR}/business_scenarios.md"
echo "   • Strategy comparison: ${OUTPUT_DIR}/strategy_comparison.md"
echo "   • Configuration templates: ${OUTPUT_DIR}/*_config.yaml"
echo "   • Template config: ${OUTPUT_DIR}/template_config.yaml"
echo ""
echo "🔍 Next steps:"
echo "   1. Review business scenarios for your use case"
echo "   2. Customize configuration templates"
echo "   3. Test with your own model and data"
echo "   4. Integrate into production pipeline"
echo ""
echo "💡 For customer demos:"
echo "   • Show console output demonstrating strategy differences"
echo "   • Highlight business scenario matching their needs"
echo "   • Demonstrate configuration flexibility"
echo "   • Emphasize post-processing approach (non-invasive)"