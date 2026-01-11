#!/bin/bash
# Pipeline Script 3: 3-Column Sketches → Model Trainer
# Usage: ./pipeline_3col_to_model.sh <positive_3col.csv> <negative_3col.csv> <lg_k> <output_dir> [config.yaml]

set -e  # Exit on any error

# Parse arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 <positive_3col.csv> <negative_3col.csv> <lg_k> <output_dir> [config.yaml]"
    echo ""
    echo "Examples:"
    echo "  $0 DU_output/positive_3col_sketches_lg_k_16.csv DU_output/negative_3col_sketches_lg_k_16.csv 16 models/"
    echo "  $0 mushroom_positive.csv mushroom_negative.csv 12 mushroom_models/ configs/mushroom_config.yaml"
    echo ""
    echo "Arguments:"
    echo "  positive_3col.csv - Input 3-column sketches for positive class"
    echo "  negative_3col.csv - Input 3-column sketches for negative class"
    echo "  lg_k              - Theta sketch precision parameter (12-18 typical)"
    echo "  output_dir        - Directory for model outputs (will be created)"
    echo "  config.yaml       - Optional: Tree training config (default: configs/du_config.yaml)"
    exit 1
fi

POSITIVE_3COL="$1"
NEGATIVE_3COL="$2"
LG_K="$3"
OUTPUT_DIR="$4"
CONFIG="${5:-configs/du_config.yaml}"

# Derive dataset name and model directory
POSITIVE_BASENAME=$(basename "$POSITIVE_3COL" .csv)
DATASET_NAME=$(echo "$POSITIVE_BASENAME" | sed 's/positive_3col_sketches_lg_k_[0-9]*//g' | sed 's/^_//g' | sed 's/_$//g')
if [ -z "$DATASET_NAME" ]; then
    DATASET_NAME="model"
fi
MODEL_DIR="${OUTPUT_DIR}/${DATASET_NAME}_model_lg_k_${LG_K}"

echo "🌳 Running 3-Column Sketches → Model Pipeline"
echo "=============================================="
echo "Positive:    $POSITIVE_3COL"
echo "Negative:    $NEGATIVE_3COL"
echo "lg_k:        $LG_K"
echo "Output dir:  $OUTPUT_DIR"
echo "Config:      $CONFIG"
echo "Dataset:     $DATASET_NAME"
echo ""

# Validate input files exist
if [ ! -f "$POSITIVE_3COL" ]; then
    echo "❌ Error: Positive sketches file not found: $POSITIVE_3COL"
    exit 1
fi

if [ ! -f "$NEGATIVE_3COL" ]; then
    echo "❌ Error: Negative sketches file not found: $NEGATIVE_3COL"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "❌ Error: Config file not found: $CONFIG"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Train Model from 3-Column Sketches
echo "🌳 Step 1: Training decision tree model..."
./venv/bin/python tools/train_from_3col_sketches.py \
    --lg_k "$LG_K" \
    --positive "$POSITIVE_3COL" \
    --negative "$NEGATIVE_3COL" \
    --config "$CONFIG" \
    --output "$MODEL_DIR"

MODEL_JSON="${MODEL_DIR}/3col_sketches_lg_k_${LG_K}_model_lg_k_${LG_K}.json"
MODEL_PKL="${MODEL_DIR}/3col_sketches_lg_k_${LG_K}_model_lg_k_${LG_K}.pkl"

echo "✅ Step 1 completed"
echo "   Output: $MODEL_PKL"
echo "   Output: $MODEL_JSON"
echo ""

# Step 2: Extract Decision Rules
echo "📋 Step 2: Extracting decision rules..."

# Determine target column name from config if possible
TARGET_COL="target"
if [ -f "$CONFIG" ]; then
    # Try to extract target column from YAML config
    TARGET_COL=$(grep -E "^[[:space:]]*target[[:space:]]*:" "$CONFIG" 2>/dev/null | sed 's/.*target[[:space:]]*:[[:space:]]*\([^[:space:]#]*\).*/\1/' || echo "target")
fi

./venv/bin/python tools/extract_tree_rules.py \
    "$MODEL_JSON" \
    --save_rules "${OUTPUT_DIR}/tree_rules.json" \
    --save_sql "${OUTPUT_DIR}/tree_validation_queries.sql" \
    --table "$DATASET_NAME" \
    --target_column "$TARGET_COL" \
    --quiet

./venv/bin/python tools/quick_tree_test.py \
    "$MODEL_JSON" \
    --num_rules 5 \
    --table "$DATASET_NAME" \
    --target "$TARGET_COL" > "${OUTPUT_DIR}/quick_test_rules.txt"

echo "✅ Step 2 completed"
echo "   Output: ${OUTPUT_DIR}/tree_rules.json"
echo "   Output: ${OUTPUT_DIR}/tree_validation_queries.sql"
echo "   Output: ${OUTPUT_DIR}/quick_test_rules.txt"
echo ""

# Step 3: Model Summary
echo "📊 Step 3: Model summary..."
echo "Model file size: $(du -h "$MODEL_PKL" | cut -f1)"
echo "JSON file size:  $(du -h "$MODEL_JSON" | cut -f1)"

# Count rules
if [ -f "${OUTPUT_DIR}/tree_rules.json" ]; then
    RULE_COUNT=$(grep -c '"node_id"' "${OUTPUT_DIR}/tree_rules.json" 2>/dev/null || echo "unknown")
    echo "Decision rules:  $RULE_COUNT"
fi

echo "✅ Step 3 completed"
echo ""

echo "🎉 Model training pipeline completed successfully!"
echo "Model and analysis files available in: $OUTPUT_DIR"
echo ""
echo "🔍 Quick validation:"
echo "  1. Check model performance in training logs"
echo "  2. Review decision rules: ${OUTPUT_DIR}/quick_test_rules.txt"
echo "  3. Use SQL queries for validation: ${OUTPUT_DIR}/tree_validation_queries.sql"