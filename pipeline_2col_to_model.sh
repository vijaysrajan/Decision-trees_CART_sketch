#!/bin/bash
# Pipeline Script 2: 2-Column Sketches → 3-Column Sketches → Model
# Usage: ./pipeline_2col_to_model.sh <2col_sketches.csv> <lg_k> <output_dir> <target_column> [config.yaml]

set -e  # Exit on any error

# Parse arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 <2col_sketches.csv> <lg_k> <output_dir> <target_column> [config.yaml]"
    echo ""
    echo "Examples:"
    echo "  $0 DU_output/DU_raw_2col_sketches_lg_k_16.csv 16 models/ tripOutcome"
    echo "  $0 mushroom_2col_sketches.csv 12 mushroom_models/ edible configs/mushroom_config.yaml"
    echo ""
    echo "Arguments:"
    echo "  2col_sketches.csv - Input 2-column sketches file"
    echo "  lg_k             - Theta sketch precision parameter (12-18 typical)"
    echo "  output_dir       - Directory for outputs (will be created)"
    echo "  target_column    - Name of target column in original data"
    echo "  config.yaml      - Optional: Tree training config (default: configs/du_config.yaml)"
    exit 1
fi

SKETCHES_2COL="$1"
LG_K="$2"
OUTPUT_DIR="$3"
TARGET_COL="$4"
CONFIG="${5:-configs/du_config.yaml}"

# Derive filenames
BASENAME=$(basename "$SKETCHES_2COL" .csv)
DATASET_NAME=$(echo "$BASENAME" | sed 's/_2col_sketches_lg_k_[0-9]*//g')
POSITIVE_3COL="${OUTPUT_DIR}/positive_3col_sketches_lg_k_${LG_K}.csv"
NEGATIVE_3COL="${OUTPUT_DIR}/negative_3col_sketches_lg_k_${LG_K}.csv"
MAPPING_FILE="${OUTPUT_DIR}/${DATASET_NAME}_feature_mapping.json"
MODEL_DIR="${OUTPUT_DIR}/${DATASET_NAME}_model_lg_k_${LG_K}"

echo "🔄 Running 2-Column Sketches → Model Pipeline"
echo "=============================================="
echo "Input:       $SKETCHES_2COL"
echo "lg_k:        $LG_K"
echo "Output dir:  $OUTPUT_DIR"
echo "Target:      $TARGET_COL"
echo "Config:      $CONFIG"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: 2-Column Sketches → 3-Column Sketches
echo "🔄 Step 1: Converting to 3-column sketches..."
./venv/bin/python tools/simple_convert_to_3col.py \
    --input "$SKETCHES_2COL" \
    --suffix "_3col_sketches_lg_k_${LG_K}" \
    --mapping "$MAPPING_FILE" \
    --output "$OUTPUT_DIR" \
    --target "$TARGET_COL"

echo "✅ Step 1 completed"
echo "   Output: $POSITIVE_3COL"
echo "   Output: $NEGATIVE_3COL"
echo "   Output: $MAPPING_FILE"
echo ""

# Step 2: 3-Column Sketches → Trained Model
echo "🌳 Step 2: Training decision tree model..."
./venv/bin/python tools/train_from_3col_sketches.py \
    --lg_k "$LG_K" \
    --positive "$POSITIVE_3COL" \
    --negative "$NEGATIVE_3COL" \
    --config "$CONFIG" \
    --output "$MODEL_DIR"

echo "✅ Step 2 completed"
echo "   Output: ${MODEL_DIR}/3col_sketches_lg_k_${LG_K}_model_lg_k_${LG_K}.pkl"
echo "   Output: ${MODEL_DIR}/3col_sketches_lg_k_${LG_K}_model_lg_k_${LG_K}.json"
echo ""

# Step 3: Extract Decision Rules
echo "📋 Step 3: Extracting decision rules..."
./venv/bin/python tools/extract_tree_rules.py \
    "${MODEL_DIR}/3col_sketches_lg_k_${LG_K}_model_lg_k_${LG_K}.json" \
    --save_rules "${OUTPUT_DIR}/tree_rules.json" \
    --save_sql "${OUTPUT_DIR}/tree_validation_queries.sql" \
    --table "$DATASET_NAME" \
    --target_column "$TARGET_COL" \
    --quiet

./venv/bin/python tools/quick_tree_test.py \
    "${MODEL_DIR}/3col_sketches_lg_k_${LG_K}_model_lg_k_${LG_K}.json" \
    --num_rules 5 \
    --table "$DATASET_NAME" \
    --target "$TARGET_COL" > "${OUTPUT_DIR}/quick_test_rules.txt"

echo "✅ Step 3 completed"
echo "   Output: ${OUTPUT_DIR}/tree_rules.json"
echo "   Output: ${OUTPUT_DIR}/tree_validation_queries.sql"
echo "   Output: ${OUTPUT_DIR}/quick_test_rules.txt"
echo ""

echo "🎉 Pipeline completed successfully!"
echo "Model and rules available in: $OUTPUT_DIR"