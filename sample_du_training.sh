#!/bin/bash
#
# Complete DU (Driver Utilization) Training Pipeline with lg_k=18
#
# This script runs the full pipeline from raw CSV to trained model with rule extraction:
#   1. Raw CSV → 2-column sketches
#   2. 2-column → 3-column sketches
#   3. 3-column sketches → trained model
#   4. Extract decision rules for manual validation
#
# Usage: ./sample_du_training.sh
#

set -e  # Exit on any error

# Configuration
LG_K=18
INPUT_FILE="tests/resources/DU_raw.csv"
TARGET_COLUMN="tripOutcome"
OUTPUT_DIR="DU_output"
MODEL_NAME="du_model_lg_k_${LG_K}"

# Tree hyperparameters
CRITERION="gini"
MAX_DEPTH=6
MIN_SAMPLES_SPLIT=100
MIN_SAMPLES_LEAF=50

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚗 DU Training Pipeline Starting${NC}"
echo "========================================"
echo "Input file: ${INPUT_FILE}"
echo "Target column: ${TARGET_COLUMN}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Model name: ${MODEL_NAME}"
echo "lg_k: ${LG_K}"
echo "Tree parameters: criterion=${CRITERION}, max_depth=${MAX_DEPTH}, min_samples_split=${MIN_SAMPLES_SPLIT}, min_samples_leaf=${MIN_SAMPLES_LEAF}"
echo ""

# Check prerequisites
echo -e "${BLUE}🔍 Checking prerequisites...${NC}"

if [ ! -f "${INPUT_FILE}" ]; then
    echo -e "${RED}❌ Error: Input file ${INPUT_FILE} not found${NC}"
    echo "Please ensure DU_raw.csv exists in the current directory"
    exit 1
fi

if [ ! -f "venv/bin/python" ]; then
    echo -e "${RED}❌ Error: Python virtual environment not found${NC}"
    echo "Please ensure venv/bin/python exists"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"
echo ""

# Create output directory
echo -e "${BLUE}📁 Setting up output directory...${NC}"
mkdir -p "${OUTPUT_DIR}"
echo -e "${GREEN}✅ Output directory ready: ${OUTPUT_DIR}${NC}"
echo ""

# Step 1: Raw CSV → 2-Column Sketches
echo -e "${BLUE}📊 Step 1: Generating 2-column sketches from raw CSV...${NC}"
echo "Command: ./venv/bin/python tools/create_2col_sketches.py --input ${INPUT_FILE} --output ${OUTPUT_DIR}/ --lg_k ${LG_K} --skip_columns tripId"

./venv/bin/python tools/create_2col_sketches.py --input "${INPUT_FILE}" --output "${OUTPUT_DIR}/" --lg_k ${LG_K} --skip_columns tripId

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Step 1 completed successfully${NC}"

    # Check outputs
    TWOCOL_FILE="${OUTPUT_DIR}/DU_raw_2col_sketches_lg_k_${LG_K}.csv"

    echo "Generated files:"
    echo "  - 2-column sketches: ${TWOCOL_FILE}"

    # Show file sizes
    if [ -f "${TWOCOL_FILE}" ]; then
        SIZE=$(du -h "${TWOCOL_FILE}" | cut -f1)
        echo "  - 2-column sketches size: ${SIZE}"

        # Show feature count
        FEATURE_COUNT=$(tail -n +2 "${TWOCOL_FILE}" | wc -l)
        echo "  - Number of features: ${FEATURE_COUNT}"
    fi
else
    echo -e "${RED}❌ Step 1 failed${NC}"
    exit 1
fi
echo ""

# Step 2: 2-Column → 3-Column Sketches
echo -e "${BLUE}🔄 Step 2: Converting 2-column to 3-column sketches...${NC}"

THREECOL_OUTPUT_POS="${OUTPUT_DIR}/positive_3col_sketches_lg_k_${LG_K}.csv"
THREECOL_OUTPUT_NEG="${OUTPUT_DIR}/negative_3col_sketches_lg_k_${LG_K}.csv"
FEATURE_MAPPING="${OUTPUT_DIR}/DU_feature_mapping.json"

echo "Command: ./venv/bin/python tools/simple_convert_to_3col.py --input ${TWOCOL_FILE} --suffix _3col_sketches_lg_k_${LG_K} --mapping DU_feature_mapping.json --output ${OUTPUT_DIR}/ --target ${TARGET_COLUMN}"

./venv/bin/python tools/simple_convert_to_3col.py \
    --input "${TWOCOL_FILE}" \
    --suffix "_3col_sketches_lg_k_${LG_K}" \
    --mapping "DU_feature_mapping.json" \
    --output "${OUTPUT_DIR}/" \
    --target "${TARGET_COLUMN}"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Step 2 completed successfully${NC}"
    echo "Generated files:"
    echo "  - Positive 3-column sketches: ${THREECOL_OUTPUT_POS}"
    echo "  - Negative 3-column sketches: ${THREECOL_OUTPUT_NEG}"
    echo "  - Feature mapping: ${FEATURE_MAPPING}"

    if [ -f "${THREECOL_OUTPUT_POS}" ]; then
        POS_SIZE=$(du -h "${THREECOL_OUTPUT_POS}" | cut -f1)
        echo "  - Positive sketches size: ${POS_SIZE}"
    fi
    if [ -f "${THREECOL_OUTPUT_NEG}" ]; then
        NEG_SIZE=$(du -h "${THREECOL_OUTPUT_NEG}" | cut -f1)
        echo "  - Negative sketches size: ${NEG_SIZE}"
    fi
    if [ -f "${FEATURE_MAPPING}" ]; then
        # Show feature count from mapping
        FEATURE_COUNT=$(jq 'length' "${FEATURE_MAPPING}" 2>/dev/null || echo "Unknown")
        echo "  - Number of features: ${FEATURE_COUNT}"
    fi
else
    echo -e "${RED}❌ Step 2 failed${NC}"
    exit 1
fi
echo ""

# Step 3: 3-Column Sketches → Trained Model
echo -e "${BLUE}🌳 Step 3: Training decision tree model...${NC}"

MODEL_OUTPUT="${OUTPUT_DIR}/${MODEL_NAME}"
echo "Command: ./venv/bin/python tools/train_from_3col_sketches.py --lg_k ${LG_K} --positive ${THREECOL_OUTPUT_POS} --negative ${THREECOL_OUTPUT_NEG} --config configs/du_config.yaml --output ${MODEL_OUTPUT}"

./venv/bin/python tools/train_from_3col_sketches.py \
    --lg_k ${LG_K} \
    --positive "${THREECOL_OUTPUT_POS}" \
    --negative "${THREECOL_OUTPUT_NEG}" \
    --config configs/du_config.yaml \
    --output "${MODEL_OUTPUT}"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Step 3 completed successfully${NC}"

    MODEL_PKL="${MODEL_OUTPUT}/3col_sketches_model_lg_k_${LG_K}.pkl"
    MODEL_JSON="${MODEL_OUTPUT}/3col_sketches_model_lg_k_${LG_K}.json"

    echo "Generated files:"
    echo "  - Trained model: ${MODEL_PKL}"
    echo "  - Tree structure JSON: ${MODEL_JSON}"

    if [ -f "${MODEL_PKL}" ]; then
        PKL_SIZE=$(du -h "${MODEL_PKL}" | cut -f1)
        echo "  - Model file size: ${PKL_SIZE}"
    fi
    if [ -f "${MODEL_JSON}" ]; then
        JSON_SIZE=$(du -h "${MODEL_JSON}" | cut -f1)
        echo "  - JSON file size: ${JSON_SIZE}"
    fi
else
    echo -e "${RED}❌ Step 3 failed${NC}"
    exit 1
fi
echo ""

# Step 4: Extract Decision Rules
echo -e "${BLUE}📋 Step 4: Extracting decision rules for validation...${NC}"

RULES_JSON="${OUTPUT_DIR}/tree_rules.json"
RULES_SQL="${OUTPUT_DIR}/tree_validation_queries.sql"
QUICK_RULES="${OUTPUT_DIR}/quick_test_rules.txt"

echo "Command: ./venv/bin/python tools/extract_tree_rules.py ${MODEL_JSON} --save_rules ${RULES_JSON} --save_sql ${RULES_SQL} --table DU_raw --target_column ${TARGET_COLUMN}"

./venv/bin/python tools/extract_tree_rules.py \
    "${MODEL_JSON}" \
    --save_rules "${RULES_JSON}" \
    --save_sql "${RULES_SQL}" \
    --table DU_raw \
    --target_column "${TARGET_COLUMN}" \
    --quiet

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Decision rules extracted successfully${NC}"
    echo "Generated files:"
    echo "  - Complete rules: ${RULES_JSON}"
    echo "  - SQL validation queries: ${RULES_SQL}"
else
    echo -e "${YELLOW}⚠️  Rule extraction encountered issues, but continuing...${NC}"
fi

# Generate quick test rules
echo -e "${BLUE}🔍 Generating quick test rules...${NC}"
echo "Command: ./venv/bin/python tools/quick_tree_test.py ${MODEL_JSON} --num_rules 5 --table DU_raw --target ${TARGET_COLUMN}"

./venv/bin/python tools/quick_tree_test.py \
    "${MODEL_JSON}" \
    --num_rules 5 \
    --table DU_raw \
    --target "${TARGET_COLUMN}" > "${QUICK_RULES}" 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Quick test rules generated: ${QUICK_RULES}${NC}"
else
    echo -e "${YELLOW}⚠️  Quick test rule generation encountered issues${NC}"
fi
echo ""

# Summary Report
echo -e "${GREEN}🎉 DU Training Pipeline Completed Successfully!${NC}"
echo "========================================"
echo ""
echo -e "${BLUE}📁 Generated Files in ${OUTPUT_DIR}/:${NC}"

for file in "${OUTPUT_DIR}"/*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        size=$(du -h "$file" | cut -f1)
        echo "  📄 ${filename} (${size})"
    fi
done

# Show directories
for dir in "${OUTPUT_DIR}"/*; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir")
        echo "  📁 ${dirname}/"
        for subfile in "$dir"/*; do
            if [ -f "$subfile" ]; then
                subfilename=$(basename "$subfile")
                size=$(du -h "$subfile" | cut -f1)
                echo "    📄 ${subfilename} (${size})"
            fi
        done
    fi
done

echo ""
echo -e "${BLUE}🔍 Next Steps for Manual Validation:${NC}"
echo "1. Open your SQL client and connect to your database"
echo "2. Run queries from: ${RULES_SQL}"
echo "3. Compare actual vs expected results"
echo "4. Quick test queries available in: ${QUICK_RULES}"
echo ""
echo -e "${BLUE}📊 Model Information:${NC}"
echo "  - Algorithm: CART Decision Tree"
echo "  - Training method: Theta sketches (lg_k=${LG_K})"
echo "  - Criterion: ${CRITERION}"
echo "  - Max depth: ${MAX_DEPTH}"
echo "  - Min samples split: ${MIN_SAMPLES_SPLIT}"
echo "  - Min samples leaf: ${MIN_SAMPLES_LEAF}"
echo ""

# Show quick preview of rules
if [ -f "${QUICK_RULES}" ]; then
    echo -e "${BLUE}🔍 Quick Test Rules Preview:${NC}"
    head -30 "${QUICK_RULES}"
    echo ""
    echo "Full rules available in: ${QUICK_RULES}"
fi

echo -e "${GREEN}✅ Pipeline completed at $(date)${NC}"
echo ""
echo -e "${YELLOW}💡 Tips:${NC}"
echo "  - Use the SQL queries in ${RULES_SQL} to validate tree logic"
echo "  - Model file ${MODEL_PKL} can be loaded for predictions"
echo "  - JSON tree structure in ${MODEL_JSON} shows decision paths"
echo "  - All intermediate files preserved for debugging"
echo ""