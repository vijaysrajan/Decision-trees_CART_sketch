#!/bin/bash
# Complete DU Pipeline Commands
# Updated to use correct file paths and tool arguments

set -e  # Exit on any error

echo "🚗 Running DU Pipeline with lg_k=18"
echo "==================================="

# Step 1: Raw CSV → 2-Column Sketches
echo "📊 Step 1: Creating 2-column sketches..."
./venv/bin/python tools/create_2col_sketches.py \
    --input tests/resources/DU_raw.csv \
    --output DU_output/ \
    --lg_k 18 \
    --skip_columns tripId

echo "✅ Step 1 completed"
echo "   Output: DU_output/DU_raw_2col_sketches_lg_k_18.csv"
echo ""

# Step 2: 2-Column Sketches → 3-Column Sketches
echo "🔄 Step 2: Converting to 3-column sketches..."
./venv/bin/python tools/simple_convert_to_3col.py \
    --input DU_output/DU_raw_2col_sketches_lg_k_18.csv \
    --suffix _3col_sketches_lg_k_18 \
    --mapping DU_feature_mapping.json \
    --output DU_output/ \
    --target tripOutcome

echo "✅ Step 2 completed"
echo "   Output: DU_output/positive_3col_sketches_lg_k_18.csv"
echo "   Output: DU_output/negative_3col_sketches_lg_k_18.csv"
echo "   Output: DU_output/DU_feature_mapping.json"
echo ""

# Step 3: 3-Column Sketches → Trained Tree Model
echo "🌳 Step 3: Training decision tree model..."
./venv/bin/python tools/train_from_3col_sketches.py \
    --lg_k 18 \
    --positive DU_output/positive_3col_sketches_lg_k_18.csv \
    --negative DU_output/negative_3col_sketches_lg_k_18.csv \
    --config configs/du_config.yaml \
    --output DU_output/du_model_lg_k_18

echo "✅ Step 3 completed"
echo "   Output: DU_output/du_model_lg_k_18/3col_sketches_lg_k_18_model_lg_k_18.pkl"
echo "   Output: DU_output/du_model_lg_k_18/3col_sketches_lg_k_18_model_lg_k_18.json"
echo ""

# Step 4: Extract Decision Rules
echo "📋 Step 4: Extracting decision rules for validation..."
./venv/bin/python tools/extract_tree_rules.py \
    DU_output/du_model_lg_k_18/3col_sketches_lg_k_18_model_lg_k_18.json \
    --save_rules DU_output/tree_rules.json \
    --save_sql DU_output/tree_validation_queries.sql \
    --table DU_raw \
    --target_column tripOutcome \
    --quiet

./venv/bin/python tools/quick_tree_test.py \
    DU_output/du_model_lg_k_18/3col_sketches_lg_k_18_model_lg_k_18.json \
    --num_rules 5 \
    --table DU_raw \
    --target tripOutcome > DU_output/quick_test_rules.txt

echo "✅ Step 4 completed"
echo "   Output: DU_output/tree_rules.json"
echo "   Output: DU_output/tree_validation_queries.sql"
echo "   Output: DU_output/quick_test_rules.txt"
echo ""

echo "🎉 DU Pipeline completed successfully!"
echo "All outputs available in DU_output/ directory"

# Alternative: You can also use the comprehensive script:
# ./sample_du_training.sh

# Key Parameters Explained:
# - --lg_k 18: Theta sketch precision parameter (higher = more accurate, more memory)
# - --target tripOutcome: Target column name in the CSV
# - --skip_columns tripId: Columns to exclude from feature creation
# - configs/du_config.yaml: Contains tree hyperparameters (max_depth=6, min_samples_split=100, etc.)
