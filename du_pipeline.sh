#Complete DU Pipeline Commands

#Step 1: Raw CSV → 2-Column Sketches
./venv/bin/python tools/create_2col_sketches.py DU_raw.csv target --lg_k 18 --output DU_output/

#  Output:
#  - DU_output/DU_raw_positive_2col_sketches_lg_k_18.csv
#  - DU_output/DU_raw_negative_2col_sketches_lg_k_18.csv
#  - DU_output/DU_raw_feature_mapping.json


#Step 2: 2-Column Sketches → 3-Column Sketches
./venv/bin/python tools/simple_convert_to_3col.py  DU_output/DU_raw_positive_2col_sketches_lg_k_18.csv  DU_output/DU_raw_negative_2col_sketches_lg_k_18.csv  DU_output/DU_raw_feature_mapping.json --output DU_output/DU_raw_3col_sketches_lg_k_18.csv

#Output:
#  - DU_output/DU_raw_3col_sketches_lg_k_18.csv


#Step 3: 3-Column Sketches → Trained Tree Model
./venv/bin/python tools/train_from_3col_sketches.py \
      DU_output/DU_raw_3col_sketches_lg_k_18.csv \
      --positive 1 \
      --lg_k 18 \
      --criterion gini \
      --max_depth 6 \
      --min_samples_split 100 \
      --min_samples_leaf 50 \
      --output DU_output/test_model_lg_k_18

#Output:
#  - DU_output/test_model_lg_k_18/3col_sketches_model_lg_k_18.pkl (trained model)
#  - DU_output/test_model_lg_k_18/3col_sketches_model_lg_k_18.json (tree structure)

#Complete One-Liner Pipeline

#If you want to run all steps in sequence:

## Step 1: Raw CSV → 2-column sketches
#./venv/bin/python tools/create_2col_sketches.py DU_raw.csv target --lg_k 18 --output DU_output/ && \

## Step 2: 2-column → 3-column sketches  
#./venv/bin/python tools/simple_convert_to_3col.py \
#      DU_output/DU_raw_positive_2col_sketches_lg_k_18.csv \
#      DU_output/DU_raw_negative_2col_sketches_lg_k_18.csv \
#      DU_output/DU_raw_feature_mapping.json \
#      --output DU_output/DU_raw_3col_sketches_lg_k_18.csv && \

## Step 3: 3-column sketches → trained model
#./venv/bin/python tools/train_from_3col_sketches.py \
#      DU_output/DU_raw_3col_sketches_lg_k_18.csv \
#      --positive 1 \
#      --lg_k 18 \
#      --criterion gini \
#      --max_depth 6 \
#      --min_samples_split 100 \
#      --min_samples_leaf 50 \
#      --output DU_output/test_model_lg_k_18

#Alternative: Using the Existing Script
#I noticed there's a sample_du_training.sh script that should contain this pipeline. You can also run:
#./sample_du_training.sh
#Key Parameters Explained
# - --lg_k 18: Theta sketch precision parameter (higher = more accurate, more memory)
#  - --positive 1: Target value for positive class in the CSV
#  - --criterion gini: Split criterion (options: gini, entropy, gain_ratio, etc.)
#  - --max_depth 6: Maximum tree depth
#  - --min_samples_split 100: Minimum samples to split a node
#  - --min_samples_leaf 50: Minimum samples in leaf nodes
