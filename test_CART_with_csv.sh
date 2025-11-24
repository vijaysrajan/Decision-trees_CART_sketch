#!/bin/bash
# Test script for CART decision tree with CSV datasets
#
# This script demonstrates how to run the theta sketch decision tree
# on any binary classification CSV dataset using the intersection tree builder.

echo "🧪 Testing CART Decision Tree with CSV Dataset"
echo "=============================================="

./venv/bin/python run_binary_classification.py \
      ./tests/resources/binary_classification_data.csv target \
      --lg_k 16 \
      --max_depth 5 \
      --criterion gini \
      --tree_builder intersection \
      --verbose 1 \
      --sample_size 10000
