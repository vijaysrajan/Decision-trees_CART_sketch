# Theta Sketch Decision Tree Tools

This directory contains specialized tools for working with theta sketch-based decision trees.

## Directory Structure

```
tools/
├── sketch_generation/
│   └── create_mushroom_sketch_files.py    # Generate theta sketches from mushroom dataset
└── comparison/
    └── compare_trees_by_lg_k.py           # Compare trees built with different lg_k values
```

## Usage

### Sketch Generation

Generate theta sketches with different lg_k parameters:

```bash
# From project root
cd /path/to/project

# Generate sketches with different precision levels
python tools/sketch_generation/create_mushroom_sketch_files.py --lg_k 11
python tools/sketch_generation/create_mushroom_sketch_files.py --lg_k 16
python tools/sketch_generation/create_mushroom_sketch_files.py --lg_k 8
```

**Output:**
- `tests/fixtures/mushroom_positive_sketches_lg_k_{lg_k}.csv`
- `tests/fixtures/mushroom_negative_sketches_lg_k_{lg_k}.csv`
- `tests/fixtures/mushroom_feature_mapping.json`

### Tree Comparison

Compare decision trees built from different lg_k sketch parameters:

```bash
# Compare trees and save as baselines
python tools/comparison/compare_trees_by_lg_k.py --baseline_lg_k 11 --comparison_lg_k 16 --criterion gini --save_baselines

# Fast comparison using existing baselines
python tools/comparison/compare_trees_by_lg_k.py --baseline_lg_k 11 --comparison_lg_k 16 --criterion gini --use_existing_baselines
```

**Output:**
- `tests/integration/mushroom/baselines/mushroom_baseline_lg_k_{lg_k}_{criterion}_depth_{max_depth}.json`
- `tree_comparison_lg_k_{baseline}_vs_{comparison}_{criterion}_{timestamp}.log`

## Purpose

These tools demonstrate that **"even if one uses very small sketches things work"** by:

1. **Generating sketches** with different precision levels (lg_k values)
2. **Building decision trees** from these sketches
3. **Comparing structural differences** between trees built from different sketch precisions
4. **Saving baselines** for future reference and validation

The lg_k parameter controls sketch precision:
- **Lower lg_k** (8, 11): Less precise, smaller memory footprint
- **Higher lg_k** (16, 20): More precise, larger memory footprint

The comparison tool shows how tree structure changes (or remains stable) across different sketch precision levels.