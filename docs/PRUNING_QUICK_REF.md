# 🌳 Pruning Quick Reference

## 📋 Command Line Quick Reference

```bash
# Basic usage with pruning
./venv/bin/python run_binary_classification.py <dataset.csv> <target> --pruning <method>

# All pruning parameters
--pruning {none,validation,cost_complexity,reduced_error,min_impurity}
--min_impurity_decrease <float>    # For min_impurity method
--validation_fraction <float>      # For validation/reduced_error methods
```

## ⚡ Quick Commands

### 🔥 Most Recommended (Cost-Complexity)
```bash
./venv/bin/python run_binary_classification.py data.csv target \
  --pruning cost_complexity --lg_k 14 --max_depth 8
```

### 🎯 High Accuracy (Validation-Based)
```bash
./venv/bin/python run_binary_classification.py data.csv target \
  --pruning validation --validation_fraction 0.25 --lg_k 14
```

### 🛡️ Conservative (Minimum Impurity)
```bash
./venv/bin/python run_binary_classification.py data.csv target \
  --pruning min_impurity --min_impurity_decrease 0.01
```

## 📊 Method Comparison

| Method | Speed | Reduction | Best For |
|--------|-------|-----------|----------|
| `cost_complexity` | Fast | High (30-60%) | **General use** ⭐ |
| `validation` | Medium | Variable | High accuracy needs |
| `reduced_error` | Medium | Variable | Accuracy preservation |
| `min_impurity` | Fast | Low (5-20%) | Conservative pruning |
| `none` | Fastest | 0% | Small/clean datasets |

## 🎯 Dataset Size Guide

- **< 500 samples**: `--pruning min_impurity`
- **500-2000 samples**: `--pruning cost_complexity`
- **> 2000 samples**: `--pruning validation`

## 🛠️ Parameter Tuning

### min_impurity_decrease
- `0.001`: Very conservative
- `0.01`: Moderate (recommended)
- `0.05`: Aggressive

### validation_fraction
- `0.1`: Small validation set
- `0.2`: Standard (recommended)
- `0.3`: Large validation set

## 📈 Interpreting Results

```bash
# Example output
Applying cost_complexity pruning...
Pruning complete: 8 nodes removed
Compression ratio: 0.529
```

**Compression ratio**: Lower = more pruning
- `0.3-0.6`: Good pruning (30-70% reduction)
- `0.7-0.9`: Light pruning (10-30% reduction)
- `1.0`: No pruning occurred

## 🚨 Common Issues

**No pruning occurs**: Try `--min_impurity_decrease 0.05` or different method
**Over-pruning**: Reduce thresholds or use `reduced_error`
**JSON errors**: Fixed in current version ✅

## 📚 More Help

- Full guide: `PRUNING_GUIDE.md`
- Examples: `EXAMPLES.md`
- CLI help: `./venv/bin/python run_binary_classification.py --help`