#!/usr/bin/env python3
"""
Generic 2-Column Sketch Generator

Creates 2-column theta sketches from any CSV dataset for conversion to 3-column format.
This script generates sketches that can then be converted using convert_2col_to_3col_sketches.py

Usage:
    python tools/create_2col_sketches.py dataset.csv target_column [--lg_k LG_K] [--id_column ID_COL]

Examples:
    # Generate mushroom sketches
    python tools/create_2col_sketches.py tests/resources/agaricus-lepiota.csv class --lg_k 12

    # Generate DU sketches with trip IDs
    python tools/create_2col_sketches.py tests/resources/DU_raw.csv tripOutcome --lg_k 16 --id_column tripId

    # Generate any dataset sketches
    python tools/create_2col_sketches.py data.csv target_col --lg_k 14
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
import json
import hashlib
import base64

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from datasketches import update_theta_sketch, theta_intersection, theta_union
    DATASKETCHES_AVAILABLE = True
except ImportError:
    DATASKETCHES_AVAILABLE = False
    raise ImportError("Apache DataSketches library required. Install with: pip install datasketches")

from tests.test_binary_classification_sketches import ThetaSketchWrapper

# Configuration
DEFAULT_LG_K = 16


def hash_identifier(identifier: str) -> int:
    """
    Create a consistent hash of identifier for sketch updates.

    Parameters
    ----------
    identifier : str
        Identifier string (row ID, trip ID, etc.)

    Returns
    -------
    int
        Hash value suitable for sketch updates
    """
    return int(hashlib.md5(str(identifier).encode()).hexdigest()[:8], 16)


def create_decision_tree_sketches(df: pd.DataFrame, target_column: str, feature_columns: List[str],
                                 id_column: str = None, lg_k: int = DEFAULT_LG_K) -> Dict[str, Dict[str, Any]]:
    """
    Create theta sketches for any dataset optimized for decision tree classification.

    Each feature=value sketch contains identifiers that have that feature value,
    allowing the decision tree to split based on feature presence while
    maintaining identifier-level cardinality estimates.

    Parameters
    ----------
    df : DataFrame
        Dataset with features and target column
    target_column : str
        Name of the target column
    feature_columns : List[str]
        List of feature column names
    id_column : str, optional
        Name of identifier column (uses row index if None)
    lg_k : int, default=DEFAULT_LG_K
        Log-base-2 of nominal entries (k = 2^lg_k)

    Returns
    -------
    sketch_data : dict
        Format: {'positive': {...}, 'negative': {...}}
        Each contains 'total' sketch and feature=value sketches
        All sketches contain hashed identifiers for cardinality estimation
    """
    print(f"📊 Creating decision tree sketches with lg_k={lg_k} (size={2**lg_k})")

    # Validate dataset structure
    required_columns = [target_column] + feature_columns
    if id_column:
        required_columns.append(id_column)

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Identify class values
    class_values = sorted(df[target_column].unique())
    if len(class_values) != 2:
        raise ValueError(f"Expected exactly 2 class values, found {len(class_values)}: {class_values}")

    negative_class, positive_class = class_values
    positive_df = df[df[target_column] == positive_class].copy()
    negative_df = df[df[target_column] == negative_class].copy()

    print(f"📊 Dataset Analysis:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Positive class ('{positive_class}'): {len(positive_df):,} samples ({len(positive_df)/len(df)*100:.1f}%)")
    print(f"  Negative class ('{negative_class}'): {len(negative_df):,} samples ({len(negative_df)/len(df)*100:.1f}%)")
    print(f"  Features: {len(feature_columns)}")

    def create_class_sketches(class_df: pd.DataFrame, class_name: str) -> Dict[str, Any]:
        """Create sketches for one class with identifier hashing."""
        print(f"  Processing {class_name} class...")

        # Create total sketch for all samples in this class
        total_sketch = ThetaSketchWrapper(update_theta_sketch(lg_k))

        # Add all identifiers to total sketch
        if id_column:
            for identifier in class_df[id_column]:
                total_sketch.update(hash_identifier(str(identifier)))
        else:
            for idx in class_df.index:
                total_sketch.update(hash_identifier(str(idx)))

        print(f"    Total sketch: {total_sketch.get_estimate():.0f} estimated samples")

        # Create feature=value sketches
        feature_sketches = {}

        for feature in feature_columns:
            feature_values = sorted(class_df[feature].unique())
            print(f"    Feature '{feature}': {len(feature_values)} unique values")

            for value in feature_values:
                feature_key = f"{feature}={value}"
                feature_sketch = ThetaSketchWrapper(update_theta_sketch(lg_k))

                # Add identifiers that have this feature=value combination
                matching_samples = class_df[class_df[feature] == value]

                if id_column:
                    for identifier in matching_samples[id_column]:
                        feature_sketch.update(hash_identifier(str(identifier)))
                else:
                    for idx in matching_samples.index:
                        feature_sketch.update(hash_identifier(str(idx)))

                feature_sketches[feature_key] = feature_sketch

                # Log significant features
                estimate = feature_sketch.get_estimate()
                if estimate > len(matching_samples) * 0.1:  # Log if >10% of expected
                    print(f"      {feature_key}: {estimate:.0f} estimated samples ({len(matching_samples)} actual)")

        return {
            'total': total_sketch,
            **feature_sketches
        }

    # Create sketches for both classes
    sketch_data = {
        'positive': create_class_sketches(positive_df, positive_class),
        'negative': create_class_sketches(negative_df, negative_class)
    }

    # Summary statistics
    total_features = len([k for k in sketch_data['positive'].keys() if k != 'total'])
    print(f"✅ Created {total_features} feature=value sketches per class")
    print(f"🎯 Ready for decision tree training!")

    return sketch_data


def create_apriori_sketches(df: pd.DataFrame, feature_columns: List[str], id_column: str = None, lg_k: int = DEFAULT_LG_K) -> Dict[str, Any]:
    """
    Create theta sketches for any dataset optimized for apriori frequent itemset mining.

    Each sketch represents an itemset (combination of feature values) and contains
    identifiers that contain that itemset, enabling support calculation for frequent
    itemset mining algorithms.

    Parameters
    ----------
    df : DataFrame
        Dataset with features
    feature_columns : List[str]
        List of feature column names
    id_column : str, optional
        Name of identifier column (uses row index if None)
    lg_k : int, default=DEFAULT_LG_K
        Log-base-2 of nominal entries (k = 2^lg_k)

    Returns
    -------
    itemset_sketches : dict
        Format: {'1-itemsets': {...}, '2-itemsets': {...}, 'transactions': sketch}
        Contains sketches for frequent itemset mining
    """
    print(f"🛒 Creating apriori sketches with lg_k={lg_k} (size={2**lg_k})")

    # Create transaction sketch (all unique identifiers)
    transaction_sketch = ThetaSketchWrapper(update_theta_sketch(lg_k))
    if id_column:
        for identifier in df[id_column]:
            transaction_sketch.update(hash_identifier(str(identifier)))
    else:
        for idx in df.index:
            transaction_sketch.update(hash_identifier(str(idx)))

    total_transactions = transaction_sketch.get_estimate()
    print(f"📦 Total transactions: {total_transactions:.0f} samples")

    # Create 1-itemsets (individual feature values)
    itemset_1_sketches = {}

    for feature in feature_columns:
        feature_values = df[feature].unique()
        print(f"  Feature '{feature}': {len(feature_values)} unique values")

        for value in feature_values:
            itemset_key = f"{feature}={value}"
            itemset_sketch = ThetaSketchWrapper(update_theta_sketch(lg_k))

            # Add identifiers that contain this item
            matching_samples = df[df[feature] == value]
            if id_column:
                for identifier in matching_samples[id_column]:
                    itemset_sketch.update(hash_identifier(str(identifier)))
            else:
                for idx in matching_samples.index:
                    itemset_sketch.update(hash_identifier(str(idx)))

            support = itemset_sketch.get_estimate() / total_transactions
            itemset_1_sketches[itemset_key] = {
                'sketch': itemset_sketch,
                'support': support,
                'count': len(matching_samples)
            }

    # Filter frequent 1-itemsets (support > 1%)
    min_support = 0.01
    frequent_1_itemsets = {k: v for k, v in itemset_1_sketches.items() if v['support'] > min_support}

    print(f"📈 Frequent 1-itemsets: {len(frequent_1_itemsets)} (support > {min_support:.1%})")

    # Create 2-itemsets from frequent 1-itemsets
    itemset_2_sketches = {}
    frequent_items = list(frequent_1_itemsets.keys())

    print(f"🔗 Generating 2-itemsets from {len(frequent_items)} frequent items...")

    for i in range(len(frequent_items)):
        for j in range(i + 1, len(frequent_items)):
            item1, item2 = frequent_items[i], frequent_items[j]

            # Skip if same feature (can't have source=ios AND source=android)
            feature1 = item1.split('=')[0]
            feature2 = item2.split('=')[0]
            if feature1 == feature2:
                continue

            # Create 2-itemset sketch using intersection
            sketch1 = frequent_1_itemsets[item1]['sketch']
            sketch2 = frequent_1_itemsets[item2]['sketch']

            intersection_sketch = sketch1.intersection(sketch2)
            support = intersection_sketch.get_estimate() / total_transactions

            if support > min_support:  # Only keep frequent 2-itemsets
                itemset_key = f"{item1},{item2}"
                itemset_2_sketches[itemset_key] = {
                    'sketch': ThetaSketchWrapper(intersection_sketch),
                    'support': support,
                    'items': [item1, item2]
                }

    print(f"📈 Frequent 2-itemsets: {len(itemset_2_sketches)} (support > {min_support:.1%})")

    return {
        'transactions': transaction_sketch,
        '1-itemsets': {k: v['sketch'] for k, v in frequent_1_itemsets.items()},
        '2-itemsets': {k: v['sketch'] for k, v in itemset_2_sketches.items()},
        'support_stats': {
            'total_transactions': total_transactions,
            'min_support': min_support,
            'frequent_1_count': len(frequent_1_itemsets),
            'frequent_2_count': len(itemset_2_sketches)
        }
    }


def create_feature_mapping(sketch_data: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """
    Create feature mapping for decision tree classifier from sketch data.

    Parameters
    ----------
    sketch_data : dict
        Output from create_decision_tree_sketches()

    Returns
    -------
    feature_mapping : dict
        Maps feature=value strings to column indices for binary matrix
    """
    # Get all feature names (excluding 'total')
    feature_names = [k for k in sketch_data['positive'].keys() if k != 'total']
    feature_names.sort()  # Consistent ordering

    # Create mapping
    feature_mapping = {name: idx for idx, name in enumerate(feature_names)}

    print(f"🗺️ Created feature mapping for {len(feature_mapping)} features")
    return feature_mapping


def save_sketches_to_2col_csv(sketch_data: Dict[str, Dict[str, Any]], output_dir: str,
                             dataset_name: str, lg_k: int) -> None:
    """Save sketch data to 2-column CSV files for conversion pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"💾 Saving 2-column sketches to {output_path}/")

    for class_name, class_sketches in sketch_data.items():
        # Create CSV data for 2-column format
        csv_data = []

        for feature_name, sketch in class_sketches.items():
            if feature_name != 'total':  # Skip total sketch for 2-column format
                # Serialize sketch to base64 (need to compact first)
                compact_sketch = sketch._sketch.compact()
                sketch_bytes = compact_sketch.serialize()
                sketch_base64 = base64.b64encode(sketch_bytes).decode('utf-8')

                csv_data.append({
                    'feature': feature_name,
                    'sketch': sketch_base64
                })

        # Save to CSV file
        csv_filename = f"{dataset_name}_{class_name}_2col_sketches_lg_k_{lg_k}.csv"
        csv_filepath = output_path / csv_filename

        df_out = pd.DataFrame(csv_data)
        df_out.to_csv(csv_filepath, index=False)

        print(f"  📄 {csv_filename} ({len(csv_data)} features)")

    # Also save feature mapping as JSON
    feature_mapping = create_feature_mapping(sketch_data)
    mapping_filename = f"{dataset_name}_feature_mapping.json"
    mapping_filepath = output_path / mapping_filename

    with open(mapping_filepath, 'w') as f:
        json.dump(feature_mapping, f, indent=2)

    print(f"  🗺️ {mapping_filename}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Generate 2-column theta sketches from any CSV dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mushroom dataset
  python tools/create_2col_sketches.py tests/resources/agaricus-lepiota.csv class --lg_k 12

  # DU dataset with ID column
  python tools/create_2col_sketches.py tests/resources/DU_raw.csv tripOutcome --lg_k 16 --id_column tripId

  # Any dataset with auto-detection
  python tools/create_2col_sketches.py data.csv target_column --lg_k 14
        """)

    # Positional arguments
    parser.add_argument('dataset_path', type=str,
                       help='Path to CSV dataset file')
    parser.add_argument('target_column', type=str,
                       help='Name of target column (must be binary classification)')

    # Optional arguments
    parser.add_argument('--lg_k', type=int, default=DEFAULT_LG_K,
                       help=f'Log-base-2 of sketch size (default: {DEFAULT_LG_K})')
    parser.add_argument('--id_column', type=str, default=None,
                       help='Name of ID column (uses row index if not specified)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for sketch files (default: auto-generated)')
    parser.add_argument('--mode', choices=['decision_tree', 'apriori', 'both'], default='decision_tree',
                       help='Type of sketches to generate (default: decision_tree)')

    args = parser.parse_args()

    print("📊 Generic 2-Column Sketch Generator")
    print("=" * 50)

    # Load dataset
    try:
        df = pd.read_csv(args.dataset_path)
        print(f"📁 Loaded dataset: {args.dataset_path}")
        print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    except FileNotFoundError:
        print(f"❌ Error: Dataset not found at {args.dataset_path}")
        return 1
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return 1

    # Validate target column
    if args.target_column not in df.columns:
        print(f"❌ Error: Target column '{args.target_column}' not found")
        print(f"Available columns: {list(df.columns)}")
        return 1

    # Auto-detect feature columns (exclude target and ID columns)
    exclude_columns = [args.target_column]
    if args.id_column:
        if args.id_column not in df.columns:
            print(f"❌ Error: ID column '{args.id_column}' not found")
            return 1
        exclude_columns.append(args.id_column)

    feature_columns = [col for col in df.columns if col not in exclude_columns]
    print(f"🔍 Auto-detected {len(feature_columns)} feature columns")
    print(f"   Features: {feature_columns[:5]}{'...' if len(feature_columns) > 5 else ''}")

    # Generate output directory name
    if args.output_dir is None:
        dataset_name = Path(args.dataset_path).stem.lower().replace('-', '_')
        args.output_dir = f"{dataset_name}_sketches"

    print(f"📂 Output directory: {args.output_dir}")

    try:
        if args.mode in ['decision_tree', 'both']:
            print(f"\n🌳 Generating decision tree sketches...")
            dt_sketches = create_decision_tree_sketches(
                df=df,
                target_column=args.target_column,
                feature_columns=feature_columns,
                id_column=args.id_column,
                lg_k=args.lg_k
            )

            dataset_name = Path(args.dataset_path).stem.lower().replace('-', '_')
            save_sketches_to_2col_csv(dt_sketches, args.output_dir, dataset_name, args.lg_k)

            feature_mapping = create_feature_mapping(dt_sketches)
            print(f"✅ Decision tree sketches ready!")
            print(f"   Features: {len(feature_mapping)} binary features")
            print(f"   Next: python tools/convert_2col_to_3col_sketches.py {args.output_dir}/*.csv")

        if args.mode in ['apriori', 'both']:
            print(f"\n🛒 Generating apriori sketches...")
            apriori_sketches = create_apriori_sketches(
                df=df,
                feature_columns=feature_columns,
                id_column=args.id_column,
                lg_k=args.lg_k
            )

            print(f"✅ Apriori sketches ready!")
            print(f"   1-itemsets: {len(apriori_sketches['1-itemsets'])}")
            print(f"   2-itemsets: {len(apriori_sketches['2-itemsets'])}")

        print(f"\n🎉 Sketch generation complete!")

    except Exception as e:
        print(f"❌ Error during sketch generation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
