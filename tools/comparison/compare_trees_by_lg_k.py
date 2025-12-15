#!/usr/bin/env python3
"""
Tree Structural Difference Comparison Tool

This script compares decision trees built with different lg_k parameters
and provides both structural summaries and detailed line-by-line differences.

Usage:
    python compare_trees_by_lg_k.py --baseline_lg_k 11 --comparison_lg_k 16 --criterion gini
    python compare_trees_by_lg_k.py --baseline_lg_k 11 --comparison_lg_k 8 --criterion entropy

The script will:
1. Load sketch files with specified lg_k parameters
2. Build trees using the theta sketch classifier
3. Compare tree structures recursively
4. Generate detailed comparison logs with visual indicators
5. Provide summary statistics of differences
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to Python path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import the classifier and related modules
try:
    from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier
    from theta_sketch_tree.tree_structure import TreeNode
    sys.path.insert(0, str(script_dir.parent / "sketch_generation"))
    from create_mushroom_sketch_files import load_sketches_from_csv, load_mushroom_dataset, create_binary_features
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)


def serialize_tree_node(node, feature_mapping, depth=0):
    """
    Serialize a tree node to JSON format matching the baseline structure.
    """
    # Reverse mapping for feature names
    idx_to_feature = {idx: name for name, idx in feature_mapping.items()}

    node_data = {
        "depth": depth,
        "n_samples": float(node.n_samples),
        "class_counts": [float(count) for count in node.class_counts],
        "impurity": round(node.impurity, 4)
    }

    if node.feature_idx is not None:  # Split node
        feature_name = idx_to_feature.get(node.feature_idx, f"feature_{node.feature_idx}")

        node_data.update({
            "is_leaf": False,
            "type": "split",
            "feature_name": feature_name,
            "feature_idx": node.feature_idx,
            "split_condition": f"{feature_name} == 1",
            "left_condition": f"{feature_name} == 0 (FALSE/NOT)",
            "right_condition": f"{feature_name} == 1 (TRUE)"
        })

        # Recursively serialize children
        if hasattr(node, 'left') and node.left is not None:
            node_data["left"] = serialize_tree_node(node.left, feature_mapping, depth + 1)

        if hasattr(node, 'right') and node.right is not None:
            node_data["right"] = serialize_tree_node(node.right, feature_mapping, depth + 1)

    else:  # Leaf node
        node_data.update({
            "is_leaf": True,
            "type": "leaf",
            "prediction": int(node.prediction),
            "class_probabilities": [float(prob) for prob in getattr(node, 'probabilities', [0.0, 1.0])]
        })

    return node_data


def calculate_tree_depth(node):
    """Calculate the maximum depth of the tree."""
    if node is None:
        return 0

    if hasattr(node, 'left') and hasattr(node, 'right'):
        left_depth = calculate_tree_depth(getattr(node, 'left', None)) if hasattr(node, 'left') else 0
        right_depth = calculate_tree_depth(getattr(node, 'right', None)) if hasattr(node, 'right') else 0
        return max(left_depth, right_depth) + 1
    else:
        return 1


def count_leaves(node):
    """Count the number of leaf nodes in the tree."""
    if node is None:
        return 0

    # Check if it's a leaf node (no feature_idx means it's a leaf)
    if node.feature_idx is None:
        return 1

    left_leaves = count_leaves(getattr(node, 'left', None)) if hasattr(node, 'left') else 0
    right_leaves = count_leaves(getattr(node, 'right', None)) if hasattr(node, 'right') else 0

    return left_leaves + right_leaves


def serialize_classifier_to_baseline_format(classifier, feature_mapping, lg_k, criterion, max_depth, config_name=None):
    """
    Serialize a classifier to the baseline JSON format.
    """
    if config_name is None:
        config_name = f"lg_k_{lg_k}_{criterion}_depth_{max_depth}"

    baseline_data = {
        config_name: {
            "description": f"lg_k={lg_k}, criterion={criterion}, max_depth={max_depth}",
            "metadata": {
                "lg_k": lg_k,
                "criterion": criterion,
                "max_depth": max_depth,
                "tree_depth": calculate_tree_depth(classifier.tree_),
                "n_leaves": count_leaves(classifier.tree_),
                "n_features": len(feature_mapping),
                "timestamp": datetime.now().isoformat()
            },
            "tree_structure": serialize_tree_node(classifier.tree_, feature_mapping)
        }
    }

    # Add feature importances if available
    if hasattr(classifier, 'feature_importances_') and classifier.feature_importances_ is not None:
        importances_dict = {}
        idx_to_feature = {idx: name for name, idx in feature_mapping.items()}

        for idx, importance in enumerate(classifier.feature_importances_):
            if importance > 0.0:  # Only include non-zero importances
                feature_name = idx_to_feature.get(idx, f"feature_{idx}")
                importances_dict[feature_name] = float(importance)

        baseline_data[config_name]["feature_importances"] = importances_dict

    return baseline_data


def save_baseline_json(classifier, feature_mapping, lg_k, criterion, max_depth, output_dir=".", config_name=None):
    """
    Save classifier as baseline JSON file.
    """
    baseline_data = serialize_classifier_to_baseline_format(
        classifier, feature_mapping, lg_k, criterion, max_depth, config_name
    )

    # Generate filename
    if config_name is None:
        config_name = f"lg_k_{lg_k}_{criterion}_depth_{max_depth}"

    baseline_filename = f"{output_dir}/mushroom_baseline_lg_k_{lg_k}_{criterion}_depth_{max_depth}.json"

    # Check if file exists and merge if it does
    if Path(baseline_filename).exists():
        with open(baseline_filename, 'r') as f:
            existing_data = json.load(f)
        existing_data.update(baseline_data)
        baseline_data = existing_data

    # Save to file
    with open(baseline_filename, 'w') as f:
        json.dump(baseline_data, f, indent=2)

    return baseline_filename


def load_baseline_tree(baseline_file, config_name):
    """
    Load tree structure from baseline JSON file.
    """
    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)

    if config_name not in baseline_data:
        available_configs = list(baseline_data.keys())
        raise KeyError(f"Configuration '{config_name}' not found. Available: {available_configs}")

    return baseline_data[config_name]


def json_to_tree_node(json_node):
    """
    Convert JSON tree structure back to TreeNode object for comparison.
    """
    # Create TreeNode with required parameters
    node = TreeNode(
        depth=json_node["depth"],
        n_samples=json_node["n_samples"],
        class_counts=np.array(json_node["class_counts"]),
        impurity=json_node["impurity"]
    )

    if json_node["type"] == "split":
        # Split node - set split information
        left_child = json_to_tree_node(json_node["left"]) if "left" in json_node else None
        right_child = json_to_tree_node(json_node["right"]) if "right" in json_node else None

        if left_child and right_child:
            node.set_split(
                feature_idx=json_node["feature_idx"],
                feature_name=json_node["feature_name"],
                left_child=left_child,
                right_child=right_child
            )
    else:
        # Leaf node
        node.make_leaf()
        # Override the computed prediction if specified
        if "prediction" in json_node:
            node.prediction = json_node["prediction"]
        if "class_probabilities" in json_node:
            node.probabilities = np.array(json_node["class_probabilities"])

    return node


class TreeComparator:
    """
    Compares two decision trees and generates detailed difference reports.
    """

    def __init__(self, baseline_tree, comparison_tree, feature_mapping, tolerance=1e-3):
        self.baseline_tree = baseline_tree
        self.comparison_tree = comparison_tree
        self.feature_mapping = feature_mapping
        self.tolerance = tolerance

        # Reverse mapping for feature names
        self.idx_to_feature = {idx: name for name, idx in feature_mapping.items()}

        # Statistics
        self.differences = {
            'sample_count_diffs': 0,
            'class_count_diffs': 0,
            'impurity_diffs': 0,
            'feature_index_diffs': 0,
            'structure_diffs': 0,
            'total_nodes': 0
        }

        self.comparison_log = []

    def format_feature_name(self, feature_idx):
        """Get feature name from index."""
        return self.idx_to_feature.get(feature_idx, f"feature_{feature_idx}")

    def log_message(self, message):
        """Add message to comparison log."""
        self.comparison_log.append(message)

    def compare_nodes_recursively(self, baseline_node, comparison_node, path="root", depth=0):
        """
        Recursively compare two tree nodes and log differences.
        """
        self.differences['total_nodes'] += 1

        # Node header
        indent = "    " * depth
        separator = "─" * 80
        self.log_message(f"{separator}")
        self.log_message(f"🌳 Comparing node at {path}:")

        # Basic node metadata
        baseline_type = "SPLIT" if baseline_node.feature_idx is not None else "LEAF"
        comparison_type = "SPLIT" if comparison_node.feature_idx is not None else "LEAF"

        self.log_message(f"   Type: {baseline_type} vs {comparison_type}, Depth: {depth}")

        if baseline_type != comparison_type:
            self.log_message(f"   ❌ Node type mismatch: baseline={baseline_type}, comparison={comparison_type}")
            self.differences['structure_diffs'] += 1
            return
        else:
            self.log_message(f"   ✅ Node types match")

        # Sample count comparison
        baseline_samples = baseline_node.n_samples
        comparison_samples = comparison_node.n_samples
        sample_diff = abs(baseline_samples - comparison_samples)

        self.log_message(f"   📊 Sample counts: baseline={baseline_samples}, comparison={comparison_samples}, diff={sample_diff:.3f}")

        if sample_diff > self.tolerance:
            self.log_message(f"   ❌ Sample count difference: {sample_diff:.3f}")
            self.differences['sample_count_diffs'] += 1
        else:
            self.log_message(f"   ✅ Sample counts match")

        # Class count comparison
        baseline_class_counts = np.array(baseline_node.class_counts)
        comparison_class_counts = np.array(comparison_node.class_counts)

        self.log_message(f"   📈 Class counts: baseline={baseline_class_counts}, comparison={comparison_class_counts}")

        for i, (b_count, c_count) in enumerate(zip(baseline_class_counts, comparison_class_counts)):
            diff = abs(b_count - c_count)
            if diff > self.tolerance:
                self.log_message(f"   ❌ Class {i} difference: baseline={b_count:.3f}, comparison={c_count:.3f}, diff={diff:.3f}")
                self.differences['class_count_diffs'] += 1
            else:
                self.log_message(f"   ✅ Class {i} counts match")

        # Impurity comparison
        baseline_impurity = baseline_node.impurity
        comparison_impurity = comparison_node.impurity
        impurity_diff = abs(baseline_impurity - comparison_impurity)

        self.log_message(f"   🎯 Impurity: baseline={baseline_impurity:.8f}, comparison={comparison_impurity:.8f}, diff={impurity_diff:.2e}")

        if impurity_diff > self.tolerance:
            self.log_message(f"   ❌ Impurity difference: {impurity_diff:.2e}")
            self.differences['impurity_diffs'] += 1
        else:
            self.log_message(f"   ✅ Impurity matches")

        # For split nodes, compare split information
        if baseline_type == "SPLIT":
            baseline_feature = baseline_node.feature_idx
            comparison_feature = comparison_node.feature_idx

            baseline_feature_name = self.format_feature_name(baseline_feature)
            comparison_feature_name = self.format_feature_name(comparison_feature)

            self.log_message(f"   🔀 Split features: baseline={baseline_feature_name} (idx={baseline_feature}), comparison={comparison_feature_name} (idx={comparison_feature})")

            if baseline_feature != comparison_feature:
                self.log_message(f"   ❌ Feature index mismatch: baseline={baseline_feature}, comparison={comparison_feature}")
                self.differences['feature_index_diffs'] += 1
            else:
                self.log_message(f"   ✅ Split features match")

            # Recursively compare children
            if hasattr(baseline_node, 'left') and hasattr(comparison_node, 'left'):
                if baseline_node.left and comparison_node.left:
                    self.log_message(f"   ✅ Both left children present")
                    self.compare_nodes_recursively(baseline_node.left, comparison_node.left, f"{path}/left", depth + 1)
                elif baseline_node.left or comparison_node.left:
                    self.log_message(f"   ❌ Left child presence mismatch")
                    self.differences['structure_diffs'] += 1

            if hasattr(baseline_node, 'right') and hasattr(comparison_node, 'right'):
                if baseline_node.right and comparison_node.right:
                    self.log_message(f"   ✅ Both right children present")
                    self.compare_nodes_recursively(baseline_node.right, comparison_node.right, f"{path}/right", depth + 1)
                elif baseline_node.right or comparison_node.right:
                    self.log_message(f"   ❌ Right child presence mismatch")
                    self.differences['structure_diffs'] += 1

        else:  # LEAF node
            baseline_prediction = baseline_node.prediction
            comparison_prediction = comparison_node.prediction

            self.log_message(f"   🍃 Leaf predictions: baseline={baseline_prediction}, comparison={comparison_prediction}")

            if baseline_prediction != comparison_prediction:
                self.log_message(f"   ❌ Prediction mismatch")
                self.differences['structure_diffs'] += 1
            else:
                self.log_message(f"   ✅ Predictions match")

            # Compare probabilities if available
            if hasattr(baseline_node, 'probabilities') and hasattr(comparison_node, 'probabilities'):
                baseline_probs = baseline_node.probabilities
                comparison_probs = comparison_node.probabilities
                self.log_message(f"   📊 Probabilities: baseline={baseline_probs}, comparison={comparison_probs}")

    def generate_summary(self, baseline_lg_k, comparison_lg_k, criterion):
        """Generate comparison summary."""
        total_diffs = sum(self.differences.values()) - self.differences['total_nodes']

        summary = [
            f"====================================================================================================",
            f"TREE COMPARISON SUMMARY",
            f"Baseline lg_k: {baseline_lg_k}, Comparison lg_k: {comparison_lg_k}, Criterion: {criterion}",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"====================================================================================================",
            f"",
            f"📊 COMPARISON STATISTICS:",
            f"   Total nodes compared: {self.differences['total_nodes']}",
            f"   Sample count differences: {self.differences['sample_count_diffs']}",
            f"   Class count differences: {self.differences['class_count_diffs']}",
            f"   Impurity differences: {self.differences['impurity_diffs']}",
            f"   Feature index differences: {self.differences['feature_index_diffs']}",
            f"   Structure differences: {self.differences['structure_diffs']}",
            f"   Total differences found: {total_diffs}",
            f"",
        ]

        if total_diffs == 0:
            summary.append(f"✅ RESULT: Trees are identical within tolerance ({self.tolerance})")
        else:
            summary.append(f"❌ RESULT: {total_diffs} differences found between trees")

        summary.extend([
            f"",
            f"====================================================================================================",
            f"DETAILED NODE-BY-NODE COMPARISON",
            f"====================================================================================================",
            f""
        ])

        return summary

    def save_comparison_log(self, output_file, baseline_lg_k, comparison_lg_k, criterion):
        """Save detailed comparison log to file."""
        summary = self.generate_summary(baseline_lg_k, comparison_lg_k, criterion)
        full_log = summary + self.comparison_log

        # Add final separator
        full_log.extend([
            f"",
            f"====================================================================================================",
            f"COMPARISON COMPLETED",
            f"Trees compared: lg_k {baseline_lg_k} vs lg_k {comparison_lg_k}",
            f"Output saved to: {output_file}",
            f"===================================================================================================="
        ])

        with open(output_file, 'w') as f:
            for line in full_log:
                f.write(line + '\n')

        return full_log


def load_data_and_build_tree(lg_k, criterion='gini', max_depth=5):
    """
    Load sketch data for specified lg_k and build decision tree.
    """
    print(f"📁 Loading sketch data for lg_k={lg_k}...")

    # Get project root for path resolution
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    # Load sketches
    fixtures_dir = project_root / "tests/fixtures"
    positive_file = fixtures_dir / f"mushroom_positive_sketches_lg_k_{lg_k}.csv"
    negative_file = fixtures_dir / f"mushroom_negative_sketches_lg_k_{lg_k}.csv"

    if not positive_file.exists() or not negative_file.exists():
        raise FileNotFoundError(f"Sketch files not found for lg_k={lg_k}. Run tools/sketch_generation/create_mushroom_sketch_files.py --lg_k {lg_k} first.")

    sketch_data = load_sketches_from_csv(str(positive_file), str(negative_file), lg_k)

    # Load feature mapping
    feature_mapping_file = fixtures_dir / "mushroom_feature_mapping.json"
    with open(feature_mapping_file, 'r') as f:
        feature_mapping = json.load(f)

    print(f"   Positive sketches: {len(sketch_data['positive'])} features")
    print(f"   Negative sketches: {len(sketch_data['negative'])} features")
    print(f"   Feature mapping: {len(feature_mapping)} features")

    # Build classifier
    print(f"🌳 Building tree with criterion={criterion}, max_depth={max_depth}...")
    classifier = ThetaSketchDecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        random_state=42
    )

    # Fit with sketch data
    classifier.fit(sketch_data, feature_mapping)

    print(f"   Tree depth: {calculate_tree_depth(classifier.tree_)}")
    print(f"   Number of leaves: {count_leaves(classifier.tree_)}")
    print(f"   Tree fitted: {classifier._is_fitted}")

    return classifier, feature_mapping


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(
        description="Compare decision trees built with different lg_k parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_trees_by_lg_k.py --baseline_lg_k 11 --comparison_lg_k 16 --criterion gini
  python compare_trees_by_lg_k.py --baseline_lg_k 11 --comparison_lg_k 8 --criterion entropy
  python compare_trees_by_lg_k.py --baseline_lg_k 16 --comparison_lg_k 11 --criterion gini --max_depth 3

  # Save trees as JSON baselines for future reference:
  python compare_trees_by_lg_k.py --baseline_lg_k 11 --comparison_lg_k 16 --criterion gini --save_baselines

  # Use existing JSON baselines for fast comparison:
  python compare_trees_by_lg_k.py --baseline_lg_k 11 --comparison_lg_k 16 --criterion gini --use_existing_baselines
        """
    )

    parser.add_argument('--baseline_lg_k', type=int, required=True,
                       help='lg_k parameter for baseline tree')
    parser.add_argument('--comparison_lg_k', type=int, required=True,
                       help='lg_k parameter for comparison tree')
    parser.add_argument('--criterion', type=str, default='gini',
                       choices=['gini', 'entropy', 'gain_ratio', 'chi_square', 'binomial'],
                       help='Split criterion to use (default: gini)')
    parser.add_argument('--max_depth', type=int, default=5,
                       help='Maximum tree depth (default: 5)')
    parser.add_argument('--tolerance', type=float, default=1e-3,
                       help='Tolerance for numerical differences (default: 1e-3)')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for comparison logs (default: current directory)')
    parser.add_argument('--save_baselines', action='store_true',
                       help='Save trees as JSON baseline files for future comparisons')
    parser.add_argument('--baseline_dir', type=str, default=None,
                       help='Directory to save/load baseline JSON files (default: tests/integration/mushroom/baselines)')
    parser.add_argument('--use_existing_baselines', action='store_true',
                       help='Load trees from existing JSON baseline files instead of building new ones')

    args = parser.parse_args()

    print(f"🔍 Tree Comparison Tool")
    print(f"=" * 50)
    print(f"Baseline lg_k: {args.baseline_lg_k}")
    print(f"Comparison lg_k: {args.comparison_lg_k}")
    print(f"Criterion: {args.criterion}")
    print(f"Max depth: {args.max_depth}")
    print(f"Tolerance: {args.tolerance}")

    try:
        # Get project root for path resolution
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent

        # Set default baseline directory if not specified
        if args.baseline_dir is None:
            args.baseline_dir = str(project_root / "tests/integration/mushroom/baselines")

        # Create baseline directory if it doesn't exist
        Path(args.baseline_dir).mkdir(parents=True, exist_ok=True)

        # Load feature mapping (needed for both modes)
        feature_mapping_file = project_root / "tests/fixtures/mushroom_feature_mapping.json"
        with open(feature_mapping_file, 'r') as f:
            feature_mapping = json.load(f)

        if args.use_existing_baselines:
            print(f"\n📁 Loading trees from existing JSON baselines...")

            # Load baseline tree from JSON
            baseline_config = f"lg_k_{args.baseline_lg_k}_{args.criterion}_depth_{args.max_depth}"
            baseline_filename = f"{args.baseline_dir}/mushroom_baseline_lg_k_{args.baseline_lg_k}_{args.criterion}_depth_{args.max_depth}.json"

            print(f"   Loading baseline from: {baseline_filename}")
            baseline_data = load_baseline_tree(baseline_filename, baseline_config)
            baseline_tree = json_to_tree_node(baseline_data["tree_structure"])

            # Load comparison tree from JSON
            comparison_config = f"lg_k_{args.comparison_lg_k}_{args.criterion}_depth_{args.max_depth}"
            comparison_filename = f"{args.baseline_dir}/mushroom_baseline_lg_k_{args.comparison_lg_k}_{args.criterion}_depth_{args.max_depth}.json"

            print(f"   Loading comparison from: {comparison_filename}")
            comparison_data = load_baseline_tree(comparison_filename, comparison_config)
            comparison_tree = json_to_tree_node(comparison_data["tree_structure"])

            print(f"   ✅ Both trees loaded from JSON baselines")

        else:
            # Build baseline tree
            print(f"\n🚀 Building baseline tree (lg_k={args.baseline_lg_k})...")
            baseline_classifier, feature_mapping = load_data_and_build_tree(
                args.baseline_lg_k, args.criterion, args.max_depth
            )
            baseline_tree = baseline_classifier.tree_

            # Build comparison tree
            print(f"\n🚀 Building comparison tree (lg_k={args.comparison_lg_k})...")
            comparison_classifier, _ = load_data_and_build_tree(
                args.comparison_lg_k, args.criterion, args.max_depth
            )
            comparison_tree = comparison_classifier.tree_

            # Save baselines if requested
            if args.save_baselines:
                print(f"\n💾 Saving tree baselines to JSON files...")

                # Save baseline tree
                baseline_filename = save_baseline_json(
                    baseline_classifier, feature_mapping, args.baseline_lg_k,
                    args.criterion, args.max_depth, args.baseline_dir
                )
                print(f"   Baseline tree saved: {baseline_filename}")

                # Save comparison tree
                comparison_filename = save_baseline_json(
                    comparison_classifier, feature_mapping, args.comparison_lg_k,
                    args.criterion, args.max_depth, args.baseline_dir
                )
                print(f"   Comparison tree saved: {comparison_filename}")

        # Compare trees
        print(f"\n🔍 Comparing tree structures...")
        comparator = TreeComparator(
            baseline_tree,
            comparison_tree,
            feature_mapping,
            tolerance=args.tolerance
        )

        # Perform recursive comparison
        comparator.compare_nodes_recursively(
            baseline_tree,
            comparison_tree
        )

        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{args.output_dir}/tree_comparison_lg_k_{args.baseline_lg_k}_vs_{args.comparison_lg_k}_{args.criterion}_{timestamp}.log"

        # Save detailed log
        print(f"\n💾 Saving detailed comparison to: {output_file}")
        comparison_log = comparator.save_comparison_log(
            output_file, args.baseline_lg_k, args.comparison_lg_k, args.criterion
        )

        # Print summary to console
        print(f"\n📊 COMPARISON SUMMARY:")
        print(f"   Total nodes compared: {comparator.differences['total_nodes']}")
        print(f"   Sample count differences: {comparator.differences['sample_count_diffs']}")
        print(f"   Class count differences: {comparator.differences['class_count_diffs']}")
        print(f"   Impurity differences: {comparator.differences['impurity_diffs']}")
        print(f"   Feature index differences: {comparator.differences['feature_index_diffs']}")
        print(f"   Structure differences: {comparator.differences['structure_diffs']}")

        total_diffs = sum(comparator.differences.values()) - comparator.differences['total_nodes']

        if total_diffs == 0:
            print(f"✅ RESULT: Trees are identical within tolerance ({args.tolerance})")
        else:
            print(f"❌ RESULT: {total_diffs} differences found between trees")

        print(f"\n📄 Detailed log saved to: {output_file}")

        if args.save_baselines:
            print(f"💾 JSON baselines saved in: {args.baseline_dir}")
            print(f"   - lg_k {args.baseline_lg_k}: mushroom_baseline_lg_k_{args.baseline_lg_k}_{args.criterion}_depth_{args.max_depth}.json")
            print(f"   - lg_k {args.comparison_lg_k}: mushroom_baseline_lg_k_{args.comparison_lg_k}_{args.criterion}_depth_{args.max_depth}.json")

        print(f"✅ Comparison completed successfully!")

    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        raise


if __name__ == "__main__":
    main()