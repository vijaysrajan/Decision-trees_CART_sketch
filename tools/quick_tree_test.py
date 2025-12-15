#!/usr/bin/env python3
"""
Quick tree testing: Extract a few key rules for manual validation.

This tool extracts the most important decision tree rules for quick SQL testing.

Usage:
    python tools/quick_tree_test.py path/to/tree.json

Example:
    python tools/quick_tree_test.py DU_output/test_model_lg_k_18/3col_sketches_model_lg_k_18.json
"""

import json
import argparse
from pathlib import Path


def extract_key_rules(tree_json_path, num_rules=5):
    """Extract the most important rules for quick testing."""

    # Load tree
    with open(tree_json_path, 'r') as f:
        tree_data = json.load(f)

    if 'tree_structure' in tree_data:
        tree = tree_data['tree_structure']
    else:
        tree = tree_data

    rules = []

    def traverse(node, conditions, depth):
        if node.get('is_leaf', False):
            # Leaf node - extract rule
            sql_conditions = []
            english_conditions = []

            for feature, value in conditions:
                if value:
                    sql_conditions.append(f"`{feature}` = 1")
                    english_conditions.append(f"{feature} = TRUE")
                else:
                    sql_conditions.append(f"`{feature}` = 0")
                    english_conditions.append(f"{feature} = FALSE")

            sql_where = " AND ".join(sql_conditions) if sql_conditions else "1=1"
            english = " AND ".join(english_conditions) if english_conditions else "All records"

            n_samples = node.get('n_samples', 0)
            prediction = node.get('prediction', -1)
            probs = node.get('class_probabilities', node.get('probabilities', [0.0, 0.0]))

            rules.append({
                'samples': n_samples,
                'depth': depth,
                'prediction': 'Positive' if prediction == 1 else 'Negative',
                'confidence': max(probs),
                'sql': sql_where,
                'english': english,
                'positive_rate': probs[1] if len(probs) > 1 else 0.0
            })
        else:
            # Split node
            feature = node.get('feature_name', 'unknown')

            # Left child (feature = 0/FALSE)
            if 'left' in node:
                traverse(node['left'], conditions + [(feature, False)], depth + 1)

            # Right child (feature = 1/TRUE)
            if 'right' in node:
                traverse(node['right'], conditions + [(feature, True)], depth + 1)

    traverse(tree, [], 0)

    # Sort by sample size (most representative rules first)
    rules.sort(key=lambda x: x['samples'], reverse=True)

    return rules[:num_rules]


def main():
    parser = argparse.ArgumentParser(description="Extract key decision tree rules for quick testing")
    parser.add_argument('tree_json', help='Path to JSON tree file')
    parser.add_argument('--num_rules', '-n', type=int, default=5, help='Number of rules to extract')
    parser.add_argument('--table', default='DU_raw', help='Table name for SQL')
    parser.add_argument('--target', default='target', help='Target column name')

    args = parser.parse_args()

    if not Path(args.tree_json).exists():
        print(f"❌ File not found: {args.tree_json}")
        return 1

    try:
        rules = extract_key_rules(args.tree_json, args.num_rules)

        print(f"🔍 Top {len(rules)} Decision Tree Rules for Manual Testing")
        print("=" * 80)
        print(f"Source: {args.tree_json}")
        print(f"Table: {args.table}")
        print(f"Target: {args.target}")
        print()

        for i, rule in enumerate(rules, 1):
            print(f"📊 Rule {i}: {rule['prediction']} ({rule['confidence']:.1%} confidence)")
            print(f"   Samples: {rule['samples']:.0f}")
            print(f"   Conditions: {rule['english']}")
            print(f"   Expected positive rate: {rule['positive_rate']:.1%}")
            print()
            print(f"   SQL Test Query:")
            print(f"   SELECT")
            print(f"     COUNT(*) as sample_count,")
            print(f"     ROUND(AVG(CASE WHEN {args.target} = 1 THEN 1.0 ELSE 0.0 END), 4) as positive_rate")
            print(f"   FROM {args.table}")
            print(f"   WHERE {rule['sql']};")
            print(f"   -- Expected: sample_count ≈ {rule['samples']:.0f}, positive_rate ≈ {rule['positive_rate']:.4f}")
            print()
            print("-" * 80)
            print()

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())