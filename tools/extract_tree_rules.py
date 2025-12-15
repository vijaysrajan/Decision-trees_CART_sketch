#!/usr/bin/env python3
"""
Extract decision tree rules from JSON tree structure for manual SQL validation.

This tool traverses a decision tree JSON file and generates:
1. SQL WHERE clauses for each leaf node
2. Human-readable English rules
3. Expected predictions and probabilities

Usage:
    python tools/extract_tree_rules.py path/to/tree.json --feature_mapping path/to/mapping.json

Example:
    python tools/extract_tree_rules.py DU_output/test_model_lg_k_18/3col_sketches_model_lg_k_18.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


class TreeRuleExtractor:
    """
    Extracts decision rules from JSON tree structure for SQL validation.
    """

    def __init__(self, tree_json_path: str, feature_mapping_path: Optional[str] = None):
        """
        Initialize the rule extractor.

        Parameters
        ----------
        tree_json_path : str
            Path to the JSON tree file
        feature_mapping_path : str, optional
            Path to feature mapping JSON file
        """
        self.tree_json_path = Path(tree_json_path)
        self.feature_mapping_path = Path(feature_mapping_path) if feature_mapping_path else None

        # Load tree structure
        with open(self.tree_json_path, 'r') as f:
            self.tree_data = json.load(f)

        # Extract tree structure from different possible formats
        if 'tree_structure' in self.tree_data:
            self.tree = self.tree_data['tree_structure']
        else:
            # Assume the whole file is the tree structure
            self.tree = self.tree_data

        # Load feature mapping if provided
        self.feature_mapping = {}
        if self.feature_mapping_path and self.feature_mapping_path.exists():
            with open(self.feature_mapping_path, 'r') as f:
                self.feature_mapping = json.load(f)

        # Store extracted rules
        self.rules = []

    def extract_all_rules(self) -> List[Dict[str, Any]]:
        """
        Extract all decision rules from the tree.

        Returns
        -------
        rules : List[Dict]
            List of rule dictionaries with SQL, English, and prediction info
        """
        self.rules = []
        self._traverse_tree(self.tree, [], "root")
        return self.rules

    def _traverse_tree(self, node: Dict[str, Any], path_conditions: List[Tuple[str, bool]], node_path: str):
        """
        Recursively traverse the tree and extract rules.

        Parameters
        ----------
        node : Dict
            Current tree node
        path_conditions : List[Tuple[str, bool]]
            List of (feature_name, is_present) conditions from root to current node
        node_path : str
            Human-readable path description
        """
        if node.get('is_leaf', False):
            # Leaf node - create rule
            rule = self._create_rule_from_path(path_conditions, node, node_path)
            self.rules.append(rule)
        else:
            # Split node - recurse to children
            feature_name = node.get('feature_name', 'unknown_feature')

            # Left child: feature absent (= 0/FALSE)
            if 'left' in node:
                left_conditions = path_conditions + [(feature_name, False)]
                left_path = f"{node_path} → {feature_name}=0"
                self._traverse_tree(node['left'], left_conditions, left_path)

            # Right child: feature present (= 1/TRUE)
            if 'right' in node:
                right_conditions = path_conditions + [(feature_name, True)]
                right_path = f"{node_path} → {feature_name}=1"
                self._traverse_tree(node['right'], right_conditions, right_path)

    def _create_rule_from_path(self, path_conditions: List[Tuple[str, bool]],
                              leaf_node: Dict[str, Any], node_path: str) -> Dict[str, Any]:
        """
        Create a complete rule from path conditions and leaf node info.

        Parameters
        ----------
        path_conditions : List[Tuple[str, bool]]
            List of (feature_name, is_present) conditions
        leaf_node : Dict
            Leaf node data
        node_path : str
            Human-readable path

        Returns
        -------
        rule : Dict
            Complete rule with SQL, English, and prediction info
        """
        # Build SQL WHERE clause
        sql_conditions = []
        english_conditions = []

        for feature_name, is_present in path_conditions:
            if is_present:
                sql_conditions.append(f"{feature_name} = 1")
                english_conditions.append(f"{feature_name} is TRUE")
            else:
                sql_conditions.append(f"{feature_name} = 0")
                english_conditions.append(f"{feature_name} is FALSE")

        sql_where = " AND ".join(sql_conditions) if sql_conditions else "TRUE"
        english_rule = " AND ".join(english_conditions) if english_conditions else "No conditions"

        # Extract prediction info
        prediction = leaf_node.get('prediction', 'unknown')
        class_counts = leaf_node.get('class_counts', [0, 0])
        n_samples = leaf_node.get('n_samples', 0)
        probabilities = leaf_node.get('class_probabilities', leaf_node.get('probabilities', [0.0, 0.0]))

        # Calculate confidence metrics
        total_samples = sum(class_counts) if class_counts else n_samples
        confidence = max(probabilities) if probabilities else 0.0

        rule = {
            'rule_id': len(self.rules) + 1,
            'node_path': node_path,
            'sql_where_clause': sql_where,
            'english_rule': english_rule,
            'prediction': int(prediction) if prediction != 'unknown' else None,
            'prediction_label': 'Positive' if prediction == 1 else 'Negative' if prediction == 0 else 'Unknown',
            'confidence': round(confidence, 4),
            'class_counts': class_counts,
            'n_samples': total_samples,
            'class_probabilities': [round(p, 4) for p in probabilities] if probabilities else [0.0, 0.0],
            'conditions_count': len(path_conditions),
            'depth': len(path_conditions)
        }

        return rule

    def generate_sql_validation_queries(self, table_name: str = "your_table",
                                       target_column: str = "target") -> List[str]:
        """
        Generate SQL queries to validate each rule against the database.

        Parameters
        ----------
        table_name : str
            Name of the database table
        target_column : str
            Name of the target column

        Returns
        -------
        queries : List[str]
            List of SQL validation queries
        """
        queries = []

        for rule in self.rules:
            where_clause = rule['sql_where_clause']
            expected_samples = rule['n_samples']
            expected_prediction = rule['prediction']

            # Count total samples matching the conditions
            count_query = f"""
-- Rule {rule['rule_id']}: {rule['english_rule']}
SELECT
    COUNT(*) as actual_samples,
    {expected_samples} as expected_samples,
    ROUND(AVG(CASE WHEN {target_column} = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    {rule['class_probabilities'][1]} as expected_positive_rate,
    '{rule['prediction_label']}' as expected_prediction
FROM {table_name}
WHERE {where_clause};
"""
            queries.append(count_query)

        return queries

    def print_summary_report(self):
        """Print a comprehensive summary of extracted rules."""
        print(f"🌳 Decision Tree Rule Extraction Report")
        print(f"=" * 60)
        print(f"Tree file: {self.tree_json_path}")
        print(f"Total rules extracted: {len(self.rules)}")
        print(f"Tree depth: {max(rule['depth'] for rule in self.rules) if self.rules else 0}")

        # Summary by prediction
        positive_rules = [r for r in self.rules if r['prediction'] == 1]
        negative_rules = [r for r in self.rules if r['prediction'] == 0]

        print(f"\nPrediction Summary:")
        print(f"  Positive prediction rules: {len(positive_rules)}")
        print(f"  Negative prediction rules: {len(negative_rules)}")

        if positive_rules:
            avg_pos_confidence = sum(r['confidence'] for r in positive_rules) / len(positive_rules)
            print(f"  Average positive confidence: {avg_pos_confidence:.4f}")

        if negative_rules:
            avg_neg_confidence = sum(r['confidence'] for r in negative_rules) / len(negative_rules)
            print(f"  Average negative confidence: {avg_neg_confidence:.4f}")

        print(f"\n" + "="*60)

    def print_detailed_rules(self, max_rules: Optional[int] = None):
        """
        Print detailed rules in human-readable format.

        Parameters
        ----------
        max_rules : int, optional
            Maximum number of rules to print (default: all)
        """
        rules_to_print = self.rules[:max_rules] if max_rules else self.rules

        print(f"\n📋 Detailed Decision Rules")
        print(f"=" * 80)

        for rule in rules_to_print:
            print(f"\n🔸 Rule {rule['rule_id']} (Depth {rule['depth']})")
            print(f"   Path: {rule['node_path']}")
            print(f"   Conditions: {rule['english_rule']}")
            print(f"   SQL WHERE: {rule['sql_where_clause']}")
            print(f"   👉 Prediction: {rule['prediction_label']} (confidence: {rule['confidence']:.1%})")
            print(f"   📊 Samples: {rule['n_samples']} (class distribution: {rule['class_counts']})")
            print(f"   🎯 Probabilities: [Negative: {rule['class_probabilities'][0]:.1%}, Positive: {rule['class_probabilities'][1]:.1%}]")

    def save_rules_to_file(self, output_path: str):
        """
        Save extracted rules to a JSON file.

        Parameters
        ----------
        output_path : str
            Path to save the rules JSON file
        """
        output_data = {
            'metadata': {
                'source_tree': str(self.tree_json_path),
                'total_rules': len(self.rules),
                'extraction_timestamp': None  # Could add datetime if needed
            },
            'rules': self.rules
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"💾 Rules saved to: {output_path}")

    def save_sql_queries(self, output_path: str, table_name: str = "DU_raw",
                        target_column: str = "target"):
        """
        Save SQL validation queries to a file.

        Parameters
        ----------
        output_path : str
            Path to save the SQL file
        table_name : str
            Database table name
        target_column : str
            Target column name
        """
        queries = self.generate_sql_validation_queries(table_name, target_column)

        with open(output_path, 'w') as f:
            f.write(f"-- Decision Tree Validation Queries\n")
            f.write(f"-- Generated from: {self.tree_json_path}\n")
            f.write(f"-- Total rules: {len(self.rules)}\n")
            f.write(f"-- Table: {table_name}\n")
            f.write(f"-- Target column: {target_column}\n\n")

            for i, query in enumerate(queries, 1):
                f.write(f"{query}\n")
                if i < len(queries):
                    f.write("\n" + "-"*80 + "\n\n")

        print(f"📝 SQL queries saved to: {output_path}")


def main():
    """Main function to run the tree rule extraction."""
    parser = argparse.ArgumentParser(
        description="Extract decision tree rules for SQL validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract rules from DU tree
    python tools/extract_tree_rules.py DU_output/test_model_lg_k_18/3col_sketches_model_lg_k_18.json

    # With feature mapping
    python tools/extract_tree_rules.py tree.json --feature_mapping feature_mapping.json

    # Save SQL queries for validation
    python tools/extract_tree_rules.py tree.json --save_sql validation_queries.sql --table DU_raw

    # Limit displayed rules
    python tools/extract_tree_rules.py tree.json --max_display 10
        """
    )

    parser.add_argument('tree_json', help='Path to the JSON tree file')
    parser.add_argument('--feature_mapping', help='Path to feature mapping JSON file')
    parser.add_argument('--save_rules', help='Save extracted rules to JSON file')
    parser.add_argument('--save_sql', help='Save SQL validation queries to file')
    parser.add_argument('--table', default='DU_raw', help='Database table name for SQL queries')
    parser.add_argument('--target_column', default='target', help='Target column name')
    parser.add_argument('--max_display', type=int, help='Maximum number of rules to display')
    parser.add_argument('--quiet', '-q', action='store_true', help='Only show summary, no detailed rules')

    args = parser.parse_args()

    # Validate input file
    if not Path(args.tree_json).exists():
        print(f"❌ Error: Tree JSON file not found: {args.tree_json}")
        return 1

    try:
        # Create extractor and extract rules
        extractor = TreeRuleExtractor(args.tree_json, args.feature_mapping)
        rules = extractor.extract_all_rules()

        # Print summary
        extractor.print_summary_report()

        # Print detailed rules unless quiet mode
        if not args.quiet:
            extractor.print_detailed_rules(args.max_display)

        # Save outputs if requested
        if args.save_rules:
            extractor.save_rules_to_file(args.save_rules)

        if args.save_sql:
            extractor.save_sql_queries(args.save_sql, args.table, args.target_column)

        print(f"\n✅ Rule extraction completed successfully!")
        print(f"   Total rules: {len(rules)}")

        if not args.save_sql and not args.quiet:
            print(f"\n💡 Tip: Use --save_sql to generate SQL validation queries")

        return 0

    except Exception as e:
        print(f"❌ Error during rule extraction: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())