"""
Test sample conservation property in theta sketch decision trees.

This test verifies that parent_samples = left_samples + right_samples
for all splits in the tree.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_mushroom_sketches import (
    load_mushroom_dataset, create_mushroom_sketches, create_mushroom_feature_mapping,
    DEFAULT_LG_K, DEFAULT_MIN_SAMPLES_SPLIT, DEFAULT_MIN_SAMPLES_LEAF, DEFAULT_TREE_BUILDER
)
from theta_sketch_tree.classifier import ThetaSketchDecisionTreeClassifier


def check_sample_conservation(node, path="root"):
    """
    Recursively check that parent_samples = left_samples + right_samples.

    Returns list of violations: [(path, parent_samples, left_samples, right_samples, deficit)]
    """
    violations = []

    def traverse(n, p):
        if not n.is_leaf:
            left_samples = n.left.n_samples if n.left else 0
            right_samples = n.right.n_samples if n.right else 0
            parent_samples = n.n_samples
            total_children = left_samples + right_samples

            if abs(parent_samples - total_children) > 0.1:  # Allow small floating point errors
                deficit = parent_samples - total_children
                violations.append((p, parent_samples, left_samples, right_samples, deficit))
                print(f"❌ VIOLATION at {p}:")
                print(f"   Parent: {parent_samples:.1f}")
                print(f"   Left:   {left_samples:.1f}")
                print(f"   Right:  {right_samples:.1f}")
                print(f"   Total:  {total_children:.1f}")
                print(f"   Deficit: {deficit:.1f}")
                print()

            # Recursively check children
            if n.left:
                traverse(n.left, p + ".left")
            if n.right:
                traverse(n.right, p + ".right")

    traverse(node, path)
    return violations


def main():
    print("🧪 Testing Sample Conservation Property")
    print("=" * 50)

    # Load dataset
    df = load_mushroom_dataset()
    sketches = create_mushroom_sketches(df, lg_k=DEFAULT_LG_K)
    mapping = create_mushroom_feature_mapping(sketches)

    # Build tree
    clf = ThetaSketchDecisionTreeClassifier(
        criterion='gini',
        max_depth=4,  # Shallow for clear testing
        min_samples_split=DEFAULT_MIN_SAMPLES_SPLIT,
        min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
        tree_builder=DEFAULT_TREE_BUILDER,
        verbose=0
    )
    clf.fit(sketches, mapping)

    # Check sample conservation
    print(f"🌳 Tree built with root having {clf.tree_.n_samples:.0f} samples")
    print(f"📊 Checking sample conservation property...")
    print()

    violations = check_sample_conservation(clf.tree_)

    if violations:
        print(f"❌ Found {len(violations)} sample conservation violations!")
        print("This indicates a bug in the theta sketch intersection operations.")
        return False
    else:
        print("✅ All nodes satisfy sample conservation: parent = left + right")
        print("The theta sketch operations are working correctly.")
        return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)