#!/usr/bin/env python3
"""
Script to run tree comparison with comprehensive logging.
Useful for debugging and manual inspection of tree structures.
"""

import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.test_mushroom_regression import TestMushroomRegression
from tests.test_binary_classification_sketches import (
    load_mushroom_dataset,
    create_mushroom_sketches,
    create_mushroom_feature_mapping
)


def load_mushroom_data():
    """Load mushroom data (equivalent to the mushroom_data fixture)."""
    print("📊 Loading mushroom dataset...")
    df = load_mushroom_dataset()
    print("🌿 Creating theta sketches...")
    sketches = create_mushroom_sketches(df)
    print("🗂️ Creating feature mapping...")
    feature_mapping = create_mushroom_feature_mapping(sketches)
    return df, sketches, feature_mapping


def load_baseline_outputs():
    """Load baseline outputs (equivalent to the baseline_outputs fixture)."""
    print("📋 Loading baseline outputs...")
    try:
        with open("mushroom_baseline_outputs.json", 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ Baseline outputs not found. Run generate_mushroom_baselines.py first.")
        sys.exit(1)


def main():
    """Run a single tree comparison test with comprehensive logging."""
    config_name = "default_gini"  # Default configuration

    if len(sys.argv) > 1:
        config_name = sys.argv[1]

    print(f"🧪 Running tree comparison test for: {config_name}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Load test data directly (not using pytest fixtures)
    mushroom_data = load_mushroom_data()
    baseline_outputs = load_baseline_outputs()

    # Create test instance
    test_instance = TestMushroomRegression()

    # Run the test with logging
    print(f"🔍 Testing configuration: {config_name}")
    try:
        test_instance.test_core_criteria_regression(mushroom_data, baseline_outputs, config_name)
        print(f"✅ Test completed successfully!")
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        raise

    print("="*80)
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Log files generated:")
    print(f"   - tree_comparison_{config_name}.log (detailed recursive comparison)")


if __name__ == "__main__":
    available_configs = [
        "default_gini",
        "entropy_shallow",
        "gain_ratio_medium",
        "binomial_deep",
        "chi_square_default"
    ]

    print("Available configurations:", ", ".join(available_configs))
    print("Usage: python run_tree_comparison_with_logging.py [config_name]")
    print()

    main()