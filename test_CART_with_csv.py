#!/usr/bin/env python3
"""
Test script for CART decision tree with any CSV dataset.

This script demonstrates how to run the theta sketch decision tree
on any binary classification CSV dataset.
"""

import subprocess
import sys
import os

def main():
    """Run CART decision tree test with binary classification CSV data."""

    # Change to script directory for consistent relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("🧪 Testing CART Decision Tree with CSV Dataset")
    print("=" * 60)

    # Command to run the binary classification
    cmd = [
        "./venv/bin/python", "run_binary_classification.py",
        "./tests/resources/binary_classification_data.csv", "target",
        "--lg_k", "16",
        "--max_depth", "5",
        "--criterion", "gini",
        "--tree_builder", "intersection",
        "--verbose", "1",
        "--sample_size", "1000"
    ]

    print("Running command:")
    print(" ".join(cmd))
    print()

    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Print the output
        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        print("✅ Test completed successfully!")
        return 0

    except subprocess.CalledProcessError as e:
        print(f"❌ Test failed with return code {e.returncode}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        return e.returncode

    except FileNotFoundError:
        print("❌ Error: Could not find the required files or virtual environment")
        print("Make sure you're running this from the project root directory")
        return 1


if __name__ == "__main__":
    exit(main())