#!/usr/bin/env python3
"""
Convert 2-column sketch CSV to 3-column format for binary decision tree training.

Usage:
    # Single target (generates complement automatically):
    python tools/convert_2col_to_3col_sketches.py \
        --input sketches_2col.csv \
        --total_sketch total_sketch_file \
        --positive_target "outcome=success" \
        --output_dir output/

    # Two targets (explicit positive/negative classes):
    python tools/convert_2col_to_3col_sketches.py \
        --input sketches_2col.csv \
        --total_sketch total_sketch_file \
        --positive_target "outcome=success" \
        --negative_target "outcome=failure" \
        --output_dir output/
"""

import argparse
import csv
import sys
import os
import base64
from pathlib import Path
from typing import Dict, List

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from datasketches import compact_theta_sketch, theta_a_not_b, theta_union
    from tests.test_binary_classification_sketches import ThetaSketchWrapper
except ImportError:
    raise ImportError("Apache DataSketches library required. Install with: pip install datasketches")


def load_sketch_from_csv(csv_file: str) -> Dict[str, ThetaSketchWrapper]:
    """Load 2-column sketch CSV into dictionary."""
    sketches = {}

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            feature_name, sketch_data = row
            sketch_bytes = base64.b64decode(sketch_data)
            compact_sketch = compact_theta_sketch.deserialize(sketch_bytes)
            sketches[feature_name] = ThetaSketchWrapper(compact_sketch)

    return sketches


def load_total_sketch(sketch_file: str) -> ThetaSketchWrapper:
    """Load total sketch from file."""
    with open(sketch_file, 'r') as f:
        sketch_data = f.read().strip()

    sketch_bytes = base64.b64decode(sketch_data)
    compact_sketch = compact_theta_sketch.deserialize(sketch_bytes)
    return ThetaSketchWrapper(compact_sketch)


def serialize_sketch(sketch: ThetaSketchWrapper) -> str:
    """Serialize sketch for CSV output."""
    sketch_bytes = sketch._sketch.serialize()
    return base64.b64encode(sketch_bytes).decode('utf-8')


def validate_sketches(present: ThetaSketchWrapper, absent: ThetaSketchWrapper,
                     class_sketch: ThetaSketchWrapper, tolerance: float) -> bool:
    """Validate sketch relationships with tolerance."""
    union = theta_union()
    union.update(present._sketch)
    union.update(absent._sketch)
    union_result = union.get_result()

    union_estimate = union_result.get_estimate()
    class_estimate = class_sketch.get_estimate()

    if class_estimate > 0:
        relative_error = abs(union_estimate - class_estimate) / class_estimate
        return relative_error <= tolerance
    return True


def generate_class_file(sketches: Dict[str, ThetaSketchWrapper],
                       class_sketch: ThetaSketchWrapper,
                       exclude_target: str,
                       output_file: Path,
                       tolerance: float,
                       validate: bool) -> None:
    """Generate 3-column CSV file for one class."""
    feature_count = 0

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['feature_name', 'sketch_present', 'sketch_absent'])

        for feature_name, feature_sketch in sketches.items():
            if feature_name == exclude_target:
                continue

            # Compute present and absent sketches
            present_sketch = feature_sketch.intersection(class_sketch)
            absent_result = theta_a_not_b(class_sketch._sketch, present_sketch._sketch)
            absent_sketch = ThetaSketchWrapper(absent_result)

            # Optional validation
            if validate and not validate_sketches(present_sketch, absent_sketch, class_sketch, tolerance):
                print(f"⚠️  Warning: {feature_name} failed validation")

            writer.writerow([feature_name, serialize_sketch(present_sketch), serialize_sketch(absent_sketch)])
            feature_count += 1

    print(f"Generated {output_file.name}: {feature_count} features")


def compute_complement_sketch(total_sketch: ThetaSketchWrapper, target_sketch: ThetaSketchWrapper) -> ThetaSketchWrapper:
    """Compute complement sketch: total - target using theta_a_not_b operation."""
    complement_result = theta_a_not_b(total_sketch._sketch, target_sketch._sketch)
    return ThetaSketchWrapper(complement_result)


def convert_to_binary_format(sketches: Dict[str, ThetaSketchWrapper],
                            total_sketch: ThetaSketchWrapper,
                            positive_target: str,
                            negative_target: str = None,
                            output_dir: str = ".",
                            tolerance: float = 0.07,
                            validate: bool = True) -> None:
    """Convert sketches to binary classification format.

    Parameters
    ----------
    sketches : dict
        Dictionary of feature sketches from 2-column CSV
    total_sketch : ThetaSketchWrapper
        Total population sketch
    positive_target : str
        Name of positive class feature
    negative_target : str, optional
        Name of negative class feature. If None, complement is computed automatically
    output_dir : str
        Output directory for generated files
    tolerance : float
        Validation tolerance for sketch relationships
    validate : bool
        Whether to perform validation checks
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Validate positive target exists
    if positive_target not in sketches:
        raise ValueError(f"Positive target '{positive_target}' not found in sketches")

    positive_sketch = sketches[positive_target]

    if negative_target is None:
        # Single target: compute complement (total - positive)
        print(f"Converting with single target: '{positive_target}'")
        print(f"Computing negative class as complement of total population")
        negative_sketch = compute_complement_sketch(total_sketch, positive_sketch)
        exclude_from_negative = positive_target
    else:
        # Two targets: use explicit negative class
        if negative_target not in sketches:
            raise ValueError(f"Negative target '{negative_target}' not found in sketches")
        print(f"Converting with two targets: '{positive_target}' vs '{negative_target}'")
        negative_sketch = sketches[negative_target]
        exclude_from_negative = negative_target

    # Generate positive class file
    positive_file = Path(output_dir) / "positive_sketches.csv"
    generate_class_file(sketches, positive_sketch, positive_target, positive_file, tolerance, validate)

    # Generate negative class file
    negative_file = Path(output_dir) / "negative_sketches.csv"
    generate_class_file(sketches, negative_sketch, exclude_from_negative, negative_file, tolerance, validate)




def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Convert 2-column to 3-column sketch format for binary classification')
    parser.add_argument('--input', required=True, help='Input 2-column CSV file')
    parser.add_argument('--total_sketch', required=True, help='Total sketch file')
    parser.add_argument('--positive_target', required=True, help='Positive class target feature')
    parser.add_argument('--negative_target', help='Negative class target feature (optional - complement computed if not provided)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--tolerance', type=float, default=0.07, help='Validation tolerance (default: 0.07)')
    parser.add_argument('--no-validate', action='store_true', help='Skip validation checks')

    args = parser.parse_args()

    try:
        print(f"Loading sketches from: {args.input}")
        sketches = load_sketch_from_csv(args.input)
        print(f"Loaded {len(sketches)} feature sketches")

        print(f"Loading total sketch from: {args.total_sketch}")
        total_sketch = load_total_sketch(args.total_sketch)

        # Convert
        convert_to_binary_format(
            sketches=sketches,
            total_sketch=total_sketch,
            positive_target=args.positive_target,
            negative_target=args.negative_target,
            output_dir=args.output_dir,
            tolerance=args.tolerance,
            validate=not args.no_validate
        )

        print(f"Conversion complete! Output files in: {args.output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())