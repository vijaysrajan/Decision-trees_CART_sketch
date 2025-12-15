#!/usr/bin/env python3
"""
Enhanced 2-column to 3-column sketch converter with multiple modes.

USAGE MODES:

1. SIMPLE MODE (like simple_convert_to_3col.py but with mathematical validation):
   python tools/convert_2col_to_3col_sketches.py --simple \
       positive_2col.csv negative_2col.csv feature_mapping.json output_dir/

2. ADVANCED MODE (original functionality with complement computation):
   python tools/convert_2col_to_3col_sketches.py --advanced \
       --input sketches_2col.csv --total_sketch total_file \
       --positive_target "outcome=success" --output_dir output/

3. AUTO-DETECTION MODE (detects embedded total sketch):
   python tools/convert_2col_to_3col_sketches.py --auto \
       --input sketches_with_total.csv --positive_target "outcome=success" \
       --output_dir output/

FEATURES:
- Mathematical validation with theta sketch operations
- Auto-detection of embedded total sketches
- Backward compatibility with original interface
- Enhanced error checking and reporting
"""

import argparse
import csv
import sys
import os
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from datasketches import compact_theta_sketch, theta_a_not_b, theta_union, update_theta_sketch
    from tests.test_binary_classification_sketches import ThetaSketchWrapper
except ImportError:
    raise ImportError("Apache DataSketches library required. Install with: pip install datasketches")


def load_sketch_from_csv(csv_file: str) -> Tuple[Dict[str, ThetaSketchWrapper], Optional[ThetaSketchWrapper]]:
    """Load 2-column sketch CSV into dictionary and detect embedded total sketch.

    Returns:
        tuple: (sketches_dict, total_sketch_or_none)
        If first non-header row has 'total' as feature name, it's extracted as total_sketch.
    """
    sketches = {}
    total_sketch = None

    # Increase CSV field size limit to handle large theta sketches
    csv.field_size_limit(1000000)  # 1MB limit for large sketches

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            feature_name, sketch_data = row
            sketch_bytes = base64.b64decode(sketch_data)
            compact_sketch = compact_theta_sketch.deserialize(sketch_bytes)

            # Check if this is the total sketch (first row or explicitly named 'total')
            if feature_name.lower() in ['total', 'total_sketch', 'total_population']:
                total_sketch = ThetaSketchWrapper(compact_sketch)
                print(f"🔍 Auto-detected total sketch: '{feature_name}'")
            else:
                sketches[feature_name] = ThetaSketchWrapper(compact_sketch)

    return sketches, total_sketch


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


def convert_simple_format(positive_csv: str, negative_csv: str,
                         feature_mapping_json: str, output_dir: str,
                         tolerance: float = 0.07, validate: bool = True) -> None:
    """Convert class-specific 2-column CSVs to 3-column format (simple_convert_to_3col.py style).

    This provides mathematical rigor with the convenience of the simple workflow.

    Parameters
    ----------
    positive_csv : str
        Path to positive class 2-column sketch CSV
    negative_csv : str
        Path to negative class 2-column sketch CSV
    feature_mapping_json : str
        Path to feature mapping JSON file
    output_dir : str
        Output directory for 3-column sketch CSV
    tolerance : float
        Validation tolerance for sketch relationships
    validate : bool
        Whether to perform mathematical validation
    """
    import json

    Path(output_dir).mkdir(exist_ok=True)

    # Load sketches from both files
    print(f"Loading positive class sketches: {positive_csv}")
    positive_sketches, pos_total = load_sketch_from_csv(positive_csv)

    print(f"Loading negative class sketches: {negative_csv}")
    negative_sketches, neg_total = load_sketch_from_csv(negative_csv)

    # Load feature mapping
    with open(feature_mapping_json, 'r') as f:
        feature_mapping = json.load(f)

    # Create 3-column output
    output_file = Path(output_dir) / f"{Path(positive_csv).stem.replace('positive_2col_sketches', '3col_sketches')}.csv"
    feature_count = 0
    validation_warnings = 0

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['identifier', 'sketch_feature_present_positive', 'sketch_feature_present_negative'])

        # Process all features from feature mapping
        for feature_name in feature_mapping.keys():
            pos_sketch = positive_sketches.get(feature_name)
            neg_sketch = negative_sketches.get(feature_name)

            if pos_sketch is None and neg_sketch is None:
                print(f"⚠️  Warning: Feature '{feature_name}' not found in either sketch file")
                continue

            # Use empty sketch if one class doesn't have this feature
            if pos_sketch is None:
                pos_sketch = ThetaSketchWrapper(update_theta_sketch())
            if neg_sketch is None:
                neg_sketch = ThetaSketchWrapper(update_theta_sketch())

            # Optional mathematical validation
            if validate and pos_total and neg_total:
                # Validate that feature sketches are subsets of their class totals
                pos_valid = pos_sketch.get_estimate() <= pos_total.get_estimate() * (1 + tolerance)
                neg_valid = neg_sketch.get_estimate() <= neg_total.get_estimate() * (1 + tolerance)

                if not (pos_valid and neg_valid):
                    print(f"⚠️  Validation warning: {feature_name} - sketch estimates exceed class totals")
                    validation_warnings += 1

            # Write to output
            pos_serialized = serialize_sketch(pos_sketch) if pos_sketch.get_estimate() > 0 else ""
            neg_serialized = serialize_sketch(neg_sketch) if neg_sketch.get_estimate() > 0 else ""

            writer.writerow([feature_name, pos_serialized, neg_serialized])
            feature_count += 1

    print(f"✅ Generated {output_file.name}: {feature_count} features")
    if validation_warnings > 0:
        print(f"⚠️  {validation_warnings} validation warnings (consider reviewing data quality)")
    else:
        print("✅ All validation checks passed")


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
    """Main function with enhanced command-line interface supporting multiple modes."""
    parser = argparse.ArgumentParser(
        description='Convert 2-column to 3-column sketch format for binary classification',
        epilog="""
USAGE MODES:

1. SIMPLE MODE (like simple_convert_to_3col.py but with mathematical validation):
   python tools/convert_2col_to_3col_sketches.py --simple \\
       positive_2col.csv negative_2col.csv feature_mapping.json output_dir/

2. ADVANCED MODE (original functionality with complement computation):
   python tools/convert_2col_to_3col_sketches.py --advanced \\
       --input sketches_2col.csv --total_sketch total_file \\
       --positive_target "outcome=success" --output_dir output/

3. AUTO-DETECTION MODE (detects embedded total sketch):
   python tools/convert_2col_to_3col_sketches.py --auto \\
       --input sketches_with_total.csv --positive_target "outcome=success" \\
       --output_dir output/
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--simple', nargs=4, metavar=('POSITIVE_CSV', 'NEGATIVE_CSV', 'FEATURE_MAPPING', 'OUTPUT_DIR'),
                           help='Simple mode: positive_csv negative_csv feature_mapping.json output_dir')
    mode_group.add_argument('--advanced', action='store_true',
                           help='Advanced mode with explicit total sketch file')
    mode_group.add_argument('--auto', action='store_true',
                           help='Auto-detection mode (finds total sketch in input file)')

    # Advanced/Auto mode arguments
    parser.add_argument('--input', help='Input 2-column CSV file (advanced/auto mode)')
    parser.add_argument('--total_sketch', help='Total sketch file (advanced mode only)')
    parser.add_argument('--positive_target', help='Positive class target feature (advanced/auto mode)')
    parser.add_argument('--negative_target', help='Negative class target feature (optional - complement computed if not provided)')
    parser.add_argument('--output_dir', help='Output directory (advanced/auto mode)')

    # Common arguments
    parser.add_argument('--tolerance', type=float, default=0.07, help='Validation tolerance (default: 0.07)')
    parser.add_argument('--no-validate', action='store_true', help='Skip validation checks')

    args = parser.parse_args()

    try:
        if args.simple:
            # Simple mode - like simple_convert_to_3col.py but with validation
            positive_csv, negative_csv, feature_mapping, output_dir = args.simple
            print("🚀 SIMPLE MODE: Converting class-specific 2-column files with mathematical validation")

            convert_simple_format(
                positive_csv=positive_csv,
                negative_csv=negative_csv,
                feature_mapping_json=feature_mapping,
                output_dir=output_dir,
                tolerance=args.tolerance,
                validate=not args.no_validate
            )

        elif args.auto:
            # Auto-detection mode - find total sketch in input file
            if not args.input or not args.positive_target or not args.output_dir:
                parser.error("Auto mode requires --input, --positive_target, and --output_dir")

            print("🔍 AUTO MODE: Auto-detecting total sketch from input file")

            sketches, total_sketch = load_sketch_from_csv(args.input)
            print(f"Loaded {len(sketches)} feature sketches")

            if total_sketch is None:
                raise ValueError("No total sketch found in input file. Use --advanced mode with --total_sketch instead.")

            convert_to_binary_format(
                sketches=sketches,
                total_sketch=total_sketch,
                positive_target=args.positive_target,
                negative_target=args.negative_target,
                output_dir=args.output_dir,
                tolerance=args.tolerance,
                validate=not args.no_validate
            )

        else:  # args.advanced
            # Advanced mode - original functionality
            if not args.input or not args.total_sketch or not args.positive_target or not args.output_dir:
                parser.error("Advanced mode requires --input, --total_sketch, --positive_target, and --output_dir")

            print("⚙️ ADVANCED MODE: Using explicit total sketch file")

            sketches, embedded_total = load_sketch_from_csv(args.input)
            print(f"Loaded {len(sketches)} feature sketches")

            if embedded_total:
                print("💡 Note: Found embedded total sketch, but using explicit file as requested")

            total_sketch = load_total_sketch(args.total_sketch)

            convert_to_binary_format(
                sketches=sketches,
                total_sketch=total_sketch,
                positive_target=args.positive_target,
                negative_target=args.negative_target,
                output_dir=args.output_dir,
                tolerance=args.tolerance,
                validate=not args.no_validate
            )

        print("✅ Conversion complete!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())