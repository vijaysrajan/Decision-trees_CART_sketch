"""
Generate test CSV files with real theta sketches for testing.

This script creates sample CSV files with actual ThetaSketch objects
for use in unit tests.
"""

import base64
from datasketches import update_theta_sketch


def create_sketch(values: list, lg_k: int = 12) -> bytes:
    """
    Create a ThetaSketch from a list of values and serialize it.

    Parameters
    ----------
    values : list
        List of values to add to the sketch
    lg_k : int, default=12
        Log2 of sketch size (12 = 4096)

    Returns
    -------
    bytes
        Serialized sketch bytes
    """
    sketch = update_theta_sketch(lg_k=lg_k)
    for value in values:
        sketch.update(str(value))
    compact = sketch.compact()
    return compact.serialize()


def create_test_csvs():
    """Create test CSV files for SketchLoader testing."""

    # Create sample data
    # Total population: IDs 1-1000
    total_ids = list(range(1, 1001))

    # Positive class (target=yes): IDs 1-400 (40%)
    pos_ids = list(range(1, 401))

    # Negative class (target=no): IDs 401-1000 (60%)
    neg_ids = list(range(401, 1001))

    # Feature: age>30
    # Positive class: 300 have age>30, 100 don't
    pos_age_gt30 = list(range(1, 301))
    pos_age_le30 = list(range(301, 401))

    # Negative class: 350 have age>30, 250 don't
    neg_age_gt30 = list(range(401, 751))
    neg_age_le30 = list(range(751, 1001))

    # Feature: income>50k
    # Positive class: 250 have income>50k, 150 don't
    pos_income_gt50k = list(range(1, 251))
    pos_income_le50k = list(range(251, 401))

    # Negative class: 200 have income>50k, 400 don't
    neg_income_gt50k = list(range(401, 601))
    neg_income_le50k = list(range(601, 1001))

    # Create sketches
    print("Generating theta sketches...")

    # Mode 2 Format 3: Dual CSV with 3 columns (RECOMMENDED)
    # target_yes.csv
    with open("tests/fixtures/target_yes_3col.csv", "w") as f:
        f.write("identifier,sketch_feature_present,sketch_feature_absent\n")

        # total
        total_sketch = base64.b64encode(create_sketch(pos_ids)).decode("utf-8")
        f.write(f"total,{total_sketch},{total_sketch}\n")

        # age>30
        age_present = base64.b64encode(create_sketch(pos_age_gt30)).decode("utf-8")
        age_absent = base64.b64encode(create_sketch(pos_age_le30)).decode("utf-8")
        f.write(f"age>30,{age_present},{age_absent}\n")

        # income>50k
        inc_present = base64.b64encode(create_sketch(pos_income_gt50k)).decode("utf-8")
        inc_absent = base64.b64encode(create_sketch(pos_income_le50k)).decode("utf-8")
        f.write(f"income>50k,{inc_present},{inc_absent}\n")

    # target_no.csv
    with open("tests/fixtures/target_no_3col.csv", "w") as f:
        f.write("identifier,sketch_feature_present,sketch_feature_absent\n")

        # total
        total_sketch = base64.b64encode(create_sketch(neg_ids)).decode("utf-8")
        f.write(f"total,{total_sketch},{total_sketch}\n")

        # age>30
        age_present = base64.b64encode(create_sketch(neg_age_gt30)).decode("utf-8")
        age_absent = base64.b64encode(create_sketch(neg_age_le30)).decode("utf-8")
        f.write(f"age>30,{age_present},{age_absent}\n")

        # income>50k
        inc_present = base64.b64encode(create_sketch(neg_income_gt50k)).decode("utf-8")
        inc_absent = base64.b64encode(create_sketch(neg_income_le50k)).decode("utf-8")
        f.write(f"income>50k,{inc_present},{inc_absent}\n")

    print("✅ Created: tests/fixtures/target_yes_3col.csv")
    print("✅ Created: tests/fixtures/target_no_3col.csv")

    # Mode 2 Format 2: Dual CSV with 2 columns
    # target_yes_2col.csv
    with open("tests/fixtures/target_yes_2col.csv", "w") as f:
        f.write("identifier,sketch\n")

        # total
        total_sketch = base64.b64encode(create_sketch(pos_ids)).decode("utf-8")
        f.write(f"total,{total_sketch}\n")

        # age>30 (only those who have it)
        age_sketch = base64.b64encode(create_sketch(pos_age_gt30)).decode("utf-8")
        f.write(f"age>30,{age_sketch}\n")

        # income>50k (only those who have it)
        inc_sketch = base64.b64encode(create_sketch(pos_income_gt50k)).decode("utf-8")
        f.write(f"income>50k,{inc_sketch}\n")

    # target_no_2col.csv
    with open("tests/fixtures/target_no_2col.csv", "w") as f:
        f.write("identifier,sketch\n")

        # total
        total_sketch = base64.b64encode(create_sketch(neg_ids)).decode("utf-8")
        f.write(f"total,{total_sketch}\n")

        # age>30
        age_sketch = base64.b64encode(create_sketch(neg_age_gt30)).decode("utf-8")
        f.write(f"age>30,{age_sketch}\n")

        # income>50k
        inc_sketch = base64.b64encode(create_sketch(neg_income_gt50k)).decode("utf-8")
        f.write(f"income>50k,{inc_sketch}\n")

    print("✅ Created: tests/fixtures/target_yes_2col.csv")
    print("✅ Created: tests/fixtures/target_no_2col.csv")

    # Mode 1: Single CSV with 2 columns
    with open("tests/fixtures/features_single_2col.csv", "w") as f:
        f.write("identifier,sketch\n")

        # target_yes
        target_yes_sketch = base64.b64encode(create_sketch(pos_ids)).decode("utf-8")
        f.write(f"target_yes,{target_yes_sketch}\n")

        # target_no
        target_no_sketch = base64.b64encode(create_sketch(neg_ids)).decode("utf-8")
        f.write(f"target_no,{target_no_sketch}\n")

        # age>30 (all who have it, regardless of target)
        age_all = pos_age_gt30 + neg_age_gt30
        age_sketch = base64.b64encode(create_sketch(age_all)).decode("utf-8")
        f.write(f"age>30,{age_sketch}\n")

        # income>50k (all who have it, regardless of target)
        inc_all = pos_income_gt50k + neg_income_gt50k
        inc_sketch = base64.b64encode(create_sketch(inc_all)).decode("utf-8")
        f.write(f"income>50k,{inc_sketch}\n")

    print("✅ Created: tests/fixtures/features_single_2col.csv")

    print("\n✨ All test fixture CSV files generated successfully!")
    print("\nSummary:")
    print("- Positive class: 400 samples (40%)")
    print("- Negative class: 600 samples (60%)")
    print("- Features: age>30, income>50k")
    print("- Sketch size: k=4096")


if __name__ == "__main__":
    create_test_csvs()
