"""
Tests for sketch_loader.

Tests both 2-column and 3-column CSV formats, single and dual CSV modes.
"""

import pytest
import csv
import base64
from pathlib import Path
from datasketches import update_theta_sketch
from theta_sketch_tree.sketch_loader import SketchLoader


@pytest.fixture
def sample_sketches():
    """Create sample theta sketches for testing."""
    # Create sketches with different data
    sketch_yes_total = update_theta_sketch()
    for i in range(100):
        sketch_yes_total.update(f"user_{i}")

    sketch_no_total = update_theta_sketch()
    for i in range(50, 150):
        sketch_no_total.update(f"user_{i}")

    sketch_yes_age_present = update_theta_sketch()
    for i in range(30):
        sketch_yes_age_present.update(f"user_{i}")

    sketch_yes_age_absent = update_theta_sketch()
    for i in range(30, 100):
        sketch_yes_age_absent.update(f"user_{i}")

    sketch_no_age_present = update_theta_sketch()
    for i in range(70, 100):
        sketch_no_age_present.update(f"user_{i}")

    sketch_no_age_absent = update_theta_sketch()
    for i in range(50, 70):
        sketch_no_age_absent.update(f"user_{i}")

    return {
        "yes_total": sketch_yes_total,
        "no_total": sketch_no_total,
        "yes_age_present": sketch_yes_age_present,
        "yes_age_absent": sketch_yes_age_absent,
        "no_age_present": sketch_no_age_present,
        "no_age_absent": sketch_no_age_absent,
    }


@pytest.fixture
def csv_3column_positive(tmp_path, sample_sketches):
    """Create 3-column CSV file for positive class."""
    csv_file = tmp_path / "target_yes.csv"

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["identifier", "sketch_feature_present", "sketch_feature_absent"])
        writer.writerow(
            [
                "total",
                base64.b64encode(sample_sketches["yes_total"].compact().serialize()).decode(),
                base64.b64encode(sample_sketches["yes_total"].compact().serialize()).decode(),
            ]
        )
        writer.writerow(
            [
                "age>30",
                base64.b64encode(sample_sketches["yes_age_present"].compact().serialize()).decode(),
                base64.b64encode(sample_sketches["yes_age_absent"].compact().serialize()).decode(),
            ]
        )

    return str(csv_file)


@pytest.fixture
def csv_3column_negative(tmp_path, sample_sketches):
    """Create 3-column CSV file for negative class."""
    csv_file = tmp_path / "target_no.csv"

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["identifier", "sketch_feature_present", "sketch_feature_absent"])
        writer.writerow(
            [
                "total",
                base64.b64encode(sample_sketches["no_total"].compact().serialize()).decode(),
                base64.b64encode(sample_sketches["no_total"].compact().serialize()).decode(),
            ]
        )
        writer.writerow(
            [
                "age>30",
                base64.b64encode(sample_sketches["no_age_present"].compact().serialize()).decode(),
                base64.b64encode(sample_sketches["no_age_absent"].compact().serialize()).decode(),
            ]
        )

    return str(csv_file)


@pytest.fixture
def csv_2column_positive(tmp_path, sample_sketches):
    """Create 2-column CSV file for positive class."""
    csv_file = tmp_path / "target_yes_2col.csv"

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["identifier", "sketch"])
        writer.writerow(
            [
                "total",
                base64.b64encode(sample_sketches["yes_total"].compact().serialize()).decode(),
            ]
        )
        writer.writerow(
            [
                "age>30",
                base64.b64encode(sample_sketches["yes_age_present"].compact().serialize()).decode(),
            ]
        )

    return str(csv_file)


@pytest.fixture
def csv_2column_negative(tmp_path, sample_sketches):
    """Create 2-column CSV file for negative class."""
    csv_file = tmp_path / "target_no_2col.csv"

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["identifier", "sketch"])
        writer.writerow(
            [
                "total",
                base64.b64encode(sample_sketches["no_total"].compact().serialize()).decode(),
            ]
        )
        writer.writerow(
            [
                "age>30",
                base64.b64encode(sample_sketches["no_age_present"].compact().serialize()).decode(),
            ]
        )

    return str(csv_file)


@pytest.fixture
def csv_single_file(tmp_path, sample_sketches):
    """Create single CSV file with both target classes (Mode 1)."""
    csv_file = tmp_path / "features.csv"

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["identifier", "sketch_feature_present", "sketch_feature_absent"])
        # Positive class
        writer.writerow(
            [
                "target_yes_total",
                base64.b64encode(sample_sketches["yes_total"].compact().serialize()).decode(),
                base64.b64encode(sample_sketches["yes_total"].compact().serialize()).decode(),
            ]
        )
        writer.writerow(
            [
                "target_yes_age>30",
                base64.b64encode(sample_sketches["yes_age_present"].compact().serialize()).decode(),
                base64.b64encode(sample_sketches["yes_age_absent"].compact().serialize()).decode(),
            ]
        )
        # Negative class
        writer.writerow(
            [
                "target_no_total",
                base64.b64encode(sample_sketches["no_total"].compact().serialize()).decode(),
                base64.b64encode(sample_sketches["no_total"].compact().serialize()).decode(),
            ]
        )
        writer.writerow(
            [
                "target_no_age>30",
                base64.b64encode(sample_sketches["no_age_present"].compact().serialize()).decode(),
                base64.b64encode(sample_sketches["no_age_absent"].compact().serialize()).decode(),
            ]
        )

    return str(csv_file)


class TestSketchLoader:
    """Test suite for SketchLoader."""

    def test_mode2_3column_csv(self, csv_3column_positive, csv_3column_negative):
        """Test Mode 2: Dual CSV with 3-column format (RECOMMENDED)."""
        loader = SketchLoader(encoding="base64")
        sketch_data = loader.load(
            positive_csv=csv_3column_positive, negative_csv=csv_3column_negative
        )

        # Verify structure
        assert "positive" in sketch_data
        assert "negative" in sketch_data
        assert "total" in sketch_data["positive"]
        assert "total" in sketch_data["negative"]
        assert "age>30" in sketch_data["positive"]
        assert "age>30" in sketch_data["negative"]

        # Verify tuple format for features
        assert isinstance(sketch_data["positive"]["age>30"], tuple)
        assert len(sketch_data["positive"]["age>30"]) == 2
        assert isinstance(sketch_data["negative"]["age>30"], tuple)
        assert len(sketch_data["negative"]["age>30"]) == 2

        # Verify sketches have data
        assert sketch_data["positive"]["total"].get_estimate() > 0
        assert sketch_data["negative"]["total"].get_estimate() > 0

    def test_mode2_2column_csv(self, csv_2column_positive, csv_2column_negative):
        """Test Mode 2: Dual CSV with 2-column format (legacy)."""
        loader = SketchLoader(encoding="base64")
        sketch_data = loader.load(
            positive_csv=csv_2column_positive, negative_csv=csv_2column_negative
        )

        # Verify structure
        assert "positive" in sketch_data
        assert "negative" in sketch_data
        assert "total" in sketch_data["positive"]
        assert "age>30" in sketch_data["positive"]

        # Verify single sketch format (not tuple)
        assert not isinstance(sketch_data["positive"]["age>30"], tuple)
        assert sketch_data["positive"]["age>30"].get_estimate() > 0

    def test_mode1_single_csv(self, csv_single_file):
        """Test Mode 1: Single CSV with both classes."""
        loader = SketchLoader(encoding="base64")
        sketch_data = loader.load(
            csv_path=csv_single_file,
            target_positive="target_yes",
            target_negative="target_no",
        )

        # Verify structure
        assert "positive" in sketch_data
        assert "negative" in sketch_data
        assert "total" in sketch_data["positive"]
        assert "total" in sketch_data["negative"]
        assert "age>30" in sketch_data["positive"]
        assert "age>30" in sketch_data["negative"]

        # Verify tuple format (3-column CSV)
        assert isinstance(sketch_data["positive"]["age>30"], tuple)
        assert len(sketch_data["positive"]["age>30"]) == 2

    def test_validation_missing_total(self, tmp_path, sample_sketches):
        """Test validation error when 'total' key is missing."""
        csv_file = tmp_path / "missing_total.csv"

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["identifier", "sketch_feature_present", "sketch_feature_absent"])
            writer.writerow(
                [
                    "age>30",
                    base64.b64encode(sample_sketches["yes_age_present"].compact().serialize()).decode(),
                    base64.b64encode(sample_sketches["yes_age_absent"].compact().serialize()).decode(),
                ]
            )

        loader = SketchLoader()

        with pytest.raises(ValueError, match="'total' key not found"):
            loader.load(positive_csv=str(csv_file), negative_csv=str(csv_file))

    def test_validation_mixed_modes(self, csv_3column_positive, csv_single_file):
        """Test error when both modes are specified."""
        loader = SketchLoader()

        with pytest.raises(ValueError, match="Cannot specify both"):
            loader.load(
                csv_path=csv_single_file,
                positive_csv=csv_3column_positive,
                negative_csv=csv_3column_positive,
            )

    def test_validation_no_mode(self):
        """Test error when no mode is specified."""
        loader = SketchLoader()

        with pytest.raises(ValueError, match="Must specify either"):
            loader.load()

    def test_validation_mode2_incomplete(self, csv_3column_positive):
        """Test error when Mode 2 is incomplete (only one CSV)."""
        loader = SketchLoader()

        with pytest.raises(ValueError, match="Mode 2 requires both"):
            loader.load(positive_csv=csv_3column_positive)

    def test_validation_mode1_incomplete(self, csv_single_file):
        """Test error when Mode 1 is incomplete (missing target identifiers)."""
        loader = SketchLoader()

        with pytest.raises(ValueError, match="Mode 1 requires both"):
            loader.load(csv_path=csv_single_file, target_positive="target_yes")

    def test_file_not_found(self):
        """Test error when CSV file doesn't exist."""
        loader = SketchLoader()

        with pytest.raises(FileNotFoundError):
            loader.load(
                positive_csv="/nonexistent/file.csv",
                negative_csv="/nonexistent/file2.csv",
            )

    def test_invalid_csv_format(self, tmp_path):
        """Test error when CSV has invalid number of columns."""
        csv_file = tmp_path / "invalid.csv"

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["col1", "col2", "col3", "col4"])  # 4 columns
            writer.writerow(["data1", "data2", "data3", "data4"])

        loader = SketchLoader()

        with pytest.raises(ValueError, match="Invalid CSV format"):
            loader.load(positive_csv=str(csv_file), negative_csv=str(csv_file))

    def test_hex_encoding(self, tmp_path, sample_sketches):
        """Test hex encoding support in sketch deserialization."""
        csv_file = tmp_path / "hex_encoded.csv"

        # Use hex encoding instead of base64
        sketch_hex = sample_sketches["yes_total"].compact().serialize().hex()

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["identifier", "sketch"])
            writer.writerow(["total", sketch_hex])

        loader = SketchLoader(encoding="hex")
        sketch_data = loader.load(
            positive_csv=str(csv_file), negative_csv=str(csv_file)
        )

        # Verify the sketch was properly deserialized
        assert sketch_data["positive"]["total"].get_estimate() > 0

    def test_unsupported_encoding(self, tmp_path, sample_sketches):
        """Test error when unsupported encoding is used."""
        csv_file = tmp_path / "test.csv"

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["identifier", "sketch"])
            writer.writerow(["total", "dummy_data"])

        loader = SketchLoader(encoding="unsupported")

        with pytest.raises(ValueError, match="Unsupported encoding"):
            loader.load(positive_csv=str(csv_file), negative_csv=str(csv_file))

    def test_malformed_csv_rows(self, tmp_path, sample_sketches):
        """Test handling of empty and malformed CSV rows."""
        csv_file = tmp_path / "malformed.csv"

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["identifier", "sketch_feature_present", "sketch_feature_absent"])
            writer.writerow([
                "total",
                base64.b64encode(sample_sketches["yes_total"].compact().serialize()).decode(),
                base64.b64encode(sample_sketches["yes_total"].compact().serialize()).decode(),
            ])
            # Empty row
            writer.writerow([])
            # Malformed row (wrong number of columns)
            writer.writerow(["incomplete"])
            # Another valid row
            writer.writerow([
                "age>30",
                base64.b64encode(sample_sketches["yes_age_present"].compact().serialize()).decode(),
                base64.b64encode(sample_sketches["yes_age_absent"].compact().serialize()).decode(),
            ])

        loader = SketchLoader()
        sketch_data = loader.load(
            positive_csv=str(csv_file), negative_csv=str(csv_file)
        )

        # Should still work despite malformed rows
        assert "total" in sketch_data["positive"]
        assert "age>30" in sketch_data["positive"]

    def test_missing_total_in_negative_csv_mode2(self, tmp_path, sample_sketches):
        """Test error when 'total' is missing in negative CSV (Mode 2)."""
        # Create positive CSV with 'total'
        pos_csv = tmp_path / "pos_with_total.csv"
        with open(pos_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["identifier", "sketch_feature_present", "sketch_feature_absent"])
            writer.writerow([
                "total",
                base64.b64encode(sample_sketches["yes_total"].compact().serialize()).decode(),
                base64.b64encode(sample_sketches["yes_total"].compact().serialize()).decode(),
            ])

        # Create negative CSV without 'total'
        neg_csv = tmp_path / "neg_without_total.csv"
        with open(neg_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["identifier", "sketch_feature_present", "sketch_feature_absent"])
            writer.writerow([
                "age>30",  # Only feature, no 'total'
                base64.b64encode(sample_sketches["no_age_present"].compact().serialize()).decode(),
                base64.b64encode(sample_sketches["no_age_absent"].compact().serialize()).decode(),
            ])

        loader = SketchLoader()
        with pytest.raises(ValueError, match="'total' key not found in negative CSV"):
            loader.load(positive_csv=str(pos_csv), negative_csv=str(neg_csv))

    def test_missing_total_in_mode1_positive(self, tmp_path, sample_sketches):
        """Test error when 'total' is missing for positive target in Mode 1."""
        csv_file = tmp_path / "mode1_missing_pos_total.csv"

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["identifier", "sketch_feature_present", "sketch_feature_absent"])
            # Only negative total, missing positive total
            writer.writerow([
                "target_no_total",
                base64.b64encode(sample_sketches["no_total"].compact().serialize()).decode(),
                base64.b64encode(sample_sketches["no_total"].compact().serialize()).decode(),
            ])
            writer.writerow([
                "target_yes_age>30",  # Feature for positive, but no total
                base64.b64encode(sample_sketches["yes_age_present"].compact().serialize()).decode(),
                base64.b64encode(sample_sketches["yes_age_absent"].compact().serialize()).decode(),
            ])

        loader = SketchLoader()
        with pytest.raises(ValueError, match="'total' key not found for target 'target_yes'"):
            loader.load(
                csv_path=str(csv_file),
                target_positive="target_yes",
                target_negative="target_no"
            )

    def test_missing_total_in_mode1_negative(self, tmp_path, sample_sketches):
        """Test error when 'total' is missing for negative target in Mode 1."""
        csv_file = tmp_path / "mode1_missing_neg_total.csv"

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["identifier", "sketch_feature_present", "sketch_feature_absent"])
            # Only positive total, missing negative total
            writer.writerow([
                "target_yes_total",
                base64.b64encode(sample_sketches["yes_total"].compact().serialize()).decode(),
                base64.b64encode(sample_sketches["yes_total"].compact().serialize()).decode(),
            ])
            writer.writerow([
                "target_no_age>30",  # Feature for negative, but no total
                base64.b64encode(sample_sketches["no_age_present"].compact().serialize()).decode(),
                base64.b64encode(sample_sketches["no_age_absent"].compact().serialize()).decode(),
            ])

        loader = SketchLoader()
        with pytest.raises(ValueError, match="'total' key not found for target 'target_no'"):
            loader.load(
                csv_path=str(csv_file),
                target_positive="target_yes",
                target_negative="target_no"
            )
