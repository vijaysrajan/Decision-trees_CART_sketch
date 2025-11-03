"""
CSV sketch loader.

Loads and deserializes theta sketches from dual CSV files.

Supported classification modes:
- Dual-Class Mode: positive.csv + negative.csv (best accuracy)
- One-vs-All Mode: positive.csv + total.csv (healthcare, CTR)

CSV format (mandatory):
- 3 columns: identifier, sketch_feature_present, sketch_feature_absent
"""

import csv
import base64
from pathlib import Path
from typing import Optional, Dict, Union, Tuple, Any
from datasketches import compact_theta_sketch


class SketchLoader:
    """
    Load theta sketches from CSV file(s).

    Supports two input modes:
    - Mode 1: Single CSV (performs intersections)
    - Mode 2: Dual CSV (pre-intersected, recommended for accuracy)
    """

    def __init__(self, encoding: str = "base64"):
        """Initialize loader."""
        self.encoding = encoding

    def _deserialize_sketch(self, sketch_bytes: str) -> compact_theta_sketch:
        """
        Deserialize a theta sketch from encoded bytes.

        Parameters
        ----------
        sketch_bytes : str
            Base64 or hex encoded sketch bytes.

        Returns
        -------
        compact_theta_sketch
            Deserialized theta sketch.
        """
        if self.encoding == "base64":
            decoded = base64.b64decode(sketch_bytes)
        elif self.encoding == "hex":
            decoded = bytes.fromhex(sketch_bytes)
        else:
            raise ValueError(f"Unsupported encoding: {self.encoding}")

        return compact_theta_sketch.deserialize(decoded)

    def _parse_csv_file(
        self, csv_path: str, target_identifier: Optional[str] = None
    ) -> Dict[str, Union[compact_theta_sketch, Tuple[compact_theta_sketch, compact_theta_sketch]]]:
        """
        Parse a single CSV file and return sketch data.

        Parameters
        ----------
        csv_path : str
            Path to CSV file.
        target_identifier : str, optional
            If provided, only parse rows with this identifier prefix.

        Returns
        -------
        dict
            Dictionary mapping feature identifiers to sketches or tuples.
        """
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        result = {}

        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)

            # Auto-detect format: 2-column or 3-column
            if len(header) == 2:
                # 2-column format: identifier, sketch
                is_three_column = False
            elif len(header) == 3:
                # 3-column format: identifier, sketch_feature_present, sketch_feature_absent
                is_three_column = True
            else:
                raise ValueError(
                    f"Invalid CSV format. Expected 2 or 3 columns, got {len(header)}"
                )

            for row in reader:
                if not row or len(row) != len(header):
                    continue  # Skip empty or malformed rows

                identifier = row[0].strip()

                # If target_identifier is specified, filter by prefix
                if target_identifier and not identifier.startswith(
                    target_identifier
                ):
                    continue

                # Remove target prefix if present (e.g., "target_yes_age>30" -> "age>30")
                if target_identifier and identifier.startswith(target_identifier):
                    # Remove prefix and separator
                    identifier = identifier[len(target_identifier) :].lstrip("_")

                if is_three_column:
                    # 3-column: (sketch_feature_present, sketch_feature_absent)
                    sketch_feature_present = self._deserialize_sketch(row[1].strip())
                    sketch_feature_absent = self._deserialize_sketch(row[2].strip())

                    # Special case: 'total' should always be a single sketch
                    if identifier == "total":
                        result[identifier] = sketch_feature_present
                    else:
                        result[identifier] = (sketch_feature_present, sketch_feature_absent)
                else:
                    # 2-column: single sketch
                    sketch = self._deserialize_sketch(row[1].strip())
                    result[identifier] = sketch

        return result

    def load(
        self,
        positive_csv: Optional[str] = None,
        negative_csv: Optional[str] = None,
        csv_path: Optional[str] = None,
        target_positive: Optional[str] = None,
        target_negative: Optional[str] = None,
    ) -> Dict[str, Dict[str, Union[Any, Tuple[Any, Any]]]]:
        """
        Load sketches from CSV file(s) into unified data structure.

        Mode 1: load(csv_path='features.csv', target_positive='yes', target_negative='no')
        Mode 2: load(positive_csv='target_yes.csv', negative_csv='target_no.csv')

        Parameters
        ----------
        positive_csv : str, optional
            Positive class CSV (Mode 2). Must be used with negative_csv.
        negative_csv : str, optional
            Negative class CSV (Mode 2). Must be used with positive_csv.
        csv_path : str, optional
            Single CSV file (Mode 1). Mutually exclusive with positive_csv/negative_csv.
        target_positive : str, optional
            Positive class identifier in CSV (Mode 1), e.g., "target_yes"
        target_negative : str, optional
            Negative class identifier in CSV (Mode 1), e.g., "target_no"

        Returns
        -------
        sketch_data : dict
            Dictionary with 'positive' and 'negative' keys, each containing:
            - 'total': ThetaSketch for class population
            - '<feature>': Tuple (sketch_feature_present, sketch_feature_absent) or single ThetaSketch

            Example:
            {
                'positive': {
                    'total': <ThetaSketch>,
                    'age>30': (<sketch_feature_present>, <sketch_feature_absent>),
                    'income>50k': (<sketch_feature_present>, <sketch_feature_absent>)
                },
                'negative': { ... }
            }

        Raises
        ------
        ValueError
            If both modes are specified or if mode parameters are incomplete.
        FileNotFoundError
            If CSV files don't exist.

        Notes
        -----
        Auto-detects 2-column vs 3-column CSV format:
        - 2 columns: identifier, sketch
        - 3 columns: identifier, sketch_feature_present, sketch_feature_absent (RECOMMENDED)

        See docs/02_low_level_design.md for full specifications.
        """
        # Validate input modes
        mode_1 = csv_path is not None
        mode_2 = positive_csv is not None or negative_csv is not None

        if mode_1 and mode_2:
            raise ValueError(
                "Cannot specify both csv_path (Mode 1) and "
                "positive_csv/negative_csv (Mode 2). Choose one mode."
            )

        if not mode_1 and not mode_2:
            raise ValueError(
                "Must specify either csv_path (Mode 1) or "
                "positive_csv/negative_csv (Mode 2)."
            )

        # Mode 2: Dual CSV
        if mode_2:
            if positive_csv is None or negative_csv is None:
                raise ValueError(
                    "Mode 2 requires both positive_csv and negative_csv."
                )

            # Parse both CSV files
            positive_data = self._parse_csv_file(positive_csv)
            negative_data = self._parse_csv_file(negative_csv)

            # Validate that 'total' exists in both
            if "total" not in positive_data:
                raise ValueError(
                    f"'total' key not found in positive CSV: {positive_csv}"
                )
            if "total" not in negative_data:
                raise ValueError(
                    f"'total' key not found in negative CSV: {negative_csv}"
                )

            return {"positive": positive_data, "negative": negative_data}

        # Mode 1: Single CSV
        else:
            if target_positive is None or target_negative is None:
                raise ValueError(
                    "Mode 1 requires both target_positive and target_negative "
                    "identifiers."
                )

            # Parse CSV for positive class
            positive_data = self._parse_csv_file(csv_path, target_positive)

            # Parse CSV for negative class
            negative_data = self._parse_csv_file(csv_path, target_negative)

            # Validate that 'total' exists in both
            if "total" not in positive_data:
                raise ValueError(
                    f"'total' key not found for target '{target_positive}' "
                    f"in CSV: {csv_path}"
                )
            if "total" not in negative_data:
                raise ValueError(
                    f"'total' key not found for target '{target_negative}' "
                    f"in CSV: {csv_path}"
                )

            return {"positive": positive_data, "negative": negative_data}
