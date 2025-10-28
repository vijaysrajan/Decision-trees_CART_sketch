"""
CSV sketch loader.

Loads and deserializes theta sketches from CSV files.
"""

# TODO: Implement SketchLoader class
# See docs/02_low_level_design.md for specifications


class SketchLoader:
    """Load theta sketches from CSV file."""

    def __init__(self, encoding: str = 'base64'):
        """Initialize loader."""
        self.encoding = encoding

    def load(self, csv_path: str, target_positive: str, target_negative: str):
        """Load sketches from CSV."""
        raise NotImplementedError("To be implemented in Week 1")
