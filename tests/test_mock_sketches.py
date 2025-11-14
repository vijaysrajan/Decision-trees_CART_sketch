"""
Mock ThetaSketch classes for testing purposes.

This module provides simple mock implementations of ThetaSketch functionality
to enable unit testing without requiring actual sketch libraries.
"""

class MockThetaSketch:
    """
    Mock ThetaSketch for testing.

    Simulates the basic ThetaSketch interface needed by the decision tree algorithm.
    """

    def __init__(self, estimated_count: float, identifier: str = "mock"):
        """
        Initialize mock sketch with a fixed estimated count.

        Parameters
        ----------
        estimated_count : float
            The count this mock sketch should return from get_estimate()
        identifier : str
            Human-readable identifier for debugging
        """
        self.estimated_count = estimated_count
        self.identifier = identifier

    def get_estimate(self) -> float:
        """Return the mock estimated count."""
        return self.estimated_count

    def intersection(self, other: 'MockThetaSketch') -> 'MockThetaSketch':
        """
        Mock intersection operation.

        For testing, we'll simulate intersection by taking the minimum count
        (this approximates real sketch intersection behavior).
        """
        if not isinstance(other, MockThetaSketch):
            raise TypeError(f"Can only intersect with MockThetaSketch, got {type(other)}")

        # Mock intersection: take minimum (approximates real behavior)
        intersected_count = min(self.estimated_count, other.estimated_count) * 0.8  # Simulated reduction

        return MockThetaSketch(
            estimated_count=intersected_count,
            identifier=f"intersection({self.identifier},{other.identifier})"
        )

    def __str__(self):
        """String representation for debugging."""
        return f"MockThetaSketch({self.estimated_count}, {self.identifier})"

    def __repr__(self):
        """Detailed representation for debugging."""
        return f"MockThetaSketch(estimated_count={self.estimated_count}, identifier='{self.identifier}')"


def create_mock_sketch_data():
    """
    Create realistic mock sketch data for testing.

    Returns a properly structured sketch_data dict that can be used
    with ThetaSketchDecisionTreeClassifier.fit().

    Returns
    -------
    dict
        Sketch data with proper MockThetaSketch objects
    """
    # Create base class sketches
    pos_total = MockThetaSketch(1000, "pos_total")
    neg_total = MockThetaSketch(800, "neg_total")

    # Create feature sketches with realistic distributions
    # Feature: age>30
    pos_age_present = MockThetaSketch(700, "pos_age_present")  # 70% of positive class has age>30
    pos_age_absent = MockThetaSketch(300, "pos_age_absent")    # 30% of positive class has age<=30

    neg_age_present = MockThetaSketch(400, "neg_age_present")  # 50% of negative class has age>30
    neg_age_absent = MockThetaSketch(400, "neg_age_absent")    # 50% of negative class has age<=30

    # Feature: income>50k
    pos_income_present = MockThetaSketch(600, "pos_income_present")  # 60% of positive class has income>50k
    pos_income_absent = MockThetaSketch(400, "pos_income_absent")    # 40% of positive class has income<=50k

    neg_income_present = MockThetaSketch(200, "neg_income_present")  # 25% of negative class has income>50k
    neg_income_absent = MockThetaSketch(600, "neg_income_absent")    # 75% of negative class has income<=50k

    return {
        'positive': {
            'total': pos_total,
            'age>30': (pos_age_present, pos_age_absent),
            'income>50k': (pos_income_present, pos_income_absent),
        },
        'negative': {
            'total': neg_total,
            'age>30': (neg_age_present, neg_age_absent),
            'income>50k': (neg_income_present, neg_income_absent),
        }
    }


def create_feature_mapping():
    """Create feature mapping for the mock data."""
    return {
        'age>30': 0,
        'income>50k': 1,
    }