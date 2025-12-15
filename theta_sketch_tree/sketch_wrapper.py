"""
Wrapper for Apache DataSketches that adds intersection method for tree builder compatibility.
"""

from datasketches import theta_intersection


class ThetaSketchWrapper:
    """
    Wrapper for Apache DataSketches that adds intersection method for tree builder compatibility.

    This wrapper provides the intersection method that the core tree building logic expects,
    while delegating other operations to the underlying DataSketches object.
    """

    def __init__(self, sketch):
        """Initialize wrapper with a sketch object."""
        self._sketch = sketch

    def get_estimate(self):
        """Return estimated cardinality."""
        return self._sketch.get_estimate()

    def update(self, item):
        """Add item to sketch."""
        return self._sketch.update(item)

    def intersection(self, other):
        """Compute intersection with another sketch using Apache DataSketches API."""
        if isinstance(other, ThetaSketchWrapper):
            other_sketch = other._sketch
        else:
            other_sketch = other

        intersector = theta_intersection()
        intersector.update(self._sketch)
        intersector.update(other_sketch)
        result = intersector.get_result()

        return ThetaSketchWrapper(result)

    def __getattr__(self, name):
        """Delegate other methods to the underlying sketch."""
        return getattr(self._sketch, name)