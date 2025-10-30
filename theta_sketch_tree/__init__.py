"""
Theta Sketch Decision Tree Classifier

A sklearn-compatible decision tree classifier that trains on theta sketches
but performs inference on binary tabular data.

Key Features:
- Trains on theta sketches (privacy-preserving set summaries)
- Infers on binary feature data (0/1 values)
- Multiple split criteria (Gini, Entropy, Gain Ratio, Binomial, Chi-Square)
- Missing value handling via majority path method
- Class weighting for imbalanced data
- Pre and post-pruning support
"""

__version__ = "0.1.0-dev"
__author__ = "Vijay Sankar Rajan"
__license__ = "MIT"

# Main imports (will be uncommented as modules are implemented)
# from .classifier import ThetaSketchDecisionTreeClassifier
# from .tree_structure import Tree, TreeNode

__all__ = [
    # 'ThetaSketchDecisionTreeClassifier',
    # 'Tree',
    # 'TreeNode',
]
