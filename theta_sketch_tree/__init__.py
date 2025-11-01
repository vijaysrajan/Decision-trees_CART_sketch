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

from .sketch_loader import SketchLoader
from .config_parser import ConfigParser
from typing import Dict, Optional, Union, Tuple, Any


def load_sketches(
    positive_csv: Optional[str] = None,
    negative_csv: Optional[str] = None,
    csv_path: Optional[str] = None,
    target_positive: Optional[str] = None,
    target_negative: Optional[str] = None,
    encoding: str = "base64",
) -> Dict[str, Dict[str, Union[Any, Tuple[Any, Any]]]]:
    """
    Load theta sketches from CSV file(s).

    Helper function that wraps SketchLoader for convenient data loading.

    Parameters
    ----------
    positive_csv : str, optional
        Path to CSV with positive class sketches (Mode 2).
    negative_csv : str, optional
        Path to CSV with negative class sketches (Mode 2).
    csv_path : str, optional
        Path to single CSV with all sketches (Mode 1).
    target_positive : str, optional
        Positive class identifier for Mode 1, e.g., "target_yes"
    target_negative : str, optional
        Negative class identifier for Mode 1, e.g., "target_no"
    encoding : str, default='base64'
        Sketch bytes encoding ('base64' or 'hex')

    Returns
    -------
    sketch_data : dict
        Dictionary with 'positive' and 'negative' keys containing sketch data.

    Examples
    --------
    >>> from theta_sketch_tree import load_sketches
    >>> # Mode 2: Dual CSV (recommended)
    >>> sketch_data = load_sketches(
    ...     positive_csv='target_yes.csv',
    ...     negative_csv='target_no.csv'
    ... )
    >>> # Mode 1: Single CSV
    >>> sketch_data = load_sketches(
    ...     csv_path='features.csv',
    ...     target_positive='target_yes',
    ...     target_negative='target_no'
    ... )

    See Also
    --------
    load_config : Load configuration from YAML file
    ThetaSketchDecisionTreeClassifier.fit : Fit model with sketch data
    """
    loader = SketchLoader(encoding=encoding)
    return loader.load(
        positive_csv=positive_csv,
        negative_csv=negative_csv,
        csv_path=csv_path,
        target_positive=target_positive,
        target_negative=target_negative,
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Helper function that wraps ConfigParser for convenient config loading.

    Parameters
    ----------
    config_path : str
        Path to YAML or JSON config file.

    Returns
    -------
    config : dict
        Configuration dictionary with keys:
        - 'targets': {'positive': str, 'negative': str}
        - 'hyperparameters': {param_name: param_value, ...}
        - 'feature_mapping': {feature_name: column_index, ...}

    Examples
    --------
    >>> from theta_sketch_tree import load_config
    >>> config = load_config('config.yaml')
    >>> print(config.keys())
    dict_keys(['targets', 'hyperparameters', 'feature_mapping'])
    >>>
    >>> # Use with classifier
    >>> from theta_sketch_tree import ThetaSketchDecisionTreeClassifier
    >>> clf = ThetaSketchDecisionTreeClassifier(**config['hyperparameters'])
    >>> clf.fit(sketch_data, config['feature_mapping'])

    See Also
    --------
    load_sketches : Load sketch data from CSV files
    ThetaSketchDecisionTreeClassifier.fit : Fit model with config data
    """
    parser = ConfigParser()
    return parser.load(config_path)


__all__ = [
    # 'ThetaSketchDecisionTreeClassifier',
    # 'Tree',
    # 'TreeNode',
    "load_sketches",
    "load_config",
    "SketchLoader",
    "ConfigParser",
]
