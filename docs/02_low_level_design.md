# Low-Level Design Document
## Theta Sketch Decision Tree Classifier - Detailed Class Design

---

## Table of Contents
1. [Class Hierarchy Overview](#class-hierarchy-overview)
2. [Core Classes](#core-classes)
3. [Data Loading Classes](#data-loading-classes)
4. [Splitting & Criteria Classes](#splitting--criteria-classes)
5. [Inference Classes](#inference-classes)
6. [Pruning Classes](#pruning-classes)
7. [Utility Classes](#utility-classes)
8. [Interaction Diagrams](#interaction-diagrams)

---

## 1. Class Hierarchy Overview

```
sklearn.base.BaseEstimator
sklearn.base.ClassifierMixin
    │
    └─► ThetaSketchDecisionTreeClassifier (main API)
            │
            ├─► Uses: SketchLoader
            ├─► Uses: ConfigParser
            ├─► Uses: TreeBuilder
            ├─► Uses: FeatureTransformer
            ├─► Uses: TreeTraverser
            └─► Uses: FeatureImportanceCalculator

Tree Data Structures:
    Tree
    └─► TreeNode (recursive structure)

Splitting:
    SplitEvaluator
    ├─► Uses: BaseCriterion (abstract)
    │       ├─► GiniCriterion
    │       ├─► EntropyCriterion
    │       ├─► GainRatioCriterion
    │       ├─► BinomialCriterion
    │       └─► ChiSquareCriterion
    └─► Uses: SketchCache

Pruning:
    BasePruner (abstract)
    ├─► PrePruner
    └─► PostPruner

Utilities:
    SketchCache
    FeatureImportanceCalculator
    MetricsCalculator
    MissingValueHandler
```

---

## 2. Core Classes

### 2.1 ThetaSketchDecisionTreeClassifier

**Responsibility**: Main API class, sklearn-compatible decision tree classifier

```python
from typing import Dict, Optional, Union, Literal, Any
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin

class ThetaSketchDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    CART Decision Tree Classifier trained on theta sketches.

    Attributes (Public - set during fit)
    -----------------------------------
    classes_ : NDArray[np.int64]
        Array of class labels [0, 1]
    n_classes_ : int
        Number of classes (always 2 for binary classification)
    n_features_in_ : int
        Number of features seen during fit (from feature_mapping)
    feature_names_in_ : NDArray[np.str_]
        Feature names (from feature_mapping keys)
    tree_ : Tree
        The trained tree structure
    feature_importances_ : NDArray[np.float64]
        Feature importance scores (computed lazily)

    Attributes (Private - internal state)
    -------------------------------------
    _sketch_dict : Dict[str, Dict[str, ThetaSketch]]
        Loaded theta sketches from CSV
    _feature_mapping : Dict[str, tuple]
        Feature name → (column_index, lambda_function)
    _is_fitted : bool
        Whether fit() has been called
    _tree_builder : TreeBuilder
        Tree construction helper
    _feature_transformer : FeatureTransformer
        Raw data → binary features transformer
    _tree_traverser : TreeTraverser
        Inference navigation helper
    _sketch_cache : SketchCache
        Cache for sketch operations
    """

    def __init__(
        self,
        # Split criteria
        criterion: Literal['gini', 'entropy', 'gain_ratio', 'binomial', 'binomial_chi'] = 'gini',

        # Tree structure
        splitter: Literal['best', 'random'] = 'best',
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float, Literal['sqrt', 'log2']]] = None,

        # Statistical tests
        min_pvalue: float = 0.05,
        use_bonferroni: bool = True,

        # Imbalanced data
        class_weight: Optional[Union[Literal['balanced'], Dict[int, float]]] = None,
        use_weighted_gini: bool = True,

        # Missing values
        missing_value_strategy: Literal['majority', 'zero', 'error'] = 'majority',

        # Pruning
        pruning: Optional[Literal['pre', 'post', 'both']] = None,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,

        # Regularization
        max_leaf_nodes: Optional[int] = None,
        min_weight_fraction_leaf: float = 0.0,

        # Performance
        use_cache: bool = True,
        cache_size_mb: int = 100,

        # Other
        random_state: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        """Initialize classifier with hyperparameters."""
        # Store all parameters (required for sklearn get_params/set_params)
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_pvalue = min_pvalue
        self.use_bonferroni = use_bonferroni
        self.class_weight = class_weight
        self.use_weighted_gini = use_weighted_gini
        self.missing_value_strategy = missing_value_strategy
        self.pruning = pruning
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.max_leaf_nodes = max_leaf_nodes
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.use_cache = use_cache
        self.cache_size_mb = cache_size_mb
        self.random_state = random_state
        self.verbose = verbose

    # ========== sklearn Required Methods ==========

    def fit(
        self,
        csv_path: str,
        config_path: str,
        sample_weight: Optional[NDArray[np.float64]] = None
    ) -> 'ThetaSketchDecisionTreeClassifier':
        """
        Build decision tree from theta sketches.

        Parameters
        ----------
        csv_path : str
            Path to CSV file containing theta sketches
        config_path : str
            Path to YAML config file with targets and feature mapping
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights (not used in sketch-based training)

        Returns
        -------
        self : ThetaSketchDecisionTreeClassifier
            Fitted classifier
        """
        pass

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Raw feature values for inference

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        pass

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Raw feature values for inference

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities [P(class=0), P(class=1)]
        """
        pass

    def predict_log_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict log of class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Raw feature values for inference

        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
            Log of class probabilities
        """
        pass

    def score(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        sample_weight: Optional[NDArray[np.float64]] = None
    ) -> float:
        """
        Return accuracy score on test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True labels
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights

        Returns
        -------
        score : float
            Accuracy score
        """
        pass

    # ========== Additional Public Methods ==========

    def compute_roc_curve(
        self,
        X: NDArray[np.float64],
        y_true: NDArray[np.int64]
    ) -> Dict[str, Any]:
        """
        Compute ROC curve data.

        Returns
        -------
        roc_data : dict
            Dictionary with keys: 'fpr', 'tpr', 'thresholds', 'auc'
        """
        pass

    def plot_roc_curve(
        self,
        X: NDArray[np.float64],
        y_true: NDArray[np.int64],
        ax: Optional[Any] = None
    ) -> Any:
        """Plot ROC curve."""
        pass

    def get_feature_importance(
        self,
        method: Literal['gini', 'split_frequency'] = 'gini'
    ) -> NDArray[np.float64]:
        """Get feature importance scores."""
        pass

    def plot_feature_importance(
        self,
        method: Literal['gini', 'split_frequency'] = 'gini',
        top_n: int = 10
    ) -> None:
        """Plot feature importance bar chart."""
        pass

    def export_tree_json(self) -> str:
        """Export tree structure as JSON string."""
        pass

    def save_model(self, path: str) -> None:
        """Save model to pickle file."""
        pass

    @classmethod
    def load_model(cls, path: str) -> 'ThetaSketchDecisionTreeClassifier':
        """Load model from pickle file."""
        pass

    # ========== Private Methods ==========

    def _validate_hyperparameters(self) -> None:
        """Validate hyperparameters."""
        pass

    def _compute_class_weights(self, sketch_dict: Dict) -> Dict[int, float]:
        """Compute class weights from sketch cardinalities."""
        pass

    def _compute_feature_importances_gini(self) -> NDArray[np.float64]:
        """Compute Gini-based feature importance."""
        pass

    def _compute_feature_importances_split_frequency(self) -> NDArray[np.float64]:
        """Compute split frequency-based importance."""
        pass
```

---

### 2.2 Tree & TreeNode

**Responsibility**: Data structures for storing the tree

```python
from typing import Optional, Union, List, Dict, Any
import numpy as np
from numpy.typing import NDArray

class TreeNode:
    """
    Single node in the decision tree.

    Attributes (for all nodes)
    --------------------------
    depth : int
        Depth of this node in tree (root = 0)
    n_samples : float
        Number of samples at this node (from sketch estimates)
    impurity : float
        Impurity value (Gini, entropy, etc.) at this node
    class_counts : NDArray[np.float64]
        Count of samples per class [n_class_0, n_class_1]

    Attributes (for internal nodes)
    -------------------------------
    feature_idx : int
        Index of feature used for split
    feature_name : str
        Name of feature (e.g., "age>30")
    left : TreeNode
        Left child (feature condition = False)
    right : TreeNode
        Right child (feature condition = True)
    n_samples_left : float
        Samples that went left
    n_samples_right : float
        Samples that went right
    missing_direction : Literal['left', 'right']
        Where to send missing values

    Attributes (for leaf nodes)
    ---------------------------
    is_leaf : bool
        True if this is a leaf node
    prediction : int
        Predicted class (0 or 1)
    probabilities : NDArray[np.float64]
        Class probabilities [P(class=0), P(class=1)]
    """

    def __init__(
        self,
        depth: int,
        n_samples: float,
        class_counts: NDArray[np.float64],
        impurity: float
    ) -> None:
        """Initialize tree node."""
        self.depth = depth
        self.n_samples = n_samples
        self.class_counts = class_counts
        self.impurity = impurity

        # Internal node attributes
        self.feature_idx: Optional[int] = None
        self.feature_name: Optional[str] = None
        self.left: Optional[TreeNode] = None
        self.right: Optional[TreeNode] = None
        self.n_samples_left: float = 0.0
        self.n_samples_right: float = 0.0
        self.missing_direction: Optional[str] = None

        # Leaf node attributes
        self.is_leaf: bool = False
        self.prediction: Optional[int] = None
        self.probabilities: Optional[NDArray[np.float64]] = None

    def make_leaf(self) -> None:
        """Convert this node to a leaf."""
        self.is_leaf = True
        self.prediction = int(np.argmax(self.class_counts))
        total = np.sum(self.class_counts)
        self.probabilities = self.class_counts / total if total > 0 else np.array([0.5, 0.5])

    def set_split(
        self,
        feature_idx: int,
        feature_name: str,
        left_child: 'TreeNode',
        right_child: 'TreeNode'
    ) -> None:
        """Set split information for internal node."""
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        self.left = left_child
        self.right = right_child
        self.n_samples_left = left_child.n_samples
        self.n_samples_right = right_child.n_samples

        # Set missing direction to majority path
        if self.n_samples_left >= self.n_samples_right:
            self.missing_direction = 'left'
        else:
            self.missing_direction = 'right'

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary (for JSON export)."""
        if self.is_leaf:
            return {
                'type': 'leaf',
                'depth': self.depth,
                'n_samples': float(self.n_samples),
                'prediction': int(self.prediction),
                'probabilities': self.probabilities.tolist(),
                'class_counts': self.class_counts.tolist()
            }
        else:
            return {
                'type': 'internal',
                'depth': self.depth,
                'n_samples': float(self.n_samples),
                'feature_idx': self.feature_idx,
                'feature_name': self.feature_name,
                'impurity': float(self.impurity),
                'missing_direction': self.missing_direction,
                'left': self.left.to_dict() if self.left else None,
                'right': self.right.to_dict() if self.right else None
            }


class Tree:
    """
    Complete decision tree structure.

    Attributes
    ----------
    root : TreeNode
        Root node of the tree
    n_nodes : int
        Total number of nodes
    n_leaves : int
        Number of leaf nodes
    max_depth : int
        Maximum depth reached
    """

    def __init__(self, root: TreeNode) -> None:
        """Initialize tree with root node."""
        self.root = root
        self.n_nodes: int = 0
        self.n_leaves: int = 0
        self.max_depth: int = 0
        self._compute_statistics()

    def _compute_statistics(self) -> None:
        """Compute tree statistics (n_nodes, n_leaves, max_depth)."""
        def traverse(node: Optional[TreeNode]) -> None:
            if node is None:
                return

            self.n_nodes += 1
            self.max_depth = max(self.max_depth, node.depth)

            if node.is_leaf:
                self.n_leaves += 1
            else:
                traverse(node.left)
                traverse(node.right)

        traverse(self.root)

    def to_dict(self) -> Dict[str, Any]:
        """Export tree to dictionary."""
        return {
            'n_nodes': self.n_nodes,
            'n_leaves': self.n_leaves,
            'max_depth': self.max_depth,
            'root': self.root.to_dict()
        }
```

---

## 3. Data Loading Classes

### 3.1 SketchLoader

**Responsibility**: Load theta sketches from CSV file

```python
from typing import Dict, Tuple
import csv
import base64
from datasketches import compact_theta_sketch

class SketchLoader:
    """
    Load theta sketches from CSV file.

    CSV Format:
    -----------
    <total|empty>, <base64_sketch_bytes>
    <feature_name>, <base64_sketch_bytes>
    ...
    """

    def __init__(self, encoding: str = 'base64') -> None:
        """
        Initialize loader.

        Parameters
        ----------
        encoding : str
            Encoding of sketch bytes ('base64' or 'hex')
        """
        self.encoding = encoding

    def load(
        self,
        csv_path: str,
        target_positive: str,
        target_negative: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load sketches from CSV file.

        Parameters
        ----------
        csv_path : str
            Path to CSV file
        target_positive : str
            Name of positive class sketch (e.g., "target_yes")
        target_negative : str
            Name of negative class sketch (e.g., "target_no")

        Returns
        -------
        sketches_positive : dict
            {feature_name: ThetaSketch} for positive class
        sketches_negative : dict
            {feature_name: ThetaSketch} for negative class
        """
        pass

    def _decode_sketch_bytes(self, sketch_str: str) -> bytes:
        """Decode sketch string to bytes."""
        if self.encoding == 'base64':
            return base64.b64decode(sketch_str)
        elif self.encoding == 'hex':
            return bytes.fromhex(sketch_str)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

    def _deserialize_sketch(self, sketch_bytes: bytes) -> Any:
        """Deserialize bytes to ThetaSketch object."""
        return compact_theta_sketch.deserialize(sketch_bytes)
```

---

### 3.2 ConfigParser

**Responsibility**: Parse YAML configuration file

```python
from typing import Dict, Any, Tuple, Callable
import yaml

class ConfigParser:
    """
    Parse YAML configuration file.

    Expected Format:
    ----------------
    targets:
      positive: "target_yes"
      negative: "target_no"
    hyperparameters:
      criterion: "gini"
      max_depth: 10
      ...
    feature_mapping:
      "age>30":
        column_index: 0
        operator: ">"
        threshold: 30
      ...
    """

    def load(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Parameters
        ----------
        config_path : str
            Path to YAML config file

        Returns
        -------
        config : dict
            Parsed configuration with keys:
            - 'targets': dict with 'positive' and 'negative'
            - 'hyperparameters': dict of hyperparameters
            - 'feature_mapping': dict of feature mappings
        """
        pass

    def parse_feature_mapping(
        self,
        feature_mapping_config: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Tuple[int, Callable]]:
        """
        Convert config feature mapping to lambda functions.

        Parameters
        ----------
        feature_mapping_config : dict
            Raw feature mapping from config

        Returns
        -------
        feature_mapping : dict
            {feature_name: (column_idx, lambda_function)}

        Example
        -------
        Input:
        {
            "age>30": {"column_index": 0, "operator": ">", "threshold": 30}
        }

        Output:
        {
            "age>30": (0, lambda x: x > 30)
        }
        """
        pass

    def _create_comparison_lambda(
        self,
        operator: str,
        threshold: float
    ) -> Callable:
        """Create comparison lambda function."""
        operator_map = {
            '>': lambda x: x > threshold,
            '>=': lambda x: x >= threshold,
            '<': lambda x: x < threshold,
            '<=': lambda x: x <= threshold,
            '==': lambda x: x == threshold,
            '!=': lambda x: x != threshold
        }

        if operator not in operator_map:
            raise ValueError(f"Unknown operator: {operator}")

        return operator_map[operator]

    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure."""
        required_keys = ['targets', 'hyperparameters', 'feature_mapping']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

        if 'positive' not in config['targets'] or 'negative' not in config['targets']:
            raise ValueError("targets must have 'positive' and 'negative' keys")
```

---

## 4. Splitting & Criteria Classes

### 4.1 TreeBuilder

**Responsibility**: Build tree recursively

```python
from typing import Dict, Optional, Any
import numpy as np
from numpy.typing import NDArray

class TreeBuilder:
    """
    Recursively build decision tree from sketches.

    Attributes
    ----------
    criterion : BaseCriterion
        Split criterion calculator
    splitter : SplitEvaluator
        Split evaluator
    pruner : Optional[BasePruner]
        Pruning strategy
    """

    def __init__(
        self,
        criterion: 'BaseCriterion',
        splitter: 'SplitEvaluator',
        pruner: Optional['BasePruner'] = None,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0
    ) -> None:
        """Initialize tree builder."""
        self.criterion = criterion
        self.splitter = splitter
        self.pruner = pruner
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.verbose = verbose

        if random_state is not None:
            np.random.seed(random_state)

    def build_tree(
        self,
        sketch_dict_pos: Dict[str, Any],
        sketch_dict_neg: Dict[str, Any],
        feature_names: List[str],
        depth: int = 0
    ) -> TreeNode:
        """
        Recursively build tree.

        Parameters
        ----------
        sketch_dict_pos : dict
            Sketches for positive class at this node
        sketch_dict_neg : dict
            Sketches for negative class at this node
        feature_names : list
            List of feature names to consider for splitting
        depth : int
            Current depth in tree

        Returns
        -------
        node : TreeNode
            Root of (sub)tree
        """
        pass

    def _should_stop_splitting(
        self,
        n_samples: float,
        depth: int,
        impurity: float,
        class_counts: NDArray[np.float64]
    ) -> bool:
        """Check stopping criteria."""
        # Pure node
        if impurity == 0:
            return True

        # Max depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            return True

        # Insufficient samples
        if n_samples < self.min_samples_split:
            return True

        # Single class
        if np.sum(class_counts > 0) <= 1:
            return True

        return False
```

---

### 4.2 SplitEvaluator

**Responsibility**: Evaluate all possible splits and select best

```python
from typing import Dict, Optional, Tuple, Any, List
import numpy as np

class SplitEvaluator:
    """
    Evaluate candidate splits and select best one.

    Attributes
    ----------
    criterion : BaseCriterion
        Split criterion
    sketch_cache : SketchCache
        Cache for sketch operations
    """

    def __init__(
        self,
        criterion: 'BaseCriterion',
        sketch_cache: Optional['SketchCache'] = None,
        max_features: Optional[Union[int, float, str]] = None,
        splitter: str = 'best',
        random_state: Optional[int] = None
    ) -> None:
        """Initialize split evaluator."""
        self.criterion = criterion
        self.sketch_cache = sketch_cache
        self.max_features = max_features
        self.splitter = splitter
        self.random_state = random_state

    def find_best_split(
        self,
        sketch_dict_pos: Dict[str, Any],
        sketch_dict_neg: Dict[str, Any],
        feature_names: List[str],
        parent_impurity: float
    ) -> Optional[Tuple[str, float, Dict, Dict, Dict, Dict]]:
        """
        Find best feature to split on.

        Parameters
        ----------
        sketch_dict_pos : dict
            Sketches for positive class
        sketch_dict_neg : dict
            Sketches for negative class
        feature_names : list
            Features to consider
        parent_impurity : float
            Impurity of parent node

        Returns
        -------
        best_split : tuple or None
            (feature_name, score,
             left_sketches_pos, left_sketches_neg,
             right_sketches_pos, right_sketches_neg)
            or None if no valid split found
        """
        pass

    def _evaluate_feature_split(
        self,
        feature_name: str,
        sketch_dict_pos: Dict[str, Any],
        sketch_dict_neg: Dict[str, Any],
        parent_impurity: float
    ) -> Tuple[float, Dict, Dict, Dict, Dict]:
        """
        Evaluate split on single feature.

        Returns
        -------
        score : float
            Split score (lower is better for impurity, higher for information gain)
        left_sketches_pos : dict
            Positive class sketches for left child
        left_sketches_neg : dict
            Negative class sketches for left child
        right_sketches_pos : dict
            Positive class sketches for right child
        right_sketches_neg : dict
            Negative class sketches for right child
        """
        pass

    def _compute_child_sketches(
        self,
        feature_name: str,
        parent_sketch: Any,
        feature_sketch: Any
    ) -> Tuple[Any, Any]:
        """
        Compute child sketches using set operations.

        For binary split on feature F:
        - Left child (F=False): parent.a_not_b(feature)
        - Right child (F=True): parent.intersection(feature)

        Returns
        -------
        left_sketch : ThetaSketch
            Sketch for left child
        right_sketch : ThetaSketch
            Sketch for right child
        """
        pass

    def _select_features_to_try(
        self,
        feature_names: List[str]
    ) -> List[str]:
        """Select subset of features based on max_features."""
        n_features = len(feature_names)

        if self.max_features is None:
            return feature_names
        elif isinstance(self.max_features, int):
            n_to_select = min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            n_to_select = max(1, int(self.max_features * n_features))
        elif self.max_features == 'sqrt':
            n_to_select = max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            n_to_select = max(1, int(np.log2(n_features)))
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")

        if self.splitter == 'random':
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(n_features, size=n_to_select, replace=False)
            return [feature_names[i] for i in indices]
        else:
            return feature_names[:n_to_select]
```

---

### 4.3 BaseCriterion (Abstract)

**Responsibility**: Define interface for split criteria

```python
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

class BaseCriterion(ABC):
    """
    Abstract base class for split criteria.
    """

    def __init__(
        self,
        class_weight: Optional[Dict[int, float]] = None,
        min_pvalue: float = 0.05,
        use_bonferroni: bool = True
    ) -> None:
        """Initialize criterion."""
        self.class_weight = class_weight
        self.min_pvalue = min_pvalue
        self.use_bonferroni = use_bonferroni

    @abstractmethod
    def compute_impurity(
        self,
        class_counts: NDArray[np.float64]
    ) -> float:
        """
        Compute impurity for a node.

        Parameters
        ----------
        class_counts : array of shape (n_classes,)
            Count of samples per class

        Returns
        -------
        impurity : float
            Impurity value
        """
        pass

    @abstractmethod
    def evaluate_split(
        self,
        parent_counts: NDArray[np.float64],
        left_counts: NDArray[np.float64],
        right_counts: NDArray[np.float64],
        parent_impurity: float
    ) -> float:
        """
        Evaluate quality of a split.

        Parameters
        ----------
        parent_counts : array of shape (n_classes,)
            Class counts at parent
        left_counts : array of shape (n_classes,)
            Class counts at left child
        right_counts : array of shape (n_classes,)
            Class counts at right child
        parent_impurity : float
            Impurity of parent node

        Returns
        -------
        score : float
            Split score (interpretation depends on criterion)
        """
        pass
```

---

### 4.4 Concrete Criterion Classes

```python
class GiniCriterion(BaseCriterion):
    """
    Gini impurity criterion.

    Gini(t) = 1 - Σ(p_i²)

    For weighted Gini:
    Gini_weighted(t) = 1 - Σ((w_i * p_i)²) / (Σw_i * p_i)²
    """

    def compute_impurity(
        self,
        class_counts: NDArray[np.float64]
    ) -> float:
        """Compute Gini impurity."""
        total = np.sum(class_counts)
        if total == 0:
            return 0.0

        if self.class_weight is None:
            # Standard Gini
            probabilities = class_counts / total
            return 1.0 - np.sum(probabilities ** 2)
        else:
            # Weighted Gini
            weights = np.array([self.class_weight.get(i, 1.0) for i in range(len(class_counts))])
            weighted_counts = class_counts * weights
            total_weighted = np.sum(weighted_counts)
            if total_weighted == 0:
                return 0.0
            weighted_probs = weighted_counts / total_weighted
            return 1.0 - np.sum(weighted_probs ** 2)

    def evaluate_split(
        self,
        parent_counts: NDArray[np.float64],
        left_counts: NDArray[np.float64],
        right_counts: NDArray[np.float64],
        parent_impurity: float
    ) -> float:
        """
        Evaluate split using weighted impurity decrease.

        Returns negative impurity decrease (lower is better).
        """
        n_parent = np.sum(parent_counts)
        n_left = np.sum(left_counts)
        n_right = np.sum(right_counts)

        if n_left == 0 or n_right == 0:
            return 0.0  # Invalid split

        left_impurity = self.compute_impurity(left_counts)
        right_impurity = self.compute_impurity(right_counts)

        # Weighted average of child impurities
        weighted_impurity = (n_left / n_parent) * left_impurity + (n_right / n_parent) * right_impurity

        # Impurity decrease (higher is better, so return negative)
        impurity_decrease = parent_impurity - weighted_impurity
        return -impurity_decrease


class EntropyCriterion(BaseCriterion):
    """
    Shannon entropy criterion (Information Gain).

    Entropy(t) = -Σ(p_i * log2(p_i))
    Information Gain = Entropy(parent) - Weighted_Avg(Entropy(children))
    """

    def compute_impurity(
        self,
        class_counts: NDArray[np.float64]
    ) -> float:
        """Compute Shannon entropy."""
        total = np.sum(class_counts)
        if total == 0:
            return 0.0

        probabilities = class_counts / total
        # Avoid log(0)
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))

    def evaluate_split(
        self,
        parent_counts: NDArray[np.float64],
        left_counts: NDArray[np.float64],
        right_counts: NDArray[np.float64],
        parent_impurity: float
    ) -> float:
        """Evaluate split using information gain (return negative)."""
        n_parent = np.sum(parent_counts)
        n_left = np.sum(left_counts)
        n_right = np.sum(right_counts)

        if n_left == 0 or n_right == 0:
            return 0.0

        left_impurity = self.compute_impurity(left_counts)
        right_impurity = self.compute_impurity(right_counts)

        weighted_impurity = (n_left / n_parent) * left_impurity + (n_right / n_parent) * right_impurity
        information_gain = parent_impurity - weighted_impurity

        return -information_gain


class GainRatioCriterion(EntropyCriterion):
    """
    Gain Ratio criterion (C4.5).

    GainRatio = InformationGain / SplitInfo
    where SplitInfo = -Σ(|child|/|parent| * log2(|child|/|parent|))
    """

    def evaluate_split(
        self,
        parent_counts: NDArray[np.float64],
        left_counts: NDArray[np.float64],
        right_counts: NDArray[np.float64],
        parent_impurity: float
    ) -> float:
        """Evaluate split using gain ratio (return negative)."""
        n_parent = np.sum(parent_counts)
        n_left = np.sum(left_counts)
        n_right = np.sum(right_counts)

        if n_left == 0 or n_right == 0:
            return 0.0

        # Information gain
        left_impurity = self.compute_impurity(left_counts)
        right_impurity = self.compute_impurity(right_counts)
        weighted_impurity = (n_left / n_parent) * left_impurity + (n_right / n_parent) * right_impurity
        information_gain = parent_impurity - weighted_impurity

        # Split information
        p_left = n_left / n_parent
        p_right = n_right / n_parent
        split_info = -(p_left * np.log2(p_left) + p_right * np.log2(p_right))

        if split_info == 0:
            return 0.0

        gain_ratio = information_gain / split_info
        return -gain_ratio


class BinomialCriterion(BaseCriterion):
    """
    Binomial statistical test criterion.

    Tests whether class proportions in children are significantly different from parent.
    Uses binomial test p-value as split criterion.
    """

    def compute_impurity(
        self,
        class_counts: NDArray[np.float64]
    ) -> float:
        """Not used for statistical tests, return Gini as fallback."""
        total = np.sum(class_counts)
        if total == 0:
            return 0.0
        probabilities = class_counts / total
        return 1.0 - np.sum(probabilities ** 2)

    def evaluate_split(
        self,
        parent_counts: NDArray[np.float64],
        left_counts: NDArray[np.float64],
        right_counts: NDArray[np.float64],
        parent_impurity: float
    ) -> float:
        """
        Evaluate split using binomial test.

        Returns p-value (lower is better for significant splits).
        """
        from scipy.stats import binomtest

        n_parent = np.sum(parent_counts)
        n_left = np.sum(left_counts)
        n_right = np.sum(right_counts)

        if n_left == 0 or n_right == 0:
            return 1.0  # Not significant

        # Proportion of positive class in parent
        p_parent = parent_counts[1] / n_parent if n_parent > 0 else 0.5

        # Test left child
        k_left = left_counts[1]
        p_value_left = binomtest(int(k_left), int(n_left), p_parent, alternative='two-sided').pvalue

        # Test right child
        k_right = right_counts[1]
        p_value_right = binomtest(int(k_right), int(n_right), p_parent, alternative='two-sided').pvalue

        # Use minimum p-value (most significant)
        p_value = min(p_value_left, p_value_right)

        # Bonferroni correction if enabled
        if self.use_bonferroni:
            # Correction factor could be number of features tested
            pass

        return p_value


class ChiSquareCriterion(BaseCriterion):
    """
    Chi-square test criterion.

    Tests independence between split and class label.
    """

    def compute_impurity(
        self,
        class_counts: NDArray[np.float64]
    ) -> float:
        """Not used for statistical tests, return Gini as fallback."""
        total = np.sum(class_counts)
        if total == 0:
            return 0.0
        probabilities = class_counts / total
        return 1.0 - np.sum(probabilities ** 2)

    def evaluate_split(
        self,
        parent_counts: NDArray[np.float64],
        left_counts: NDArray[np.float64],
        right_counts: NDArray[np.float64],
        parent_impurity: float
    ) -> float:
        """
        Evaluate split using chi-square test.

        Returns p-value (lower is better).
        """
        from scipy.stats import chi2_contingency

        # Contingency table: [left_class0, left_class1]
        #                     [right_class0, right_class1]
        contingency_table = np.array([
            left_counts,
            right_counts
        ])

        # Avoid issues with zero counts
        if np.any(np.sum(contingency_table, axis=0) == 0) or np.any(np.sum(contingency_table, axis=1) == 0):
            return 1.0

        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        return p_value
```

---

## 5. Inference Classes

### 5.1 FeatureTransformer

**Responsibility**: Transform raw features to binary features

```python
from typing import Dict, Tuple, Callable
import numpy as np
from numpy.typing import NDArray
import pandas as pd

class FeatureTransformer:
    """
    Transform raw tabular data to binary features using feature mapping.

    Attributes
    ----------
    feature_mapping : dict
        {feature_name: (column_idx, lambda_function)}
    feature_names : list
        Ordered list of feature names
    """

    def __init__(
        self,
        feature_mapping: Dict[str, Tuple[int, Callable]]
    ) -> None:
        """Initialize transformer."""
        self.feature_mapping = feature_mapping
        self.feature_names = list(feature_mapping.keys())
        self.n_features = len(feature_mapping)

    def transform(
        self,
        X: NDArray[np.float64]
    ) -> NDArray[np.bool_]:
        """
        Transform raw features to binary features.

        Parameters
        ----------
        X : array of shape (n_samples, n_raw_features)
            Raw feature values

        Returns
        -------
        X_binary : array of shape (n_samples, n_binary_features)
            Binary feature matrix (True/False)
        """
        n_samples = X.shape[0]
        X_binary = np.zeros((n_samples, self.n_features), dtype=np.bool_)

        for i, feature_name in enumerate(self.feature_names):
            col_idx, condition_fn = self.feature_mapping[feature_name]

            # Apply condition to column
            X_binary[:, i] = np.array([
                condition_fn(x) if not pd.isna(x) else np.nan
                for x in X[:, col_idx]
            ])

        return X_binary

    def get_feature_index(self, feature_name: str) -> int:
        """Get index of feature by name."""
        return self.feature_names.index(feature_name)
```

---

### 5.2 TreeTraverser

**Responsibility**: Navigate tree for inference

```python
from typing import Optional
import numpy as np
from numpy.typing import NDArray
import pandas as pd

class TreeTraverser:
    """
    Navigate decision tree for making predictions.

    Attributes
    ----------
    tree : Tree
        The decision tree
    missing_value_strategy : str
        How to handle missing values ('majority', 'zero', 'error')
    """

    def __init__(
        self,
        tree: Tree,
        missing_value_strategy: str = 'majority'
    ) -> None:
        """Initialize traverser."""
        self.tree = tree
        self.missing_value_strategy = missing_value_strategy

    def predict(
        self,
        X_binary: NDArray[np.bool_]
    ) -> NDArray[np.int64]:
        """
        Predict class labels for binary feature matrix.

        Parameters
        ----------
        X_binary : array of shape (n_samples, n_features)
            Binary features (True/False, possibly with NaN for missing)

        Returns
        -------
        predictions : array of shape (n_samples,)
            Predicted class labels
        """
        predictions = np.array([
            self._predict_single(sample)
            for sample in X_binary
        ])
        return predictions

    def predict_proba(
        self,
        X_binary: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        """
        Predict class probabilities.

        Returns
        -------
        probabilities : array of shape (n_samples, n_classes)
            Class probabilities
        """
        probabilities = np.array([
            self._predict_proba_single(sample)
            for sample in X_binary
        ])
        return probabilities

    def _predict_single(self, sample: NDArray) -> int:
        """Predict single sample."""
        node = self._traverse_to_leaf(sample, self.tree.root)
        return node.prediction

    def _predict_proba_single(self, sample: NDArray) -> NDArray[np.float64]:
        """Predict probabilities for single sample."""
        node = self._traverse_to_leaf(sample, self.tree.root)
        return node.probabilities

    def _traverse_to_leaf(
        self,
        sample: NDArray,
        node: TreeNode
    ) -> TreeNode:
        """
        Recursively traverse tree to leaf.

        Handles missing values according to strategy.
        """
        if node.is_leaf:
            return node

        # Get feature value
        feature_value = sample[node.feature_idx]

        # Handle missing value
        if pd.isna(feature_value):
            if self.missing_value_strategy == 'error':
                raise ValueError(f"Missing value at feature {node.feature_name}")
            elif self.missing_value_strategy == 'zero':
                feature_value = False  # Treat as False
            elif self.missing_value_strategy == 'majority':
                # Use precomputed majority direction
                if node.missing_direction == 'left':
                    return self._traverse_to_leaf(sample, node.left)
                else:
                    return self._traverse_to_leaf(sample, node.right)

        # Standard traversal
        if feature_value:  # True
            return self._traverse_to_leaf(sample, node.right)
        else:  # False
            return self._traverse_to_leaf(sample, node.left)
```

---

## 6. Pruning Classes

### 6.1 PrePruner

```python
class PrePruner:
    """
    Pre-pruning (early stopping) strategy.

    Checks stopping criteria before creating split.
    """

    def __init__(
        self,
        min_impurity_decrease: float = 0.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_depth: Optional[int] = None
    ) -> None:
        """Initialize pre-pruner."""
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth

    def should_prune(
        self,
        node: TreeNode,
        proposed_split: Optional[Tuple],
        impurity_decrease: float
    ) -> bool:
        """
        Check if node should be pruned (made a leaf).

        Parameters
        ----------
        node : TreeNode
            Current node
        proposed_split : tuple or None
            Proposed split information
        impurity_decrease : float
            Impurity decrease from proposed split

        Returns
        -------
        should_prune : bool
            True if node should be made a leaf
        """
        # Max depth reached
        if self.max_depth is not None and node.depth >= self.max_depth:
            return True

        # Insufficient samples
        if node.n_samples < self.min_samples_split:
            return True

        # No valid split found
        if proposed_split is None:
            return True

        # Impurity decrease too small
        if impurity_decrease < self.min_impurity_decrease:
            return True

        # Check min_samples_leaf for proposed children
        if proposed_split is not None:
            _, _, left_sketches_pos, left_sketches_neg, right_sketches_pos, right_sketches_neg = proposed_split
            n_left = left_sketches_pos['total'].get_estimate() + left_sketches_neg['total'].get_estimate()
            n_right = right_sketches_pos['total'].get_estimate() + right_sketches_neg['total'].get_estimate()

            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                return True

        return False
```

---

### 6.2 PostPruner

```python
class PostPruner:
    """
    Post-pruning using cost-complexity pruning (CCP).

    After building full tree, prune back using cost-complexity criterion.
    """

    def __init__(self, ccp_alpha: float = 0.0) -> None:
        """Initialize post-pruner."""
        self.ccp_alpha = ccp_alpha

    def prune_tree(self, tree: Tree) -> Tree:
        """
        Prune tree using cost-complexity criterion.

        Parameters
        ----------
        tree : Tree
            Unpruned tree

        Returns
        -------
        pruned_tree : Tree
            Pruned tree
        """
        if self.ccp_alpha <= 0:
            return tree  # No pruning

        # Implement minimal cost-complexity pruning
        # This is a simplified version
        pass

    def compute_pruning_path(
        self,
        tree: Tree
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute sequence of alphas and corresponding impurities.

        Returns
        -------
        ccp_alphas : array
            Sequence of alpha values
        impurities : array
            Corresponding impurities
        """
        pass
```

---

## 7. Utility Classes

### 7.1 SketchCache

```python
import hashlib
from typing import Dict, Any, Optional

class SketchCache:
    """
    LRU cache for theta sketch operations.

    Caches expensive operations:
    - sketch.get_estimate()
    - sketch_a.intersection(sketch_b)
    - sketch_a.union(sketch_b)
    - sketch_a.a_not_b(sketch_b)
    """

    def __init__(self, max_size_mb: int = 100) -> None:
        """
        Initialize cache.

        Parameters
        ----------
        max_size_mb : int
            Maximum cache size in megabytes
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache: Dict[str, Any] = {}
        self.current_size_bytes: int = 0
        self.access_order: List[str] = []  # For LRU eviction

    def get_key(self, operation: str, *sketch_ids: str) -> str:
        """Generate cache key."""
        key_str = f"{operation}:{':'.join(sorted(sketch_ids))}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve from cache."""
        if key in self.cache:
            # Update access order (LRU)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Any, size_bytes: int = 8) -> None:
        """
        Store in cache with LRU eviction.

        Parameters
        ----------
        key : str
            Cache key
        value : any
            Value to cache
        size_bytes : int
            Size of value in bytes (default 8 for float64)
        """
        if size_bytes > self.max_size_bytes:
            return  # Don't cache if too large

        # Evict if necessary
        while self.current_size_bytes + size_bytes > self.max_size_bytes:
            if not self.access_order:
                break
            oldest_key = self.access_order.pop(0)
            if oldest_key in self.cache:
                del self.cache[oldest_key]
                self.current_size_bytes -= 8  # Assume float64

        self.cache[key] = value
        self.access_order.append(key)
        self.current_size_bytes += size_bytes

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.access_order.clear()
        self.current_size_bytes = 0
```

---

### 7.2 FeatureImportanceCalculator

```python
class FeatureImportanceCalculator:
    """
    Calculate feature importance scores.
    """

    def __init__(self, tree: Tree, n_features: int) -> None:
        """Initialize calculator."""
        self.tree = tree
        self.n_features = n_features

    def compute_gini_importance(self) -> NDArray[np.float64]:
        """
        Compute Gini-based feature importance.

        Importance = sum of weighted impurity decreases for each feature.
        """
        importances = np.zeros(self.n_features, dtype=np.float64)

        def traverse(node: Optional[TreeNode]) -> None:
            if node is None or node.is_leaf:
                return

            # Weighted impurity decrease
            n_samples = node.n_samples
            impurity_decrease = node.impurity

            if node.left and node.right:
                impurity_decrease -= (
                    (node.left.n_samples / n_samples) * node.left.impurity +
                    (node.right.n_samples / n_samples) * node.right.impurity
                )

            importances[node.feature_idx] += n_samples * impurity_decrease

            traverse(node.left)
            traverse(node.right)

        traverse(self.tree.root)

        # Normalize
        total = np.sum(importances)
        if total > 0:
            importances /= total

        return importances

    def compute_split_frequency_importance(self) -> NDArray[np.float64]:
        """
        Compute importance based on split frequency.

        Importance = number of times feature is used for splitting.
        """
        counts = np.zeros(self.n_features, dtype=np.float64)

        def traverse(node: Optional[TreeNode]) -> None:
            if node is None or node.is_leaf:
                return

            counts[node.feature_idx] += 1
            traverse(node.left)
            traverse(node.right)

        traverse(self.tree.root)

        # Normalize
        total = np.sum(counts)
        if total > 0:
            counts /= total

        return counts
```

---

### 7.3 MetricsCalculator

```python
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

class MetricsCalculator:
    """
    Calculate performance metrics.
    """

    @staticmethod
    def compute_roc_curve(
        y_true: NDArray[np.int64],
        y_proba: NDArray[np.float64]
    ) -> Dict[str, Any]:
        """
        Compute ROC curve data.

        Parameters
        ----------
        y_true : array of shape (n_samples,)
            True labels
        y_proba : array of shape (n_samples,)
            Predicted probabilities for positive class

        Returns
        -------
        roc_data : dict
            Dictionary with keys: 'fpr', 'tpr', 'thresholds', 'auc'
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': auc
        }

    @staticmethod
    def compute_precision_recall_curve(
        y_true: NDArray[np.int64],
        y_proba: NDArray[np.float64]
    ) -> Dict[str, Any]:
        """Compute precision-recall curve data."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds
        }
```

---

## 8. Interaction Diagrams

### 8.1 Training (fit) Workflow

```
User
  │
  ├─► fit(csv_path, config_path)
  │       │
  │       ├─► SketchLoader.load(csv_path, targets)
  │       │       └─► Returns: sketches_pos, sketches_neg
  │       │
  │       ├─► ConfigParser.load(config_path)
  │       │       └─► Returns: config{targets, hyperparameters, feature_mapping}
  │       │
  │       ├─► ConfigParser.parse_feature_mapping()
  │       │       └─► Returns: {feature_name: (col_idx, lambda)}
  │       │
  │       ├─► Initialize components:
  │       │       ├─► SketchCache(cache_size_mb)
  │       │       ├─► Criterion (Gini/Entropy/etc.)
  │       │       ├─► SplitEvaluator(criterion, cache)
  │       │       ├─► Pruner (Pre/Post/Both)
  │       │       └─► TreeBuilder(criterion, splitter, pruner)
  │       │
  │       ├─► TreeBuilder.build_tree(sketches_pos, sketches_neg, feature_names)
  │       │       │
  │       │       ├─► Calculate node statistics
  │       │       ├─► Check stopping criteria
  │       │       │
  │       │       ├─► SplitEvaluator.find_best_split()
  │       │       │       │
  │       │       │       ├─► For each feature:
  │       │       │       │       ├─► Compute child sketches (intersection, a_not_b)
  │       │       │       │       ├─► SketchCache.get() or compute
  │       │       │       │       ├─► Criterion.evaluate_split()
  │       │       │       │       └─► Track best split
  │       │       │       │
  │       │       │       └─► Return best_split
  │       │       │
  │       │       ├─► Pruner.should_prune()?
  │       │       │       └─► If yes: make leaf, return
  │       │       │
  │       │       ├─► Create node with split
  │       │       ├─► Recursively build left subtree
  │       │       └─► Recursively build right subtree
  │       │
  │       ├─► Set attributes: classes_, n_classes_, n_features_in_, tree_
  │       │
  │       └─► Return self
  │
  └─► Returns: Fitted classifier
```

---

### 8.2 Inference (predict) Workflow

```
User
  │
  ├─► predict(X_raw)
  │       │
  │       ├─► Validate: check_is_fitted()
  │       ├─► Validate: check_array(X)
  │       │
  │       ├─► FeatureTransformer.transform(X_raw)
  │       │       │
  │       │       ├─► For each feature in feature_mapping:
  │       │       │       ├─► Get column index
  │       │       │       ├─► Apply lambda function
  │       │       │       └─► Store binary result
  │       │       │
  │       │       └─► Return X_binary (n_samples × n_features, bool)
  │       │
  │       ├─► TreeTraverser.predict(X_binary)
  │       │       │
  │       │       ├─► For each sample:
  │       │       │       │
  │       │       │       └─► _traverse_to_leaf(sample, root)
  │       │       │               │
  │       │       │               ├─► If leaf: return prediction
  │       │       │               │
  │       │       │               ├─► Get feature value
  │       │       │               │
  │       │       │               ├─► If missing:
  │       │       │               │       ├─► strategy='error': raise error
  │       │       │               │       ├─► strategy='zero': value = False
  │       │       │               │       └─► strategy='majority': follow majority_direction
  │       │       │               │
  │       │       │               ├─► If value==True: go right
  │       │       │               └─► If value==False: go left
  │       │       │
  │       │       └─► Return predictions array
  │       │
  │       └─► Return predictions
  │
  └─► Returns: Predicted labels
```

---

### 8.3 Split Evaluation Detail

```
SplitEvaluator.find_best_split()
  │
  ├─► Select features to try (based on max_features)
  │       └─► Returns: feature_subset
  │
  ├─► For each feature in feature_subset:
  │       │
  │       └─► _evaluate_feature_split(feature_name)
  │               │
  │               ├─► _compute_child_sketches()
  │               │       │
  │               │       ├─► For positive class:
  │               │       │       ├─► total_sketch = sketches_pos['total']
  │               │       │       ├─► feature_sketch = sketches_pos[feature_name]
  │               │       │       │
  │               │       │       ├─► Cache key: "intersection:total_id:feature_id"
  │               │       │       ├─► SketchCache.get(key)
  │               │       │       │       └─► If miss:
  │               │       │       │               ├─► right = total.intersection(feature)
  │               │       │       │               ├─► n_right = right.get_estimate()
  │               │       │       │               └─► SketchCache.put(key, n_right)
  │               │       │       │
  │               │       │       ├─► Cache key: "a_not_b:total_id:feature_id"
  │               │       │       ├─► SketchCache.get(key)
  │               │       │       │       └─► If miss:
  │               │       │       │               ├─► left = total.a_not_b(feature)
  │               │       │       │               ├─► n_left = left.get_estimate()
  │               │       │       │               └─► SketchCache.put(key, n_left)
  │               │       │       │
  │               │       │       └─► Returns: (left_sketch, right_sketch)
  │               │       │
  │               │       └─► Repeat for negative class
  │               │
  │               ├─► Compute class counts:
  │               │       ├─► left_counts = [n_left_neg, n_left_pos]
  │               │       └─► right_counts = [n_right_neg, n_right_pos]
  │               │
  │               ├─► Criterion.evaluate_split(parent_counts, left_counts, right_counts)
  │               │       └─► Returns: score
  │               │
  │               └─► Returns: (score, left_sketches_pos, left_sketches_neg,
  │                            right_sketches_pos, right_sketches_neg)
  │
  ├─► Track best split (lowest score for Gini/Entropy, lowest p-value for statistical)
  │
  └─► Return best_split or None
```

---

## Summary

This low-level design provides:

✅ **Complete class specifications** with all methods and attributes
✅ **Type hints** for all parameters and return values
✅ **Clear responsibilities** for each class
✅ **Detailed interaction diagrams** showing data flow
✅ **Sketch operation caching** for performance
✅ **Extensible architecture** with abstract base classes
✅ **sklearn compatibility** through proper inheritance

**Key Design Patterns Used**:
- **Strategy Pattern**: Pluggable criteria (Gini, Entropy, etc.)
- **Template Method**: BaseCriterion defines interface, subclasses implement
- **Builder Pattern**: TreeBuilder constructs tree recursively
- **Cache Pattern**: SketchCache for performance optimization
- **Facade Pattern**: ThetaSketchDecisionTreeClassifier provides simple API

**Next Steps**:
1. Implement each class following this design
2. Write unit tests for each class
3. Integration tests for complete workflow
4. Performance profiling and optimization
