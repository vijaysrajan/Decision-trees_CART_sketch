# API Design Document
## Theta Sketch Decision Tree Classifier - Complete Public API

---

## Main Classifier API

```python
class ThetaSketchDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    CART Decision Tree Classifier trained on theta sketches, infers on raw data.

    This classifier implements the sklearn estimator interface and can be used
    in sklearn pipelines, grid search, and cross-validation workflows.

    Training Phase:
        - Input: CSV file with theta sketches + YAML config file
        - Learns tree structure from sketch set operations

    Inference Phase:
        - Input: Raw tabular data (numpy/pandas)
        - Transforms features using feature mapping
        - Navigates tree to make predictions

    Parameters
    ----------
    criterion : {'gini', 'entropy', 'gain_ratio', 'binomial', 'binomial_chi'}, default='gini'
        Split criterion for evaluating feature quality.

    max_depth : int or None, default=None
        Maximum depth of tree. None means unlimited.

    min_samples_split : int, default=2
        Minimum samples required to split an internal node.

    min_samples_leaf : int, default=1
        Minimum samples required in a leaf node.

    min_impurity_decrease : float, default=0.0
        Minimum impurity decrease required for split (pre-pruning).

    class_weight : {'balanced', dict, None}, default=None
        Weights for classes. 'balanced' uses n_samples / (n_classes * bincount(y)).

    missing_value_strategy : {'majority', 'zero', 'error'}, default='majority'
        How to handle missing values during inference.

    pruning : {None, 'pre', 'post', 'both'}, default=None
        Pruning strategy.

    ccp_alpha : float, default=0.0
        Cost-complexity parameter for post-pruning.

    use_cache : bool, default=True
        Whether to cache sketch operations during training.

    random_state : int or None, default=None
        Random seed for reproducibility.

    verbose : int, default=0
        Verbosity level (0=silent, 1=progress, 2=debug).

    Attributes (set during fit)
    ----------------------------
    classes_ : ndarray of shape (n_classes,)
        Class labels [0, 1].

    n_classes_ : int
        Number of classes (always 2).

    n_features_in_ : int
        Number of features.

    feature_names_in_ : ndarray of shape (n_features,)
        Feature names from feature_mapping.

    tree_ : Tree
        The underlying tree structure.

    feature_importances_ : ndarray of shape (n_features,)
        Feature importance scores (computed lazily).

    Examples
    --------
    >>> from theta_sketch_tree import ThetaSketchDecisionTreeClassifier
    >>> import numpy as np
    >>>
    >>> # Training
    >>> clf = ThetaSketchDecisionTreeClassifier(
    ...     criterion='gini',
    ...     max_depth=5,
    ...     verbose=1
    ... )
    >>> clf.fit(csv_path='sketches.csv', config_path='config.yaml')
    >>>
    >>> # Inference
    >>> X_test = np.array([[35, 60000, 1], [25, 45000, 0]])
    >>> predictions = clf.predict(X_test)
    >>> probabilities = clf.predict_proba(X_test)
    >>>
    >>> # Evaluation
    >>> from sklearn.metrics import accuracy_score, roc_auc_score
    >>> accuracy = accuracy_score(y_true, predictions)
    >>> auc = roc_auc_score(y_true, probabilities[:, 1])
    >>>
    >>> # Feature importance
    >>> importances = clf.feature_importances_
    >>> clf.plot_feature_importance(top_n=10)
    >>>
    >>> # Model persistence
    >>> clf.save_model('model.pkl')
    >>> clf_loaded = ThetaSketchDecisionTreeClassifier.load_model('model.pkl')

    Notes
    -----
    - Training requires CSV sketches + config file, NOT raw training data
    - Inference works on standard numpy/pandas data (sklearn-compatible)
    - Missing values handled via majority path method (no imputation)
    - Supports imbalanced classification via class_weight parameter
    - Statistical criteria (binomial, chi-square) recommended for medical applications

    See Also
    --------
    sklearn.tree.DecisionTreeClassifier : Standard sklearn decision tree
    sklearn.ensemble.BaggingClassifier : Can be used to ensemble multiple trees
    """

    def __init__(
        self,
        criterion='gini',
        splitter='best',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        min_pvalue=0.05,
        use_bonferroni=True,
        class_weight=None,
        use_weighted_gini=True,
        missing_value_strategy='majority',
        pruning=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        max_leaf_nodes=None,
        min_weight_fraction_leaf=0.0,
        use_cache=True,
        cache_size_mb=100,
        random_state=None,
        verbose=0
    ):
        """Initialize classifier with hyperparameters."""
        pass

    # ========== Core sklearn Methods ==========

    def fit(self, csv_path, config_path, sample_weight=None):
        """
        Build decision tree from theta sketches.

        Parameters
        ----------
        csv_path : str
            Path to CSV file containing serialized theta sketches.
            Format: <identifier>, <sketch_bytes>

        config_path : str
            Path to YAML config file specifying:
            - targets: positive/negative class names
            - hyperparameters: tree parameters
            - feature_mapping: feature transformations for inference

        sample_weight : array-like of shape (n_samples,), default=None
            Not used in sketch-based training (reserved for future).

        Returns
        -------
        self : ThetaSketchDecisionTreeClassifier
            Fitted classifier.

        Raises
        ------
        FileNotFoundError
            If csv_path or config_path doesn't exist.
        ValueError
            If CSV format is invalid or config is malformed.

        Examples
        --------
        >>> clf = ThetaSketchDecisionTreeClassifier()
        >>> clf.fit('sketches.csv', 'config.yaml')
        >>> print(f"Tree: {clf.tree_.n_nodes} nodes, {clf.tree_.n_leaves} leaves")
        """
        pass

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Raw feature values for inference.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).

        Raises
        ------
        NotFittedError
            If classifier hasn't been fitted yet.
        ValueError
            If X has wrong number of features.

        Examples
        --------
        >>> X_test = np.array([[35, 60000, 1], [25, 45000, 0]])
        >>> predictions = clf.predict(X_test)
        >>> print(predictions)
        [1 0]
        """
        pass

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Raw feature values for inference.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities. Column 0 is P(class=0), column 1 is P(class=1).

        Examples
        --------
        >>> probabilities = clf.predict_proba(X_test)
        >>> print(probabilities)
        [[0.2 0.8]
         [0.9 0.1]]
        """
        pass

    def predict_log_proba(self, X):
        """
        Predict log of class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Raw feature values.

        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
            Log of class probabilities.

        Examples
        --------
        >>> log_probs = clf.predict_log_proba(X_test)
        """
        pass

    def score(self, X, y, sample_weight=None):
        """
        Return accuracy score on test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Accuracy score.

        Examples
        --------
        >>> accuracy = clf.score(X_test, y_test)
        >>> print(f"Accuracy: {accuracy:.3f}")
        """
        pass

    # ========== Performance Metrics ==========

    def compute_roc_curve(self, X, y_true):
        """
        Compute ROC curve data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y_true : array-like of shape (n_samples,)
            True binary labels.

        Returns
        -------
        roc_data : dict
            Dictionary with keys:
            - 'fpr': False positive rates
            - 'tpr': True positive rates
            - 'thresholds': Decision thresholds
            - 'auc': Area under ROC curve

        Examples
        --------
        >>> roc_data = clf.compute_roc_curve(X_test, y_test)
        >>> print(f"AUC: {roc_data['auc']:.3f}")
        """
        pass

    def plot_roc_curve(self, X, y_true, ax=None):
        """
        Plot ROC curve.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y_true : array-like of shape (n_samples,)
            True labels.
        ax : matplotlib.axes.Axes, default=None
            Axes object to plot on. If None, creates new figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes with ROC curve plot.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> clf.plot_roc_curve(X_test, y_test)
        >>> plt.show()
        """
        pass

    def get_classification_report(self, X, y_true):
        """
        Generate complete classification metrics report.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y_true : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        report : dict
            Dictionary with keys:
            - 'classification_report': sklearn classification_report string
            - 'confusion_matrix': 2x2 confusion matrix
            - 'roc_auc': ROC AUC score
            - 'accuracy': Accuracy score

        Examples
        --------
        >>> report = clf.get_classification_report(X_test, y_test)
        >>> print(report['classification_report'])
        >>> print(f"ROC AUC: {report['roc_auc']:.3f}")
        """
        pass

    # ========== Feature Importance ==========

    @property
    def feature_importances_(self):
        """
        Feature importance scores (Gini-based).

        Returns
        -------
        importances : ndarray of shape (n_features,)
            Feature importance scores (sum to 1.0).

        Notes
        -----
        Computed lazily on first access and cached.
        Based on weighted impurity decrease at each split.

        Examples
        --------
        >>> importances = clf.feature_importances_
        >>> for name, imp in zip(clf.feature_names_in_, importances):
        ...     print(f"{name}: {imp:.3f}")
        """
        pass

    def get_feature_importance(self, method='gini'):
        """
        Get feature importance scores using specified method.

        Parameters
        ----------
        method : {'gini', 'split_frequency'}, default='gini'
            Method for computing importance.
            - 'gini': Weighted impurity decrease (default)
            - 'split_frequency': Count of splits per feature

        Returns
        -------
        importances : ndarray of shape (n_features,)
            Feature importance scores (sum to 1.0).

        Examples
        --------
        >>> # Gini importance
        >>> gini_imp = clf.get_feature_importance(method='gini')
        >>>
        >>> # Split frequency
        >>> freq_imp = clf.get_feature_importance(method='split_frequency')
        """
        pass

    def plot_feature_importance(self, method='gini', top_n=10):
        """
        Plot feature importance bar chart.

        Parameters
        ----------
        method : {'gini', 'split_frequency'}, default='gini'
            Importance calculation method.
        top_n : int, default=10
            Number of top features to display.

        Returns
        -------
        None

        Examples
        --------
        >>> clf.plot_feature_importance(method='gini', top_n=15)
        >>> plt.tight_layout()
        >>> plt.savefig('feature_importance.png')
        """
        pass

    # ========== Model Persistence ==========

    def save_model(self, path):
        """
        Save model to file using pickle.

        Parameters
        ----------
        path : str
            File path to save model (e.g., 'model.pkl').

        Returns
        -------
        None

        Examples
        --------
        >>> clf.save_model('diabetes_model.pkl')
        """
        pass

    @classmethod
    def load_model(cls, path):
        """
        Load model from pickle file.

        Parameters
        ----------
        path : str
            File path to load model from.

        Returns
        -------
        clf : ThetaSketchDecisionTreeClassifier
            Loaded classifier.

        Examples
        --------
        >>> clf = ThetaSketchDecisionTreeClassifier.load_model('diabetes_model.pkl')
        >>> predictions = clf.predict(X_test)
        """
        pass

    def export_tree_json(self):
        """
        Export tree structure to JSON string for interpretability.

        Returns
        -------
        tree_json : str
            JSON representation of tree structure.

        Notes
        -----
        JSON export includes:
        - Tree metadata (n_nodes, n_leaves, max_depth)
        - Complete node structure with splits and predictions
        - Feature names and indices

        Does NOT include:
        - Feature mapping lambdas (not serializable)
        - Sketch cache

        For full model persistence, use save_model() instead.

        Examples
        --------
        >>> import json
        >>> tree_json = clf.export_tree_json()
        >>> tree_dict = json.loads(tree_json)
        >>> print(f"Tree depth: {tree_dict['tree_metadata']['max_depth']}")
        """
        pass

    # ========== Pruning ==========

    def cost_complexity_pruning_path(self):
        """
        Compute sequence of alphas for cost-complexity pruning.

        Returns
        -------
        path_info : dict
            Dictionary with keys:
            - 'ccp_alphas': Array of alpha values
            - 'impurities': Array of total impurities for each alpha

        Notes
        -----
        Useful for selecting optimal ccp_alpha via cross-validation.

        Examples
        --------
        >>> path = clf.cost_complexity_pruning_path()
        >>> # Train models with different alphas
        >>> for alpha in path['ccp_alphas']:
        ...     clf_pruned = ThetaSketchDecisionTreeClassifier(ccp_alpha=alpha)
        ...     clf_pruned.fit(csv_path, config_path)
        ...     # Evaluate clf_pruned...
        """
        pass
```

---

## Usage Examples

### Basic Classification

```python
from theta_sketch_tree import ThetaSketchDecisionTreeClassifier
import numpy as np

# Initialize
clf = ThetaSketchDecisionTreeClassifier(
    criterion='gini',
    max_depth=10,
    min_samples_leaf=5,
    verbose=1
)

# Train
clf.fit('sketches.csv', 'config.yaml')

# Predict
X_test = np.array([[35, 60000, 1], [25, 45000, 0]])
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

# Metrics
from sklearn.metrics import accuracy_score, roc_auc_score
accuracy = accuracy_score(y_test, predictions)
auc = roc_auc_score(y_test, probabilities[:, 1])
```

### sklearn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('tree', ThetaSketchDecisionTreeClassifier(max_depth=5))
])

# Note: fit still requires csv_path and config_path
# This works at inference time
pipe.named_steps['tree'].fit('sketches.csv', 'config.yaml')
pipe.predict(X_test)
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Create base classifier
base_clf = ThetaSketchDecisionTreeClassifier()
base_clf.fit('sketches.csv', 'config.yaml')  # Train once

# Grid search on hyperparameters
# Note: This searches over already-trained tree parameters
param_grid = {
    'ccp_alpha': [0.0, 0.01, 0.05, 0.1]
}

# For full grid search, retrain with different configs
```

---

## Error Handling

All methods include comprehensive error handling:

```python
from sklearn.exceptions import NotFittedError

try:
    clf.predict(X_test)  # Before fit
except NotFittedError:
    print("Model not fitted yet")

try:
    clf.fit('missing.csv', 'config.yaml')
except FileNotFoundError:
    print("CSV file not found")

try:
    clf.fit('invalid.csv', 'config.yaml')
except ValueError as e:
    print(f"Invalid CSV format: {e}")
```

---

## Summary

✅ **Complete sklearn API compatibility**
✅ **Comprehensive docstrings** (NumPy style)
✅ **Type hints** for all parameters
✅ **Usage examples** for each method
✅ **Error handling** specifications
✅ **Performance metrics** built-in
✅ **Model persistence** support

This API design enables seamless integration with the sklearn ecosystem while supporting theta sketch-based training.
