"""
Model evaluation utilities for theta sketch decision trees.

This module provides comprehensive evaluation metrics including confusion matrix,
F1 score, accuracy, Type I/II errors, and ROC/AUC analysis.
"""

from typing import Tuple, Dict, Optional, Union
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    precision_score, recall_score, roc_curve, auc,
    classification_report
)

from .classifier import ThetaSketchDecisionTreeClassifier


class ModelEvaluator:
    """
    Comprehensive model evaluation for binary classification.

    Provides standard metrics plus configurable majority class selection
    and detailed error analysis.
    """

    def __init__(self, classifier: ThetaSketchDecisionTreeClassifier, majority_class: int = 1):
        """
        Initialize evaluator with fitted classifier.

        Parameters
        ----------
        classifier : ThetaSketchDecisionTreeClassifier
            Fitted classifier to evaluate
        majority_class : int, default=1
            Which class to treat as majority/default (0 or 1)
        """
        self.classifier = classifier
        self.majority_class = majority_class
        self.minority_class = 1 - majority_class

    def evaluate_comprehensive(
        self,
        X_test: NDArray,
        y_test: NDArray,
        print_results: bool = True
    ) -> Dict[str, Union[float, NDArray, Dict]]:
        """
        Perform comprehensive evaluation with all metrics.

        Parameters
        ----------
        X_test : ndarray
            Test features (binary matrix)
        y_test : ndarray
            True test labels
        print_results : bool, default=True
            Whether to print formatted results

        Returns
        -------
        results : dict
            Dictionary containing all evaluation metrics
        """
        # Generate predictions
        y_pred = self.classifier.predict(X_test)
        y_proba = self.classifier.predict_proba(X_test)

        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        # Confusion matrix and error analysis
        cm = confusion_matrix(y_test, y_pred)
        type1_error, type2_error = self._calculate_error_rates(cm)

        # ROC analysis
        roc_data = self._calculate_roc_metrics(y_test, y_proba)

        # Threshold analysis
        threshold_analysis = self._analyze_thresholds(y_test, y_proba)

        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'type1_error': type1_error,
            'type2_error': type2_error,
            'roc_auc': roc_data['auc'],
            'roc_curve': roc_data,
            'threshold_analysis': threshold_analysis,
            'classification_report': classification_report(y_test, y_pred)
        }

        if print_results:
            self._print_evaluation_results(results)

        return results

    def _calculate_error_rates(self, cm: NDArray) -> Tuple[float, float]:
        """
        Calculate Type I and Type II error rates from confusion matrix.

        Type I Error (False Positive): Predicting positive when actually negative
        Type II Error (False Negative): Predicting negative when actually positive
        """
        tn, fp, fn, tp = cm.ravel()

        # Type I Error: False Positive Rate
        type1_error = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Type II Error: False Negative Rate
        type2_error = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        return type1_error, type2_error

    def _calculate_roc_metrics(self, y_true: NDArray, y_proba: NDArray) -> Dict:
        """Calculate ROC curve and AUC score."""
        # Use probability of positive class
        y_scores = y_proba[:, 1] if y_proba.ndim == 2 else y_proba

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc
        }

    def _analyze_thresholds(self, y_true: NDArray, y_proba: NDArray) -> Dict:
        """
        Analyze model performance across different decision thresholds.

        Varies threshold from 10% to 90% in steps of 10%.
        """
        thresholds = np.arange(0.1, 1.0, 0.1)
        y_scores = y_proba[:, 1] if y_proba.ndim == 2 else y_proba

        analysis = {
            'thresholds': thresholds,
            'accuracies': [],
            'f1_scores': [],
            'precisions': [],
            'recalls': [],
            'type1_errors': [],
            'type2_errors': []
        }

        for threshold in thresholds:
            # Generate predictions at this threshold
            y_pred_thresh = (y_scores >= threshold).astype(int)

            # Calculate metrics
            acc = accuracy_score(y_true, y_pred_thresh)
            f1 = f1_score(y_true, y_pred_thresh, average='weighted')
            prec = precision_score(y_true, y_pred_thresh, average='weighted')
            rec = recall_score(y_true, y_pred_thresh, average='weighted')

            # Error rates
            cm = confusion_matrix(y_true, y_pred_thresh)
            type1, type2 = self._calculate_error_rates(cm)

            # Store results
            analysis['accuracies'].append(acc)
            analysis['f1_scores'].append(f1)
            analysis['precisions'].append(prec)
            analysis['recalls'].append(rec)
            analysis['type1_errors'].append(type1)
            analysis['type2_errors'].append(type2)

        return analysis

    def _print_evaluation_results(self, results: Dict) -> None:
        """Print formatted evaluation results."""
        print("=" * 60)
        print("📊 MODEL EVALUATION RESULTS")
        print("=" * 60)

        # Basic metrics
        print(f"\n🎯 CLASSIFICATION METRICS:")
        print(f"   Accuracy:     {results['accuracy']:.4f}")
        print(f"   F1 Score:     {results['f1_score']:.4f}")
        print(f"   Precision:    {results['precision']:.4f}")
        print(f"   Recall:       {results['recall']:.4f}")
        print(f"   ROC AUC:      {results['roc_auc']:.4f}")

        # Error analysis
        print(f"\n❌ ERROR ANALYSIS:")
        print(f"   Type I Error (FPR):  {results['type1_error']:.4f}")
        print(f"   Type II Error (FNR): {results['type2_error']:.4f}")

        # Confusion matrix
        print(f"\n🔢 CONFUSION MATRIX:")
        cm = results['confusion_matrix']
        print(f"                 Predicted")
        print(f"              Neg    Pos")
        print(f"   Actual Neg  {cm[0,0]:4d}   {cm[0,1]:4d}")
        print(f"          Pos  {cm[1,0]:4d}   {cm[1,1]:4d}")

        # Threshold analysis summary
        thresh_data = results['threshold_analysis']
        best_f1_idx = np.argmax(thresh_data['f1_scores'])
        best_acc_idx = np.argmax(thresh_data['accuracies'])

        print(f"\n📈 THRESHOLD ANALYSIS:")
        print(f"   Best F1 at threshold {thresh_data['thresholds'][best_f1_idx]:.1f}: {thresh_data['f1_scores'][best_f1_idx]:.4f}")
        print(f"   Best Accuracy at threshold {thresh_data['thresholds'][best_acc_idx]:.1f}: {thresh_data['accuracies'][best_acc_idx]:.4f}")

        print(f"\n🏷️  DETAILED CLASSIFICATION REPORT:")
        print(results['classification_report'])

        print("=" * 60)

    def plot_roc_curve(self, results: Dict, save_path: Optional[str] = None) -> None:
        """Plot ROC curve with AUC score."""
        roc_data = results['roc_curve']

        plt.figure(figsize=(8, 6))
        plt.plot(roc_data['fpr'], roc_data['tpr'],
                label=f'ROC Curve (AUC = {roc_data["auc"]:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Theta Sketch Decision Tree')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_threshold_analysis(self, results: Dict, save_path: Optional[str] = None) -> None:
        """Plot metrics across different decision thresholds."""
        thresh_data = results['threshold_analysis']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        thresholds = thresh_data['thresholds']

        # Accuracy and F1
        ax1.plot(thresholds, thresh_data['accuracies'], 'b-o', label='Accuracy')
        ax1.plot(thresholds, thresh_data['f1_scores'], 'r-s', label='F1 Score')
        ax1.set_xlabel('Decision Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Accuracy and F1 vs Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Precision and Recall
        ax2.plot(thresholds, thresh_data['precisions'], 'g-^', label='Precision')
        ax2.plot(thresholds, thresh_data['recalls'], 'purple', marker='v', label='Recall')
        ax2.set_xlabel('Decision Threshold')
        ax2.set_ylabel('Score')
        ax2.set_title('Precision and Recall vs Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Error Rates
        ax3.plot(thresholds, thresh_data['type1_errors'], 'orange', marker='d', label='Type I Error')
        ax3.plot(thresholds, thresh_data['type2_errors'], 'brown', marker='h', label='Type II Error')
        ax3.set_xlabel('Decision Threshold')
        ax3.set_ylabel('Error Rate')
        ax3.set_title('Error Rates vs Threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Combined view
        ax4.plot(thresholds, thresh_data['accuracies'], 'b-o', label='Accuracy')
        ax4.plot(thresholds, thresh_data['f1_scores'], 'r-s', label='F1')
        ax4.plot(thresholds, np.array(thresh_data['type1_errors']) + np.array(thresh_data['type2_errors']),
                'k--', label='Total Error Rate')
        ax4.set_xlabel('Decision Threshold')
        ax4.set_ylabel('Score/Rate')
        ax4.set_title('Overall Performance vs Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def evaluate_model(
    classifier: ThetaSketchDecisionTreeClassifier,
    X_test: NDArray,
    y_test: NDArray,
    majority_class: int = 1,
    print_results: bool = True,
    plot_roc: bool = False,
    plot_thresholds: bool = False
) -> Dict:
    """
    Convenience function for complete model evaluation.

    Parameters
    ----------
    classifier : ThetaSketchDecisionTreeClassifier
        Fitted classifier to evaluate
    X_test : ndarray
        Test features
    y_test : ndarray
        Test labels
    majority_class : int, default=1
        Majority class for error analysis
    print_results : bool, default=True
        Print formatted results
    plot_roc : bool, default=False
        Generate ROC curve plot
    plot_thresholds : bool, default=False
        Generate threshold analysis plots

    Returns
    -------
    results : dict
        Complete evaluation results
    """
    evaluator = ModelEvaluator(classifier, majority_class)
    results = evaluator.evaluate_comprehensive(X_test, y_test, print_results)

    if plot_roc:
        evaluator.plot_roc_curve(results)

    if plot_thresholds:
        evaluator.plot_threshold_analysis(results)

    return results