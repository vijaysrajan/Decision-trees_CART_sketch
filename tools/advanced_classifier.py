#!/usr/bin/env python3
"""
Advanced Classification Tool for Theta Sketch Decision Trees

Post-processing classification with multiple strategies:
1. Class weightage - Adjust decision threshold based on class costs
2. Binomial p-values - Statistical significance testing
3. Ratio threshold - User-defined percentage cutoffs
4. Confidence intervals - Wilson score intervals for leaf probabilities

Optional minimum collateral damage constraint applies to all strategies.
"""

import json
import pickle
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import argparse
import yaml
from pathlib import Path
import sys
import os

# Add parent directory to path to import theta_sketch_tree
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class AdvancedClassifier:
    """Post-processing classifier with multiple strategies"""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        """Initialize classifier with trained model and configuration"""
        self.model = self._load_model(model_path)
        self.config = config
        self.strategy = config['strategy']
        self.min_collateral = config.get('min_collateral_damage', None)

    def _load_model(self, model_path: str):
        """Load trained model from pickle file"""
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def predict_advanced(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Advanced prediction with selected strategy

        Returns list of prediction dictionaries with:
        - prediction: final class prediction
        - confidence: strategy-specific confidence score
        - leaf_stats: raw leaf statistics
        - strategy_info: strategy-specific details
        """
        predictions = []

        for idx, row in X.iterrows():
            # Get basic prediction and leaf stats from model
            basic_pred = self.model.predict([row.values])[0]
            leaf_stats = self._get_leaf_stats(row.values)

            # Apply selected strategy
            result = self._apply_strategy(leaf_stats, basic_pred)

            # Apply collateral damage constraint if specified
            if self.min_collateral is not None:
                result = self._apply_collateral_constraint(result, leaf_stats)

            result['row_index'] = idx
            predictions.append(result)

        return predictions

    def _get_leaf_stats(self, sample: np.ndarray) -> Dict[str, Any]:
        """Extract leaf node statistics for a sample"""
        # Use the model's tree traverser to get leaf node
        from theta_sketch_tree.tree_traverser import TreeTraverser

        if not hasattr(self.model, 'tree_'):
            raise ValueError("Model doesn't have a fitted tree")

        traverser = TreeTraverser(self.model.tree_)
        leaf_node = traverser.traverse_to_leaf(sample)

        # Extract statistics from leaf node
        total_samples = int(leaf_node.n_samples)
        class_counts = leaf_node.class_counts
        positive_samples = int(class_counts[1]) if len(class_counts) > 1 else 0
        negative_samples = int(class_counts[0]) if len(class_counts) > 0 else 0
        positive_rate = positive_samples / total_samples if total_samples > 0 else 0.0

        return {
            'total_samples': total_samples,
            'positive_samples': positive_samples,
            'negative_samples': negative_samples,
            'positive_rate': positive_rate,
            'leaf_node': leaf_node
        }

    def _apply_strategy(self, leaf_stats: Dict[str, Any], basic_pred: int) -> Dict[str, Any]:
        """Apply selected classification strategy"""

        if self.strategy == 'class_weightage':
            return self._class_weightage_strategy(leaf_stats, basic_pred)
        elif self.strategy == 'binomial_pvalue':
            return self._binomial_pvalue_strategy(leaf_stats, basic_pred)
        elif self.strategy == 'ratio_threshold':
            return self._ratio_threshold_strategy(leaf_stats, basic_pred)
        elif self.strategy == 'confidence_intervals':
            return self._confidence_intervals_strategy(leaf_stats, basic_pred)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _class_weightage_strategy(self, leaf_stats: Dict[str, Any], basic_pred: int) -> Dict[str, Any]:
        """Class weightage strategy - adjust threshold based on class costs"""
        positive_weight = self.config.get('positive_class_weight', 1.0)
        negative_weight = self.config.get('negative_class_weight', 1.0)

        # Adjust threshold based on class weights
        # Higher positive weight -> lower threshold for positive prediction
        weight_ratio = negative_weight / positive_weight
        adjusted_threshold = 0.5 / (1 + weight_ratio)

        positive_rate = leaf_stats['positive_rate']
        prediction = 1 if positive_rate >= adjusted_threshold else 0
        confidence = abs(positive_rate - adjusted_threshold)

        return {
            'prediction': prediction,
            'confidence': confidence,
            'leaf_stats': leaf_stats,
            'strategy_info': {
                'strategy': 'class_weightage',
                'positive_weight': positive_weight,
                'negative_weight': negative_weight,
                'adjusted_threshold': adjusted_threshold,
                'raw_positive_rate': positive_rate
            }
        }

    def _binomial_pvalue_strategy(self, leaf_stats: Dict[str, Any], basic_pred: int) -> Dict[str, Any]:
        """Binomial p-value strategy - statistical significance testing"""
        population_rate = self.config.get('population_positive_rate', 0.5)
        significance_level = self.config.get('significance_level', 0.05)

        n = leaf_stats['total_samples']
        k = leaf_stats['positive_samples']

        if n == 0:
            return {
                'prediction': 0,
                'confidence': 0.0,
                'leaf_stats': leaf_stats,
                'strategy_info': {
                    'strategy': 'binomial_pvalue',
                    'pvalue': 1.0,
                    'significant': False,
                    'reason': 'no_samples'
                }
            }

        # Two-tailed binomial test
        pvalue = stats.binomtest(k, n, population_rate, alternative='two-sided').pvalue
        is_significant = pvalue < significance_level

        # If significant, predict based on whether positive_rate > population_rate
        if is_significant:
            prediction = 1 if leaf_stats['positive_rate'] > population_rate else 0
            confidence = 1 - pvalue  # Higher confidence for lower p-values
        else:
            prediction = 0  # Default to negative if not significant
            confidence = pvalue

        return {
            'prediction': prediction,
            'confidence': confidence,
            'leaf_stats': leaf_stats,
            'strategy_info': {
                'strategy': 'binomial_pvalue',
                'pvalue': pvalue,
                'significant': is_significant,
                'population_rate': population_rate,
                'significance_level': significance_level
            }
        }

    def _ratio_threshold_strategy(self, leaf_stats: Dict[str, Any], basic_pred: int) -> Dict[str, Any]:
        """Ratio threshold strategy - simple percentage cutoffs"""
        threshold = self.config.get('ratio_threshold', 0.5)
        min_samples = self.config.get('min_samples_for_prediction', 10)

        positive_rate = leaf_stats['positive_rate']
        n = leaf_stats['total_samples']

        if n < min_samples:
            return {
                'prediction': 0,
                'confidence': 0.0,
                'leaf_stats': leaf_stats,
                'strategy_info': {
                    'strategy': 'ratio_threshold',
                    'threshold': threshold,
                    'positive_rate': positive_rate,
                    'reason': 'insufficient_samples',
                    'min_samples_required': min_samples
                }
            }

        prediction = 1 if positive_rate >= threshold else 0
        confidence = abs(positive_rate - threshold)

        return {
            'prediction': prediction,
            'confidence': confidence,
            'leaf_stats': leaf_stats,
            'strategy_info': {
                'strategy': 'ratio_threshold',
                'threshold': threshold,
                'positive_rate': positive_rate
            }
        }

    def _confidence_intervals_strategy(self, leaf_stats: Dict[str, Any], basic_pred: int) -> Dict[str, Any]:
        """Confidence intervals strategy - Wilson score intervals"""
        confidence_level = self.config.get('confidence_level', 0.95)
        decision_threshold = self.config.get('decision_threshold', 0.5)

        n = leaf_stats['total_samples']
        p = leaf_stats['positive_rate']

        if n == 0:
            return {
                'prediction': 0,
                'confidence': 0.0,
                'leaf_stats': leaf_stats,
                'strategy_info': {
                    'strategy': 'confidence_intervals',
                    'reason': 'no_samples'
                }
            }

        # Wilson score interval
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

        ci_lower = center - margin
        ci_upper = center + margin

        # Decision based on confidence interval
        if ci_lower > decision_threshold:
            prediction = 1
            confidence = ci_lower - decision_threshold
        elif ci_upper < decision_threshold:
            prediction = 0
            confidence = decision_threshold - ci_upper
        else:
            # Interval spans threshold - low confidence
            prediction = 1 if p > decision_threshold else 0
            confidence = 0.0

        return {
            'prediction': prediction,
            'confidence': confidence,
            'leaf_stats': leaf_stats,
            'strategy_info': {
                'strategy': 'confidence_intervals',
                'confidence_level': confidence_level,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'decision_threshold': decision_threshold,
                'interval_width': ci_upper - ci_lower
            }
        }

    def _apply_collateral_constraint(self, result: Dict[str, Any], leaf_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Apply minimum collateral damage constraint"""
        if result['prediction'] == 1:  # Only apply to positive predictions
            safe_samples = leaf_stats['negative_samples']
            if safe_samples < self.min_collateral:
                # Override prediction due to insufficient collateral
                result['prediction'] = 0
                result['confidence'] = 0.0
                result['strategy_info']['collateral_override'] = True
                result['strategy_info']['safe_samples'] = safe_samples
                result['strategy_info']['min_collateral_required'] = self.min_collateral

        return result

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_predictions(predictions: List[Dict[str, Any]], output_path: str):
    """Save predictions to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2, default=str)

def print_prediction_summary(predictions: List[Dict[str, Any]]):
    """Print summary of predictions"""
    total = len(predictions)
    positive_preds = sum(1 for p in predictions if p['prediction'] == 1)

    print(f"\n📊 Prediction Summary")
    print(f"Total samples: {total}")
    print(f"Positive predictions: {positive_preds} ({positive_preds/total*100:.1f}%)")
    print(f"Negative predictions: {total-positive_preds} ({(total-positive_preds)/total*100:.1f}%)")

    # Strategy-specific summary
    if predictions:
        strategy = predictions[0]['strategy_info']['strategy']
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        print(f"Strategy: {strategy}")
        print(f"Average confidence: {avg_confidence:.3f}")

        # Count overrides if any
        overrides = sum(1 for p in predictions if p['strategy_info'].get('collateral_override', False))
        if overrides > 0:
            print(f"Collateral constraint overrides: {overrides}")

def main():
    parser = argparse.ArgumentParser(description='Advanced Classification Tool')
    parser.add_argument('--model', required=True, help='Path to trained model (.pkl)')
    parser.add_argument('--data', required=True, help='Path to test data (.csv)')
    parser.add_argument('--config', required=True, help='Path to configuration (.yaml)')
    parser.add_argument('--output', help='Output path for predictions (.json)')
    parser.add_argument('--summary', action='store_true', help='Print prediction summary')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load test data
    test_data = pd.read_csv(args.data)

    # Initialize classifier
    classifier = AdvancedClassifier(args.model, config)

    # Generate predictions
    predictions = classifier.predict_advanced(test_data)

    # Save predictions if output path specified
    if args.output:
        save_predictions(predictions, args.output)
        print(f"Predictions saved to: {args.output}")

    # Print summary if requested
    if args.summary:
        print_prediction_summary(predictions)

if __name__ == '__main__':
    main()