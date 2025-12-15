#!/usr/bin/env python3
"""
Demo of Advanced Classification Strategies
Simulates the strategies without needing the actual model
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any
import json

def demo_classification_strategies():
    """Demonstrate all 4 classification strategies with simulated leaf data"""

    print("🎯 Advanced Classification Strategies Demo")
    print("=" * 50)

    # Simulated leaf node statistics from different scenarios
    test_cases = [
        {
            'name': 'High Confidence Positive',
            'total_samples': 1000,
            'positive_samples': 800,
            'description': 'Strong positive signal with many samples'
        },
        {
            'name': 'Low Confidence Positive',
            'total_samples': 50,
            'positive_samples': 30,
            'description': 'Positive signal but few samples'
        },
        {
            'name': 'Borderline Case',
            'total_samples': 200,
            'positive_samples': 100,
            'description': 'Equal positive/negative rates'
        },
        {
            'name': 'Strong Negative',
            'total_samples': 500,
            'positive_samples': 25,
            'description': 'Strong negative signal'
        },
        {
            'name': 'Tiny Sample',
            'total_samples': 5,
            'positive_samples': 4,
            'description': 'High rate but very few samples'
        }
    ]

    strategies = [
        ('Class Weightage', strategy_class_weightage),
        ('Binomial P-value', strategy_binomial_pvalue),
        ('Ratio Threshold', strategy_ratio_threshold),
        ('Confidence Intervals', strategy_confidence_intervals)
    ]

    # Test each strategy on each case
    for case in test_cases:
        print(f"\n📊 Test Case: {case['name']}")
        print(f"   {case['description']}")
        print(f"   Samples: {case['total_samples']}, Positive: {case['positive_samples']}")
        print(f"   Positive Rate: {case['positive_samples']/case['total_samples']:.3f}")
        print("   " + "─" * 60)

        leaf_stats = {
            'total_samples': case['total_samples'],
            'positive_samples': case['positive_samples'],
            'negative_samples': case['total_samples'] - case['positive_samples'],
            'positive_rate': case['positive_samples'] / case['total_samples']
        }

        for strategy_name, strategy_func in strategies:
            result = strategy_func(leaf_stats)
            prediction_text = "🔴 NEGATIVE" if result['prediction'] == 0 else "🟢 POSITIVE"
            print(f"   {strategy_name:20} → {prediction_text} (conf: {result['confidence']:.3f})")

        print()

def strategy_class_weightage(leaf_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Class weightage strategy - false negatives cost 3x more than false positives"""
    positive_weight = 3.0  # False negatives expensive
    negative_weight = 1.0  # False positives less expensive

    weight_ratio = negative_weight / positive_weight
    adjusted_threshold = 0.5 / (1 + weight_ratio)  # Lower threshold for positive

    positive_rate = leaf_stats['positive_rate']
    prediction = 1 if positive_rate >= adjusted_threshold else 0
    confidence = abs(positive_rate - adjusted_threshold)

    return {
        'prediction': prediction,
        'confidence': confidence,
        'strategy_info': {
            'adjusted_threshold': adjusted_threshold,
            'positive_weight': positive_weight
        }
    }

def strategy_binomial_pvalue(leaf_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Binomial p-value strategy - statistical significance testing"""
    population_rate = 0.15  # Overall positive rate in population
    significance_level = 0.05

    n = leaf_stats['total_samples']
    k = leaf_stats['positive_samples']

    if n == 0:
        return {'prediction': 0, 'confidence': 0.0, 'strategy_info': {'reason': 'no_samples'}}

    # Two-tailed binomial test
    pvalue = stats.binomtest(k, n, population_rate, alternative='two-sided').pvalue
    is_significant = pvalue < significance_level

    if is_significant:
        prediction = 1 if leaf_stats['positive_rate'] > population_rate else 0
        confidence = 1 - pvalue
    else:
        prediction = 0  # Default to negative if not significant
        confidence = pvalue

    return {
        'prediction': prediction,
        'confidence': confidence,
        'strategy_info': {
            'pvalue': pvalue,
            'significant': is_significant,
            'population_rate': population_rate
        }
    }

def strategy_ratio_threshold(leaf_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Ratio threshold strategy - simple percentage cutoff"""
    threshold = 0.40  # 40% positive rate required
    min_samples = 30  # Minimum samples for reliable prediction

    positive_rate = leaf_stats['positive_rate']
    n = leaf_stats['total_samples']

    if n < min_samples:
        return {
            'prediction': 0,
            'confidence': 0.0,
            'strategy_info': {'reason': 'insufficient_samples', 'min_samples': min_samples}
        }

    prediction = 1 if positive_rate >= threshold else 0
    confidence = abs(positive_rate - threshold)

    return {
        'prediction': prediction,
        'confidence': confidence,
        'strategy_info': {
            'threshold': threshold,
            'positive_rate': positive_rate
        }
    }

def strategy_confidence_intervals(leaf_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Confidence intervals strategy - Wilson score intervals"""
    confidence_level = 0.95
    decision_threshold = 0.30

    n = leaf_stats['total_samples']
    p = leaf_stats['positive_rate']

    if n == 0:
        return {'prediction': 0, 'confidence': 0.0, 'strategy_info': {'reason': 'no_samples'}}

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
        confidence = 0.1  # Low confidence when spanning threshold

    return {
        'prediction': prediction,
        'confidence': confidence,
        'strategy_info': {
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'decision_threshold': decision_threshold
        }
    }

def demo_collateral_damage():
    """Demonstrate minimum collateral damage constraint"""
    print("\n💥 Minimum Collateral Damage Constraint Demo")
    print("=" * 50)

    min_collateral = 100  # Require at least 100 negative samples

    test_case = {
        'total_samples': 150,
        'positive_samples': 120,
        'negative_samples': 30,  # Only 30 negative samples - below threshold
        'positive_rate': 0.8
    }

    print(f"Test Case: {test_case['positive_samples']} positive, {test_case['negative_samples']} negative")
    print(f"Positive rate: {test_case['positive_rate']:.3f}")
    print(f"Minimum collateral required: {min_collateral}")

    # Original prediction (would be positive)
    original_prediction = 1
    print(f"\nOriginal prediction: {'🟢 POSITIVE' if original_prediction == 1 else '🔴 NEGATIVE'}")

    # Apply collateral constraint
    if original_prediction == 1 and test_case['negative_samples'] < min_collateral:
        final_prediction = 0
        print(f"After collateral constraint: 🔴 NEGATIVE (insufficient safe samples)")
        print(f"   → Only {test_case['negative_samples']} safe samples, need {min_collateral}")
    else:
        final_prediction = original_prediction
        print(f"After collateral constraint: {'🟢 POSITIVE' if final_prediction == 1 else '🔴 NEGATIVE'}")

def demo_real_world_scenarios():
    """Demo with realistic business scenarios"""
    print("\n🏢 Real-World Business Scenarios")
    print("=" * 50)

    scenarios = [
        {
            'name': 'High-Value Customer',
            'context': 'Expensive to lose, cheap to retain',
            'leaf_stats': {'total_samples': 300, 'positive_samples': 90},
            'strategy': 'Class Weightage (3:1)',
            'reasoning': 'False negatives (losing customers) cost 3x more'
        },
        {
            'name': 'Fraud Detection',
            'context': 'Need statistical confidence',
            'leaf_stats': {'total_samples': 1000, 'positive_samples': 50},
            'strategy': 'Binomial P-value',
            'reasoning': 'Must be significantly different from baseline'
        },
        {
            'name': 'Safety Critical',
            'context': 'Require high confidence and collateral',
            'leaf_stats': {'total_samples': 200, 'positive_samples': 140},
            'strategy': 'Confidence Intervals + Collateral',
            'reasoning': 'Need certainty and minimum risk exposure'
        },
        {
            'name': 'Resource Allocation',
            'context': 'Simple threshold with sample size',
            'leaf_stats': {'total_samples': 80, 'positive_samples': 32},
            'strategy': 'Ratio Threshold',
            'reasoning': 'Clear cutoff with reliability check'
        }
    ]

    for scenario in scenarios:
        print(f"\n📋 {scenario['name']}")
        print(f"   Context: {scenario['context']}")

        total = scenario['leaf_stats']['total_samples']
        positive = scenario['leaf_stats']['positive_samples']
        rate = positive / total

        print(f"   Data: {positive}/{total} positive ({rate:.3f})")
        print(f"   Strategy: {scenario['strategy']}")
        print(f"   Reasoning: {scenario['reasoning']}")

if __name__ == '__main__':
    demo_classification_strategies()
    demo_collateral_damage()
    demo_real_world_scenarios()

    print("\n✅ Advanced Classification Strategies Demo Complete!")
    print("\n💡 Key Insights:")
    print("   • Class Weightage: Adjusts for business costs")
    print("   • Binomial P-value: Ensures statistical significance")
    print("   • Ratio Threshold: Simple, interpretable rules")
    print("   • Confidence Intervals: Uncertainty-aware decisions")
    print("   • Collateral Damage: Risk management constraint")