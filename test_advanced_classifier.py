#!/usr/bin/env python3
"""
Test script for advanced classifier with DU model
"""

import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path to import tools
sys.path.append('.')

from tools.advanced_classifier import AdvancedClassifier

def test_advanced_classifier():
    """Test the advanced classifier with DU model and data"""

    # Configuration for ratio threshold strategy
    config = {
        'strategy': 'ratio_threshold',
        'ratio_threshold': 0.30,
        'min_samples_for_prediction': 20
    }

    print("🧪 Testing Advanced Classifier")
    print("=" * 50)

    # Load model
    model_path = 'DU_output/du_model_lg_k_18/3col_sketches_lg_k_18_model_lg_k_18.pkl'
    print(f"Loading model from: {model_path}")

    try:
        classifier = AdvancedClassifier(model_path, config)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Create test data - binary features matching the model's expected format
    print("\n📊 Creating test data...")

    try:
        # Get feature names from the model
        feature_names = classifier.model.feature_names_in_
        n_features = len(feature_names)
        print(f"Model expects {n_features} features")

        # Create simple test samples with binary values (0/1)
        np.random.seed(42)
        test_samples = pd.DataFrame(
            np.random.choice([0, 1], size=(5, n_features)),
            columns=feature_names
        )
        print(f"Test samples shape: {test_samples.shape}")

    except Exception as e:
        print(f"❌ Failed to create test data: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test predictions
    print("\n🔮 Testing predictions...")
    try:
        predictions = classifier.predict_advanced(test_samples)
        print(f"✅ Generated {len(predictions)} predictions")

        # Show sample results
        for i, pred in enumerate(predictions[:3]):
            print(f"\nSample {i+1}:")
            print(f"  Prediction: {pred['prediction']}")
            print(f"  Confidence: {pred['confidence']:.3f}")
            print(f"  Leaf samples: {pred['leaf_stats']['total_samples']}")
            print(f"  Positive rate: {pred['leaf_stats']['positive_rate']:.3f}")

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n✅ Advanced classifier test completed successfully!")

if __name__ == '__main__':
    test_advanced_classifier()