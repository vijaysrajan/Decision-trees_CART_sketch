#!/usr/bin/env python3
"""
Generated decision rules for classification
Source: theta_sketch_tree
Total rules: 34
Feature count: 0
"""

def predict_sample(features):
    """
    Classify a single sample using decision rules
    
    Parameters
    ----------
    features : dict
        Feature dictionary with binary values (0/1)
        Keys should match feature names from training
    
    Returns
    -------
    dict
        Prediction result with confidence and metadata
    """

    # Rule 1: NEGATIVE (90.4% confidence, 12794.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 0 and
        features.get('pickUpHourOfDay=LATENIGHT', 0) == 0 and
        features.get('zone=221', 0) == 0 and
        features.get('zone=194', 0) == 0 and
        features.get('sla=Immediate', 0) == 0):
        return {'prediction': 0, 'confidence': 0.904, 'rule_id': 1, 'samples': 12794.0}

    # Rule 2: NEGATIVE (89.0% confidence, 1397.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 1 and
        features.get('pickUpHourOfDay=LATENIGHT', 0) == 0 and
        features.get('city=Pune', 0) == 0 and
        features.get('city=Mumbai', 0) == 0 and
        features.get('city=Hyderabad', 0) == 0):
        return {'prediction': 0, 'confidence': 0.89, 'rule_id': 2, 'samples': 1397.0}

    # Rule 3: NEGATIVE (74.4% confidence, 996.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 1 and
        features.get('city=Delhi NCR', 0) == 0 and
        features.get('dayType=WEEKEND', 0) == 1 and
        features.get('city=Chennai', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_4', 0) == 0 and
        features.get('zone=216', 0) == 0):
        return {'prediction': 0, 'confidence': 0.744, 'rule_id': 3, 'samples': 996.0}

    # Rule 4: NEGATIVE (99.0% confidence, 697.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 0 and
        features.get('pickUpHourOfDay=LATENIGHT', 0) == 0 and
        features.get('zone=221', 0) == 0 and
        features.get('zone=194', 0) == 0 and
        features.get('sla=Immediate', 0) == 1):
        return {'prediction': 0, 'confidence': 0.99, 'rule_id': 4, 'samples': 697.0}

    # Rule 5: NEGATIVE (72.0% confidence, 644.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 0 and
        features.get('pickUpHourOfDay=LATENIGHT', 0) == 1 and
        features.get('dayType=NORMAL_WEEKDAY', 0) == 0 and
        features.get('zone=217', 0) == 0 and
        features.get('city=Mumbai', 0) == 0):
        return {'prediction': 0, 'confidence': 0.72, 'rule_id': 5, 'samples': 644.0}

    # Rule 6: NEGATIVE (82.1% confidence, 397.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 1 and
        features.get('pickUpHourOfDay=LATENIGHT', 0) == 0 and
        features.get('city=Pune', 0) == 0 and
        features.get('city=Mumbai', 0) == 1 and
        features.get('booking_type=round_trip', 0) == 0):
        return {'prediction': 0, 'confidence': 0.821, 'rule_id': 6, 'samples': 397.0}

    # Rule 7: NEGATIVE (88.6% confidence, 395.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 1 and
        features.get('city=Delhi NCR', 0) == 0 and
        features.get('dayType=WEEKEND', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 0 and
        features.get('city=Hyderabad', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_1', 0) == 0):
        return {'prediction': 0, 'confidence': 0.886, 'rule_id': 7, 'samples': 395.0}

    # Rule 8: NEGATIVE (95.4% confidence, 326.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 1 and
        features.get('city=Delhi NCR', 0) == 0 and
        features.get('dayType=WEEKEND', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 0 and
        features.get('city=Hyderabad', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_1', 0) == 1):
        return {'prediction': 0, 'confidence': 0.954, 'rule_id': 8, 'samples': 326.0}

    # Rule 9: NEGATIVE (81.9% confidence, 249.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 0 and
        features.get('pickUpHourOfDay=LATENIGHT', 0) == 1 and
        features.get('dayType=NORMAL_WEEKDAY', 0) == 1 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_1', 0) == 0 and
        features.get('city=Mumbai', 0) == 0):
        return {'prediction': 0, 'confidence': 0.819, 'rule_id': 9, 'samples': 249.0}

    # Rule 10: NEGATIVE (94.2% confidence, 241.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 0 and
        features.get('pickUpHourOfDay=LATENIGHT', 0) == 1 and
        features.get('dayType=NORMAL_WEEKDAY', 0) == 1 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_1', 0) == 1 and
        features.get('zone=219', 0) == 0):
        return {'prediction': 0, 'confidence': 0.942, 'rule_id': 10, 'samples': 241.0}

    # Rule 11: NEGATIVE (83.9% confidence, 230.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 0 and
        features.get('pickUpHourOfDay=LATENIGHT', 0) == 1 and
        features.get('dayType=NORMAL_WEEKDAY', 0) == 0 and
        features.get('zone=217', 0) == 0 and
        features.get('city=Mumbai', 0) == 1):
        return {'prediction': 0, 'confidence': 0.839, 'rule_id': 11, 'samples': 230.0}

    # Rule 12: NEGATIVE (61.2% confidence, 201.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 1 and
        features.get('pickUpHourOfDay=LATENIGHT', 0) == 0 and
        features.get('city=Pune', 0) == 1 and
        features.get('booking_type=outstation', 0) == 0 and
        features.get('pickUpHourOfDay=MORNING', 0) == 0):
        return {'prediction': 0, 'confidence': 0.612, 'rule_id': 12, 'samples': 201.0}

    # Rule 13: NEGATIVE (68.3% confidence, 189.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 1 and
        features.get('pickUpHourOfDay=LATENIGHT', 0) == 0 and
        features.get('city=Pune', 0) == 0 and
        features.get('city=Mumbai', 0) == 1 and
        features.get('booking_type=round_trip', 0) == 1):
        return {'prediction': 0, 'confidence': 0.683, 'rule_id': 13, 'samples': 189.0}

    # Rule 14: NEGATIVE (96.6% confidence, 116.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 0 and
        features.get('pickUpHourOfDay=LATENIGHT', 0) == 1 and
        features.get('dayType=NORMAL_WEEKDAY', 0) == 1 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_1', 0) == 0 and
        features.get('city=Mumbai', 0) == 1):
        return {'prediction': 0, 'confidence': 0.966, 'rule_id': 14, 'samples': 116.0}

    # Rule 15: NEGATIVE (84.5% confidence, 97.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 0 and
        features.get('pickUpHourOfDay=LATENIGHT', 0) == 0 and
        features.get('zone=221', 0) == 1 and
        features.get('booking_type=one_way_trip', 0) == 0 and
        features.get('estimated_usage_bins=GT_3_LTE_4_HOURS', 0) == 0):
        return {'prediction': 0, 'confidence': 0.845, 'rule_id': 15, 'samples': 97.0}

    # Rule 16: NEGATIVE (96.9% confidence, 96.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 0 and
        features.get('pickUpHourOfDay=LATENIGHT', 0) == 1 and
        features.get('dayType=NORMAL_WEEKDAY', 0) == 1 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_1', 0) == 1 and
        features.get('zone=219', 0) == 1):
        return {'prediction': 0, 'confidence': 0.969, 'rule_id': 16, 'samples': 96.0}

    # Rule 17: NEGATIVE (71.7% confidence, 92.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 1 and
        features.get('pickUpHourOfDay=LATENIGHT', 0) == 0 and
        features.get('city=Pune', 0) == 0 and
        features.get('city=Mumbai', 0) == 0 and
        features.get('city=Hyderabad', 0) == 1):
        return {'prediction': 0, 'confidence': 0.717, 'rule_id': 17, 'samples': 92.0}

    # Rule 18: NEGATIVE (61.1% confidence, 90.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 0 and
        features.get('pickUpHourOfDay=LATENIGHT', 0) == 0 and
        features.get('zone=221', 0) == 1 and
        features.get('booking_type=one_way_trip', 0) == 1):
        return {'prediction': 0, 'confidence': 0.611, 'rule_id': 18, 'samples': 90.0}

    # Rule 19: NEGATIVE (81.1% confidence, 90.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 1 and
        features.get('pickUpHourOfDay=LATENIGHT', 0) == 0 and
        features.get('city=Pune', 0) == 1 and
        features.get('booking_type=outstation', 0) == 0 and
        features.get('pickUpHourOfDay=MORNING', 0) == 1):
        return {'prediction': 0, 'confidence': 0.811, 'rule_id': 19, 'samples': 90.0}

    # Rule 20: NEGATIVE (74.4% confidence, 90.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 1 and
        features.get('city=Delhi NCR', 0) == 0 and
        features.get('dayType=WEEKEND', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 0 and
        features.get('city=Hyderabad', 0) == 1):
        return {'prediction': 0, 'confidence': 0.744, 'rule_id': 20, 'samples': 90.0}

    # Rule 21: NEGATIVE (50.6% confidence, 85.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 1 and
        features.get('pickUpHourOfDay=LATENIGHT', 0) == 1 and
        features.get('city=Mumbai', 0) == 0):
        return {'prediction': 0, 'confidence': 0.506, 'rule_id': 21, 'samples': 85.0}

    # Rule 22: NEGATIVE (92.8% confidence, 83.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 1 and
        features.get('city=Delhi NCR', 0) == 0 and
        features.get('dayType=WEEKEND', 0) == 1 and
        features.get('city=Chennai', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_4', 0) == 0 and
        features.get('zone=216', 0) == 1):
        return {'prediction': 0, 'confidence': 0.928, 'rule_id': 22, 'samples': 83.0}

    # Rule 23: NEGATIVE (51.8% confidence, 83.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 1 and
        features.get('city=Delhi NCR', 0) == 0 and
        features.get('dayType=WEEKEND', 0) == 1 and
        features.get('city=Chennai', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_4', 0) == 1 and
        features.get('city=Mumbai', 0) == 1):
        return {'prediction': 0, 'confidence': 0.518, 'rule_id': 23, 'samples': 83.0}

    # Rule 24: NEGATIVE (60.9% confidence, 69.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 0 and
        features.get('pickUpHourOfDay=LATENIGHT', 0) == 0 and
        features.get('zone=221', 0) == 0 and
        features.get('zone=194', 0) == 1):
        return {'prediction': 0, 'confidence': 0.609, 'rule_id': 24, 'samples': 69.0}

    # Rule 25: NEGATIVE (60.6% confidence, 66.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 1 and
        features.get('city=Delhi NCR', 0) == 0 and
        features.get('dayType=WEEKEND', 0) == 1 and
        features.get('city=Chennai', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_4', 0) == 1 and
        features.get('city=Mumbai', 0) == 0):
        return {'prediction': 0, 'confidence': 0.606, 'rule_id': 25, 'samples': 66.0}

    # Rule 26: NEGATIVE (95.5% confidence, 66.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 1 and
        features.get('city=Delhi NCR', 0) == 0 and
        features.get('dayType=WEEKEND', 0) == 1 and
        features.get('city=Chennai', 0) == 1):
        return {'prediction': 0, 'confidence': 0.955, 'rule_id': 26, 'samples': 66.0}

    # Rule 27: NEGATIVE (96.8% confidence, 62.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 0 and
        features.get('pickUpHourOfDay=LATENIGHT', 0) == 1 and
        features.get('dayType=NORMAL_WEEKDAY', 0) == 0 and
        features.get('zone=217', 0) == 1):
        return {'prediction': 0, 'confidence': 0.968, 'rule_id': 27, 'samples': 62.0}

    # Rule 28: NEGATIVE (90.3% confidence, 62.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 1 and
        features.get('pickUpHourOfDay=LATENIGHT', 0) == 0 and
        features.get('city=Pune', 0) == 1 and
        features.get('booking_type=outstation', 0) == 1):
        return {'prediction': 0, 'confidence': 0.903, 'rule_id': 28, 'samples': 62.0}

    # Rule 29: NEGATIVE (77.0% confidence, 61.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 0 and
        features.get('pickUpHourOfDay=LATENIGHT', 0) == 0 and
        features.get('zone=221', 0) == 1 and
        features.get('booking_type=one_way_trip', 0) == 0 and
        features.get('estimated_usage_bins=GT_3_LTE_4_HOURS', 0) == 1):
        return {'prediction': 0, 'confidence': 0.77, 'rule_id': 29, 'samples': 61.0}

    # Rule 30: POSITIVE (63.3% confidence, 60.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 1 and
        features.get('city=Delhi NCR', 0) == 1 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_2', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_4', 0) == 0):
        return {'prediction': 1, 'confidence': 0.633, 'rule_id': 30, 'samples': 60.0}

    # Rule 31: NEGATIVE (68.4% confidence, 57.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 1 and
        features.get('city=Delhi NCR', 0) == 0 and
        features.get('dayType=WEEKEND', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 1):
        return {'prediction': 0, 'confidence': 0.684, 'rule_id': 31, 'samples': 57.0}

    # Rule 32: NEGATIVE (52.6% confidence, 57.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 1 and
        features.get('city=Delhi NCR', 0) == 1 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_2', 0) == 1):
        return {'prediction': 0, 'confidence': 0.526, 'rule_id': 32, 'samples': 57.0}

    # Rule 33: NEGATIVE (70.4% confidence, 54.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_5', 0) == 1 and
        features.get('pickUpHourOfDay=LATENIGHT', 0) == 1 and
        features.get('city=Mumbai', 0) == 1):
        return {'prediction': 0, 'confidence': 0.704, 'rule_id': 33, 'samples': 54.0}

    # Rule 34: POSITIVE (51.0% confidence, 51.0 samples)
    if (features.get('pickUpHourOfDay=VERYLATE', 0) == 1 and
        features.get('city=Delhi NCR', 0) == 1 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_2', 0) == 0 and
        features.get('zone_demand_popularity=POPULARITY_INDEX_4', 0) == 1):
        return {'prediction': 1, 'confidence': 0.51, 'rule_id': 34, 'samples': 51.0}

    # Default fallback (should not be reached if tree is complete)
    return {'prediction': 0, 'confidence': 0.500, 'rule_id': -1}


def predict_batch(features_list):
    """
    Classify multiple samples
    
    Parameters
    ----------
    features_list : list of dict
        List of feature dictionaries
    
    Returns
    -------
    list of dict
        List of prediction results
    """
    return [predict_sample(features) for features in features_list]


if __name__ == '__main__':
    # Example usage
    sample_features = {
    }
    
    result = predict_sample(sample_features)
    print(f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.3f}")
    print(f"Total rules in model: 34")