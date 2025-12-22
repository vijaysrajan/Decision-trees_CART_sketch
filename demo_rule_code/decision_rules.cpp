// Generated decision rules for classification
// Source: theta_sketch_tree
// Total rules: 34
// Feature count: 0

#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>

struct PredictionResult {
    int prediction;
    double confidence;
    int rule_id;
    int samples;
};

// Helper function to get feature value with default
inline int get_feature(const std::unordered_map<std::string, int>& features, const std::string& name) {
    auto it = features.find(name);
    return (it != features.end()) ? it->second : 0;
}

/**
 * Classify a single sample using decision rules
 * @param features Feature map with binary values (0/1)
 * @return Prediction result with confidence and metadata
 */
PredictionResult predict(const std::unordered_map<std::string, int>& features) {

    // Rule 1: NEGATIVE (90.4% confidence, 12794.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 0 &&
        get_feature(features, "pickUpHourOfDay=LATENIGHT") == 0 &&
        get_feature(features, "zone=221") == 0 &&
        get_feature(features, "zone=194") == 0 &&
        get_feature(features, "sla=Immediate") == 0) {
        return {0, 0.904, 1, 12794.0};
    }

    // Rule 2: NEGATIVE (89.0% confidence, 1397.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 1 &&
        get_feature(features, "pickUpHourOfDay=LATENIGHT") == 0 &&
        get_feature(features, "city=Pune") == 0 &&
        get_feature(features, "city=Mumbai") == 0 &&
        get_feature(features, "city=Hyderabad") == 0) {
        return {0, 0.890, 2, 1397.0};
    }

    // Rule 3: NEGATIVE (74.4% confidence, 996.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 1 &&
        get_feature(features, "city=Delhi NCR") == 0 &&
        get_feature(features, "dayType=WEEKEND") == 1 &&
        get_feature(features, "city=Chennai") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_4") == 0 &&
        get_feature(features, "zone=216") == 0) {
        return {0, 0.744, 3, 996.0};
    }

    // Rule 4: NEGATIVE (99.0% confidence, 697.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 0 &&
        get_feature(features, "pickUpHourOfDay=LATENIGHT") == 0 &&
        get_feature(features, "zone=221") == 0 &&
        get_feature(features, "zone=194") == 0 &&
        get_feature(features, "sla=Immediate") == 1) {
        return {0, 0.990, 4, 697.0};
    }

    // Rule 5: NEGATIVE (72.0% confidence, 644.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 0 &&
        get_feature(features, "pickUpHourOfDay=LATENIGHT") == 1 &&
        get_feature(features, "dayType=NORMAL_WEEKDAY") == 0 &&
        get_feature(features, "zone=217") == 0 &&
        get_feature(features, "city=Mumbai") == 0) {
        return {0, 0.720, 5, 644.0};
    }

    // Rule 6: NEGATIVE (82.1% confidence, 397.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 1 &&
        get_feature(features, "pickUpHourOfDay=LATENIGHT") == 0 &&
        get_feature(features, "city=Pune") == 0 &&
        get_feature(features, "city=Mumbai") == 1 &&
        get_feature(features, "booking_type=round_trip") == 0) {
        return {0, 0.821, 6, 397.0};
    }

    // Rule 7: NEGATIVE (88.6% confidence, 395.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 1 &&
        get_feature(features, "city=Delhi NCR") == 0 &&
        get_feature(features, "dayType=WEEKEND") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 0 &&
        get_feature(features, "city=Hyderabad") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_1") == 0) {
        return {0, 0.886, 7, 395.0};
    }

    // Rule 8: NEGATIVE (95.4% confidence, 326.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 1 &&
        get_feature(features, "city=Delhi NCR") == 0 &&
        get_feature(features, "dayType=WEEKEND") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 0 &&
        get_feature(features, "city=Hyderabad") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_1") == 1) {
        return {0, 0.954, 8, 326.0};
    }

    // Rule 9: NEGATIVE (81.9% confidence, 249.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 0 &&
        get_feature(features, "pickUpHourOfDay=LATENIGHT") == 1 &&
        get_feature(features, "dayType=NORMAL_WEEKDAY") == 1 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_1") == 0 &&
        get_feature(features, "city=Mumbai") == 0) {
        return {0, 0.819, 9, 249.0};
    }

    // Rule 10: NEGATIVE (94.2% confidence, 241.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 0 &&
        get_feature(features, "pickUpHourOfDay=LATENIGHT") == 1 &&
        get_feature(features, "dayType=NORMAL_WEEKDAY") == 1 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_1") == 1 &&
        get_feature(features, "zone=219") == 0) {
        return {0, 0.942, 10, 241.0};
    }

    // Rule 11: NEGATIVE (83.9% confidence, 230.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 0 &&
        get_feature(features, "pickUpHourOfDay=LATENIGHT") == 1 &&
        get_feature(features, "dayType=NORMAL_WEEKDAY") == 0 &&
        get_feature(features, "zone=217") == 0 &&
        get_feature(features, "city=Mumbai") == 1) {
        return {0, 0.839, 11, 230.0};
    }

    // Rule 12: NEGATIVE (61.2% confidence, 201.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 1 &&
        get_feature(features, "pickUpHourOfDay=LATENIGHT") == 0 &&
        get_feature(features, "city=Pune") == 1 &&
        get_feature(features, "booking_type=outstation") == 0 &&
        get_feature(features, "pickUpHourOfDay=MORNING") == 0) {
        return {0, 0.612, 12, 201.0};
    }

    // Rule 13: NEGATIVE (68.3% confidence, 189.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 1 &&
        get_feature(features, "pickUpHourOfDay=LATENIGHT") == 0 &&
        get_feature(features, "city=Pune") == 0 &&
        get_feature(features, "city=Mumbai") == 1 &&
        get_feature(features, "booking_type=round_trip") == 1) {
        return {0, 0.683, 13, 189.0};
    }

    // Rule 14: NEGATIVE (96.6% confidence, 116.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 0 &&
        get_feature(features, "pickUpHourOfDay=LATENIGHT") == 1 &&
        get_feature(features, "dayType=NORMAL_WEEKDAY") == 1 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_1") == 0 &&
        get_feature(features, "city=Mumbai") == 1) {
        return {0, 0.966, 14, 116.0};
    }

    // Rule 15: NEGATIVE (84.5% confidence, 97.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 0 &&
        get_feature(features, "pickUpHourOfDay=LATENIGHT") == 0 &&
        get_feature(features, "zone=221") == 1 &&
        get_feature(features, "booking_type=one_way_trip") == 0 &&
        get_feature(features, "estimated_usage_bins=GT_3_LTE_4_HOURS") == 0) {
        return {0, 0.845, 15, 97.0};
    }

    // Rule 16: NEGATIVE (96.9% confidence, 96.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 0 &&
        get_feature(features, "pickUpHourOfDay=LATENIGHT") == 1 &&
        get_feature(features, "dayType=NORMAL_WEEKDAY") == 1 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_1") == 1 &&
        get_feature(features, "zone=219") == 1) {
        return {0, 0.969, 16, 96.0};
    }

    // Rule 17: NEGATIVE (71.7% confidence, 92.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 1 &&
        get_feature(features, "pickUpHourOfDay=LATENIGHT") == 0 &&
        get_feature(features, "city=Pune") == 0 &&
        get_feature(features, "city=Mumbai") == 0 &&
        get_feature(features, "city=Hyderabad") == 1) {
        return {0, 0.717, 17, 92.0};
    }

    // Rule 18: NEGATIVE (61.1% confidence, 90.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 0 &&
        get_feature(features, "pickUpHourOfDay=LATENIGHT") == 0 &&
        get_feature(features, "zone=221") == 1 &&
        get_feature(features, "booking_type=one_way_trip") == 1) {
        return {0, 0.611, 18, 90.0};
    }

    // Rule 19: NEGATIVE (81.1% confidence, 90.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 1 &&
        get_feature(features, "pickUpHourOfDay=LATENIGHT") == 0 &&
        get_feature(features, "city=Pune") == 1 &&
        get_feature(features, "booking_type=outstation") == 0 &&
        get_feature(features, "pickUpHourOfDay=MORNING") == 1) {
        return {0, 0.811, 19, 90.0};
    }

    // Rule 20: NEGATIVE (74.4% confidence, 90.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 1 &&
        get_feature(features, "city=Delhi NCR") == 0 &&
        get_feature(features, "dayType=WEEKEND") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 0 &&
        get_feature(features, "city=Hyderabad") == 1) {
        return {0, 0.744, 20, 90.0};
    }

    // Rule 21: NEGATIVE (50.6% confidence, 85.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 1 &&
        get_feature(features, "pickUpHourOfDay=LATENIGHT") == 1 &&
        get_feature(features, "city=Mumbai") == 0) {
        return {0, 0.506, 21, 85.0};
    }

    // Rule 22: NEGATIVE (92.8% confidence, 83.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 1 &&
        get_feature(features, "city=Delhi NCR") == 0 &&
        get_feature(features, "dayType=WEEKEND") == 1 &&
        get_feature(features, "city=Chennai") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_4") == 0 &&
        get_feature(features, "zone=216") == 1) {
        return {0, 0.928, 22, 83.0};
    }

    // Rule 23: NEGATIVE (51.8% confidence, 83.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 1 &&
        get_feature(features, "city=Delhi NCR") == 0 &&
        get_feature(features, "dayType=WEEKEND") == 1 &&
        get_feature(features, "city=Chennai") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_4") == 1 &&
        get_feature(features, "city=Mumbai") == 1) {
        return {0, 0.518, 23, 83.0};
    }

    // Rule 24: NEGATIVE (60.9% confidence, 69.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 0 &&
        get_feature(features, "pickUpHourOfDay=LATENIGHT") == 0 &&
        get_feature(features, "zone=221") == 0 &&
        get_feature(features, "zone=194") == 1) {
        return {0, 0.609, 24, 69.0};
    }

    // Rule 25: NEGATIVE (60.6% confidence, 66.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 1 &&
        get_feature(features, "city=Delhi NCR") == 0 &&
        get_feature(features, "dayType=WEEKEND") == 1 &&
        get_feature(features, "city=Chennai") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_4") == 1 &&
        get_feature(features, "city=Mumbai") == 0) {
        return {0, 0.606, 25, 66.0};
    }

    // Rule 26: NEGATIVE (95.5% confidence, 66.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 1 &&
        get_feature(features, "city=Delhi NCR") == 0 &&
        get_feature(features, "dayType=WEEKEND") == 1 &&
        get_feature(features, "city=Chennai") == 1) {
        return {0, 0.955, 26, 66.0};
    }

    // Rule 27: NEGATIVE (96.8% confidence, 62.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 0 &&
        get_feature(features, "pickUpHourOfDay=LATENIGHT") == 1 &&
        get_feature(features, "dayType=NORMAL_WEEKDAY") == 0 &&
        get_feature(features, "zone=217") == 1) {
        return {0, 0.968, 27, 62.0};
    }

    // Rule 28: NEGATIVE (90.3% confidence, 62.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 1 &&
        get_feature(features, "pickUpHourOfDay=LATENIGHT") == 0 &&
        get_feature(features, "city=Pune") == 1 &&
        get_feature(features, "booking_type=outstation") == 1) {
        return {0, 0.903, 28, 62.0};
    }

    // Rule 29: NEGATIVE (77.0% confidence, 61.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 0 &&
        get_feature(features, "pickUpHourOfDay=LATENIGHT") == 0 &&
        get_feature(features, "zone=221") == 1 &&
        get_feature(features, "booking_type=one_way_trip") == 0 &&
        get_feature(features, "estimated_usage_bins=GT_3_LTE_4_HOURS") == 1) {
        return {0, 0.770, 29, 61.0};
    }

    // Rule 30: POSITIVE (63.3% confidence, 60.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 1 &&
        get_feature(features, "city=Delhi NCR") == 1 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_2") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_4") == 0) {
        return {1, 0.633, 30, 60.0};
    }

    // Rule 31: NEGATIVE (68.4% confidence, 57.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 1 &&
        get_feature(features, "city=Delhi NCR") == 0 &&
        get_feature(features, "dayType=WEEKEND") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 1) {
        return {0, 0.684, 31, 57.0};
    }

    // Rule 32: NEGATIVE (52.6% confidence, 57.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 1 &&
        get_feature(features, "city=Delhi NCR") == 1 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_2") == 1) {
        return {0, 0.526, 32, 57.0};
    }

    // Rule 33: NEGATIVE (70.4% confidence, 54.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_5") == 1 &&
        get_feature(features, "pickUpHourOfDay=LATENIGHT") == 1 &&
        get_feature(features, "city=Mumbai") == 1) {
        return {0, 0.704, 33, 54.0};
    }

    // Rule 34: POSITIVE (51.0% confidence, 51.0 samples)
    if (get_feature(features, "pickUpHourOfDay=VERYLATE") == 1 &&
        get_feature(features, "city=Delhi NCR") == 1 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_2") == 0 &&
        get_feature(features, "zone_demand_popularity=POPULARITY_INDEX_4") == 1) {
        return {1, 0.510, 34, 51.0};
    }

    // Default fallback
    return {0, 0.500, -1, 0};
}

/**
 * Classify multiple samples
 * @param features_list Vector of feature maps
 * @return Vector of prediction results
 */
std::vector<PredictionResult> predict_batch(const std::vector<std::unordered_map<std::string, int>>& features_list) {
    std::vector<PredictionResult> results;
    results.reserve(features_list.size());
    
    for (const auto& features : features_list) {
        results.push_back(predict(features));
    }
    
    return results;
}

// Example usage
int main() {
    std::unordered_map<std::string, int> sample_features = {
    };
    
    auto result = predict(sample_features);
    std::cout << "Prediction: " << result.prediction << ", Confidence: " << result.confidence << std::endl;
    std::cout << "Total rules in model: 34" << std::endl;
    
    return 0;
}