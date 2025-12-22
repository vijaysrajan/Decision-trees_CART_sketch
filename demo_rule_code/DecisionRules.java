// Generated decision rules for classification
// Source: theta_sketch_tree
// Total rules: 34
// Feature count: 0

import java.util.*;

public class DecisionRules {

    public record PredictionResult(int prediction, double confidence, int ruleId, int samples) {}

    /**
     * Classify a single sample using decision rules
     * @param features Feature map with binary values (0/1)
     * @return Prediction result with confidence and metadata
     */
    public static PredictionResult predict(Map<String, Integer> features) {

        // Rule 1: NEGATIVE (90.4% confidence, 12794.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 0 &&
            features.getOrDefault("pickUpHourOfDay=LATENIGHT", 0) == 0 &&
            features.getOrDefault("zone=221", 0) == 0 &&
            features.getOrDefault("zone=194", 0) == 0 &&
            features.getOrDefault("sla=Immediate", 0) == 0) {
            return new PredictionResult(0, 0.904, 1, 12794.0);
        }

        // Rule 2: NEGATIVE (89.0% confidence, 1397.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 1 &&
            features.getOrDefault("pickUpHourOfDay=LATENIGHT", 0) == 0 &&
            features.getOrDefault("city=Pune", 0) == 0 &&
            features.getOrDefault("city=Mumbai", 0) == 0 &&
            features.getOrDefault("city=Hyderabad", 0) == 0) {
            return new PredictionResult(0, 0.890, 2, 1397.0);
        }

        // Rule 3: NEGATIVE (74.4% confidence, 996.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 1 &&
            features.getOrDefault("city=Delhi NCR", 0) == 0 &&
            features.getOrDefault("dayType=WEEKEND", 0) == 1 &&
            features.getOrDefault("city=Chennai", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_4", 0) == 0 &&
            features.getOrDefault("zone=216", 0) == 0) {
            return new PredictionResult(0, 0.744, 3, 996.0);
        }

        // Rule 4: NEGATIVE (99.0% confidence, 697.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 0 &&
            features.getOrDefault("pickUpHourOfDay=LATENIGHT", 0) == 0 &&
            features.getOrDefault("zone=221", 0) == 0 &&
            features.getOrDefault("zone=194", 0) == 0 &&
            features.getOrDefault("sla=Immediate", 0) == 1) {
            return new PredictionResult(0, 0.990, 4, 697.0);
        }

        // Rule 5: NEGATIVE (72.0% confidence, 644.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 0 &&
            features.getOrDefault("pickUpHourOfDay=LATENIGHT", 0) == 1 &&
            features.getOrDefault("dayType=NORMAL_WEEKDAY", 0) == 0 &&
            features.getOrDefault("zone=217", 0) == 0 &&
            features.getOrDefault("city=Mumbai", 0) == 0) {
            return new PredictionResult(0, 0.720, 5, 644.0);
        }

        // Rule 6: NEGATIVE (82.1% confidence, 397.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 1 &&
            features.getOrDefault("pickUpHourOfDay=LATENIGHT", 0) == 0 &&
            features.getOrDefault("city=Pune", 0) == 0 &&
            features.getOrDefault("city=Mumbai", 0) == 1 &&
            features.getOrDefault("booking_type=round_trip", 0) == 0) {
            return new PredictionResult(0, 0.821, 6, 397.0);
        }

        // Rule 7: NEGATIVE (88.6% confidence, 395.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 1 &&
            features.getOrDefault("city=Delhi NCR", 0) == 0 &&
            features.getOrDefault("dayType=WEEKEND", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 0 &&
            features.getOrDefault("city=Hyderabad", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_1", 0) == 0) {
            return new PredictionResult(0, 0.886, 7, 395.0);
        }

        // Rule 8: NEGATIVE (95.4% confidence, 326.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 1 &&
            features.getOrDefault("city=Delhi NCR", 0) == 0 &&
            features.getOrDefault("dayType=WEEKEND", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 0 &&
            features.getOrDefault("city=Hyderabad", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_1", 0) == 1) {
            return new PredictionResult(0, 0.954, 8, 326.0);
        }

        // Rule 9: NEGATIVE (81.9% confidence, 249.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 0 &&
            features.getOrDefault("pickUpHourOfDay=LATENIGHT", 0) == 1 &&
            features.getOrDefault("dayType=NORMAL_WEEKDAY", 0) == 1 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_1", 0) == 0 &&
            features.getOrDefault("city=Mumbai", 0) == 0) {
            return new PredictionResult(0, 0.819, 9, 249.0);
        }

        // Rule 10: NEGATIVE (94.2% confidence, 241.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 0 &&
            features.getOrDefault("pickUpHourOfDay=LATENIGHT", 0) == 1 &&
            features.getOrDefault("dayType=NORMAL_WEEKDAY", 0) == 1 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_1", 0) == 1 &&
            features.getOrDefault("zone=219", 0) == 0) {
            return new PredictionResult(0, 0.942, 10, 241.0);
        }

        // Rule 11: NEGATIVE (83.9% confidence, 230.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 0 &&
            features.getOrDefault("pickUpHourOfDay=LATENIGHT", 0) == 1 &&
            features.getOrDefault("dayType=NORMAL_WEEKDAY", 0) == 0 &&
            features.getOrDefault("zone=217", 0) == 0 &&
            features.getOrDefault("city=Mumbai", 0) == 1) {
            return new PredictionResult(0, 0.839, 11, 230.0);
        }

        // Rule 12: NEGATIVE (61.2% confidence, 201.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 1 &&
            features.getOrDefault("pickUpHourOfDay=LATENIGHT", 0) == 0 &&
            features.getOrDefault("city=Pune", 0) == 1 &&
            features.getOrDefault("booking_type=outstation", 0) == 0 &&
            features.getOrDefault("pickUpHourOfDay=MORNING", 0) == 0) {
            return new PredictionResult(0, 0.612, 12, 201.0);
        }

        // Rule 13: NEGATIVE (68.3% confidence, 189.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 1 &&
            features.getOrDefault("pickUpHourOfDay=LATENIGHT", 0) == 0 &&
            features.getOrDefault("city=Pune", 0) == 0 &&
            features.getOrDefault("city=Mumbai", 0) == 1 &&
            features.getOrDefault("booking_type=round_trip", 0) == 1) {
            return new PredictionResult(0, 0.683, 13, 189.0);
        }

        // Rule 14: NEGATIVE (96.6% confidence, 116.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 0 &&
            features.getOrDefault("pickUpHourOfDay=LATENIGHT", 0) == 1 &&
            features.getOrDefault("dayType=NORMAL_WEEKDAY", 0) == 1 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_1", 0) == 0 &&
            features.getOrDefault("city=Mumbai", 0) == 1) {
            return new PredictionResult(0, 0.966, 14, 116.0);
        }

        // Rule 15: NEGATIVE (84.5% confidence, 97.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 0 &&
            features.getOrDefault("pickUpHourOfDay=LATENIGHT", 0) == 0 &&
            features.getOrDefault("zone=221", 0) == 1 &&
            features.getOrDefault("booking_type=one_way_trip", 0) == 0 &&
            features.getOrDefault("estimated_usage_bins=GT_3_LTE_4_HOURS", 0) == 0) {
            return new PredictionResult(0, 0.845, 15, 97.0);
        }

        // Rule 16: NEGATIVE (96.9% confidence, 96.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 0 &&
            features.getOrDefault("pickUpHourOfDay=LATENIGHT", 0) == 1 &&
            features.getOrDefault("dayType=NORMAL_WEEKDAY", 0) == 1 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_1", 0) == 1 &&
            features.getOrDefault("zone=219", 0) == 1) {
            return new PredictionResult(0, 0.969, 16, 96.0);
        }

        // Rule 17: NEGATIVE (71.7% confidence, 92.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 1 &&
            features.getOrDefault("pickUpHourOfDay=LATENIGHT", 0) == 0 &&
            features.getOrDefault("city=Pune", 0) == 0 &&
            features.getOrDefault("city=Mumbai", 0) == 0 &&
            features.getOrDefault("city=Hyderabad", 0) == 1) {
            return new PredictionResult(0, 0.717, 17, 92.0);
        }

        // Rule 18: NEGATIVE (61.1% confidence, 90.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 0 &&
            features.getOrDefault("pickUpHourOfDay=LATENIGHT", 0) == 0 &&
            features.getOrDefault("zone=221", 0) == 1 &&
            features.getOrDefault("booking_type=one_way_trip", 0) == 1) {
            return new PredictionResult(0, 0.611, 18, 90.0);
        }

        // Rule 19: NEGATIVE (81.1% confidence, 90.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 1 &&
            features.getOrDefault("pickUpHourOfDay=LATENIGHT", 0) == 0 &&
            features.getOrDefault("city=Pune", 0) == 1 &&
            features.getOrDefault("booking_type=outstation", 0) == 0 &&
            features.getOrDefault("pickUpHourOfDay=MORNING", 0) == 1) {
            return new PredictionResult(0, 0.811, 19, 90.0);
        }

        // Rule 20: NEGATIVE (74.4% confidence, 90.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 1 &&
            features.getOrDefault("city=Delhi NCR", 0) == 0 &&
            features.getOrDefault("dayType=WEEKEND", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 0 &&
            features.getOrDefault("city=Hyderabad", 0) == 1) {
            return new PredictionResult(0, 0.744, 20, 90.0);
        }

        // Rule 21: NEGATIVE (50.6% confidence, 85.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 1 &&
            features.getOrDefault("pickUpHourOfDay=LATENIGHT", 0) == 1 &&
            features.getOrDefault("city=Mumbai", 0) == 0) {
            return new PredictionResult(0, 0.506, 21, 85.0);
        }

        // Rule 22: NEGATIVE (92.8% confidence, 83.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 1 &&
            features.getOrDefault("city=Delhi NCR", 0) == 0 &&
            features.getOrDefault("dayType=WEEKEND", 0) == 1 &&
            features.getOrDefault("city=Chennai", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_4", 0) == 0 &&
            features.getOrDefault("zone=216", 0) == 1) {
            return new PredictionResult(0, 0.928, 22, 83.0);
        }

        // Rule 23: NEGATIVE (51.8% confidence, 83.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 1 &&
            features.getOrDefault("city=Delhi NCR", 0) == 0 &&
            features.getOrDefault("dayType=WEEKEND", 0) == 1 &&
            features.getOrDefault("city=Chennai", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_4", 0) == 1 &&
            features.getOrDefault("city=Mumbai", 0) == 1) {
            return new PredictionResult(0, 0.518, 23, 83.0);
        }

        // Rule 24: NEGATIVE (60.9% confidence, 69.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 0 &&
            features.getOrDefault("pickUpHourOfDay=LATENIGHT", 0) == 0 &&
            features.getOrDefault("zone=221", 0) == 0 &&
            features.getOrDefault("zone=194", 0) == 1) {
            return new PredictionResult(0, 0.609, 24, 69.0);
        }

        // Rule 25: NEGATIVE (60.6% confidence, 66.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 1 &&
            features.getOrDefault("city=Delhi NCR", 0) == 0 &&
            features.getOrDefault("dayType=WEEKEND", 0) == 1 &&
            features.getOrDefault("city=Chennai", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_4", 0) == 1 &&
            features.getOrDefault("city=Mumbai", 0) == 0) {
            return new PredictionResult(0, 0.606, 25, 66.0);
        }

        // Rule 26: NEGATIVE (95.5% confidence, 66.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 1 &&
            features.getOrDefault("city=Delhi NCR", 0) == 0 &&
            features.getOrDefault("dayType=WEEKEND", 0) == 1 &&
            features.getOrDefault("city=Chennai", 0) == 1) {
            return new PredictionResult(0, 0.955, 26, 66.0);
        }

        // Rule 27: NEGATIVE (96.8% confidence, 62.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 0 &&
            features.getOrDefault("pickUpHourOfDay=LATENIGHT", 0) == 1 &&
            features.getOrDefault("dayType=NORMAL_WEEKDAY", 0) == 0 &&
            features.getOrDefault("zone=217", 0) == 1) {
            return new PredictionResult(0, 0.968, 27, 62.0);
        }

        // Rule 28: NEGATIVE (90.3% confidence, 62.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 1 &&
            features.getOrDefault("pickUpHourOfDay=LATENIGHT", 0) == 0 &&
            features.getOrDefault("city=Pune", 0) == 1 &&
            features.getOrDefault("booking_type=outstation", 0) == 1) {
            return new PredictionResult(0, 0.903, 28, 62.0);
        }

        // Rule 29: NEGATIVE (77.0% confidence, 61.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 0 &&
            features.getOrDefault("pickUpHourOfDay=LATENIGHT", 0) == 0 &&
            features.getOrDefault("zone=221", 0) == 1 &&
            features.getOrDefault("booking_type=one_way_trip", 0) == 0 &&
            features.getOrDefault("estimated_usage_bins=GT_3_LTE_4_HOURS", 0) == 1) {
            return new PredictionResult(0, 0.770, 29, 61.0);
        }

        // Rule 30: POSITIVE (63.3% confidence, 60.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 1 &&
            features.getOrDefault("city=Delhi NCR", 0) == 1 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_2", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_4", 0) == 0) {
            return new PredictionResult(1, 0.633, 30, 60.0);
        }

        // Rule 31: NEGATIVE (68.4% confidence, 57.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 1 &&
            features.getOrDefault("city=Delhi NCR", 0) == 0 &&
            features.getOrDefault("dayType=WEEKEND", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 1) {
            return new PredictionResult(0, 0.684, 31, 57.0);
        }

        // Rule 32: NEGATIVE (52.6% confidence, 57.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 1 &&
            features.getOrDefault("city=Delhi NCR", 0) == 1 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_2", 0) == 1) {
            return new PredictionResult(0, 0.526, 32, 57.0);
        }

        // Rule 33: NEGATIVE (70.4% confidence, 54.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_5", 0) == 1 &&
            features.getOrDefault("pickUpHourOfDay=LATENIGHT", 0) == 1 &&
            features.getOrDefault("city=Mumbai", 0) == 1) {
            return new PredictionResult(0, 0.704, 33, 54.0);
        }

        // Rule 34: POSITIVE (51.0% confidence, 51.0 samples)
        if (features.getOrDefault("pickUpHourOfDay=VERYLATE", 0) == 1 &&
            features.getOrDefault("city=Delhi NCR", 0) == 1 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_2", 0) == 0 &&
            features.getOrDefault("zone_demand_popularity=POPULARITY_INDEX_4", 0) == 1) {
            return new PredictionResult(1, 0.510, 34, 51.0);
        }

        // Default fallback
        return new PredictionResult(0, 0.500, -1, 0);
    }

    /**
     * Classify multiple samples
     * @param featuresList List of feature maps
     * @return List of prediction results
     */
    public static List<PredictionResult> predictBatch(List<Map<String, Integer>> featuresList) {
        return featuresList.stream()
                          .map(DecisionRules::predict)
                          .toList();
    }

    // Example usage
    public static void main(String[] args) {
        Map<String, Integer> sampleFeatures = Map.of(
        );
        
        var result = predict(sampleFeatures);
        System.out.println("Prediction: " + result.prediction() + ", Confidence: " + result.confidence());
        System.out.println("Total rules in model: 34");
    }
}