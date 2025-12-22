# Generated Decision Rules Summary
**Source Model**: theta_sketch_tree
**Generation Time**: unknown

## Rule Statistics
- **Total Rules**: 34
- **Positive Predictions**: 2 (5.9%)
- **Negative Predictions**: 32 (94.1%)
- **Average Conditions per Rule**: 5.4
- **Maximum Conditions**: 6
- **High Confidence Rules** (>80%): 17 (50.0%)

## Model Statistics
- **Total Training Samples**: 20,343.0
- **Average Rule Confidence**: 0.773
- **Feature Count**: 0
- **Classes**: [0, 1]

## Top 5 Rules by Sample Count
| Rule | Prediction | Confidence | Samples | Conditions |
|------|------------|------------|---------|------------|
| 1 | NEGATIVE | 0.904 | 12794.0 | 6 conditions |
| 2 | NEGATIVE | 0.890 | 1397.0 | 6 conditions |
| 3 | NEGATIVE | 0.744 | 996.0 | 6 conditions |
| 4 | NEGATIVE | 0.990 | 697.0 | 6 conditions |
| 5 | NEGATIVE | 0.720 | 644.0 | 6 conditions |

## Usage Instructions

### Python
```python
from decision_rules import predict_sample

features = {"feature1": 1, "feature2": 0}
result = predict_sample(features)
print(f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.3f}")
```

### Java
```java
Map<String, Integer> features = Map.of("feature1", 1, "feature2", 0);
var result = DecisionRules.predict(features);
System.out.println("Prediction: " + result.prediction() + ", Confidence: " + result.confidence());
```

### C++
```cpp
std::unordered_map<std::string, int> features = {{"feature1", 1}, {"feature2", 0}};
auto result = predict(features);
std::cout << "Prediction: " << result.prediction << ", Confidence: " << result.confidence;
```