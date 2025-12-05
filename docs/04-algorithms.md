# Algorithm Reference

## Mathematical Foundations

The Theta Sketch Decision Tree implements the CART (Classification and Regression Trees) algorithm with probabilistic sketch-based split evaluation. This document provides complete algorithmic details and mathematical formulations.

## Training Algorithm

### Main Training Workflow

```
FUNCTION fit(sketch_data, feature_mapping):
    """Build decision tree from theta sketches."""

    # Step 1: Extract and validate sketch data
    positive_sketches = sketch_data['positive']
    negative_sketches = sketch_data['negative'] (or computed from total)
    features = list(feature_mapping.keys())

    # Step 2: Initialize root node
    root_node = TreeNode(
        depth=0,
        n_samples=total_sample_count,
        class_counts=[negative_count, positive_count],
        impurity=criterion.compute_impurity(class_counts)
    )

    # Step 3: Recursively build tree
    build_tree_recursive(root_node, positive_sketches, negative_sketches,
                        features, depth=0)

    # Step 4: Apply pruning if specified
    if pruning_method:
        prune_tree(root_node, pruning_method)

    # Step 5: Compute feature importance
    compute_feature_importance(root_node)

    return root_node

FUNCTION build_tree_recursive(node, pos_sketches, neg_sketches, features, depth):
    """Recursive tree building with CART algorithm."""

    # Stopping criteria
    if should_stop_splitting(node, depth):
        make_leaf_node(node)
        return

    # Find best split
    best_split = find_best_split(node, pos_sketches, neg_sketches, features)

    if best_split is None or best_split.gain <= 0:
        make_leaf_node(node)
        return

    # Apply split
    node.is_leaf = False
    node.feature_idx = best_split.feature_idx
    node.feature_name = best_split.feature_name

    # Create child nodes
    left_node, right_node = create_child_nodes(node, best_split)

    # Recursive calls
    build_tree_recursive(left_node, best_split.left_pos_sketches,
                        best_split.left_neg_sketches, features, depth+1)
    build_tree_recursive(right_node, best_split.right_pos_sketches,
                        best_split.right_neg_sketches, features, depth+1)
```

### Split Finding Algorithm

```
FUNCTION find_best_split(node, pos_sketches, neg_sketches, features):
    """Find optimal feature split using sketch intersections."""

    best_split = None
    best_score = negative_infinity

    FOR each feature in features:
        # Get sketches for this feature
        pos_present_sketch, pos_absent_sketch = pos_sketches[feature]
        neg_present_sketch, neg_absent_sketch = neg_sketches[feature]

        # Compute sample counts via intersection
        pos_present_count = pos_present_sketch.get_estimate()
        pos_absent_count = pos_absent_sketch.get_estimate()
        neg_present_count = neg_present_sketch.get_estimate()
        neg_absent_count = neg_absent_sketch.get_estimate()

        # Validate counts
        if pos_present_count + pos_absent_count != node.positive_count:
            log_warning("Sketch count inconsistency detected")

        # Child node class distributions
        left_counts = [neg_present_count, pos_present_count]   # feature=1
        right_counts = [neg_absent_count, pos_absent_count]    # feature=0
        parent_counts = [node.negative_count, node.positive_count]

        # Evaluate split quality
        split_score = criterion.evaluate_split(parent_counts, left_counts, right_counts)

        # Track best split
        if split_score > best_score:
            best_score = split_score
            best_split = SplitInfo(
                feature_idx=feature_mapping[feature],
                feature_name=feature,
                score=split_score,
                left_counts=left_counts,
                right_counts=right_counts,
                left_sketches=(pos_present_sketch, neg_present_sketch),
                right_sketches=(pos_absent_sketch, neg_absent_sketch)
            )

    return best_split

FUNCTION should_stop_splitting(node, depth):
    """Evaluate stopping criteria."""

    # Maximum depth reached
    if depth >= max_depth:
        return True

    # Minimum samples for split
    if node.n_samples < min_samples_split:
        return True

    # Minimum samples in leaf (projected)
    if any(child_count < min_samples_leaf for child_count in projected_child_sizes):
        return True

    # Pure node (single class)
    if node.impurity <= 0.0:
        return True

    # Minimum impurity decrease
    if projected_impurity_decrease < min_impurity_decrease:
        return True

    return False
```

## Split Criteria Mathematics

### Gini Impurity

**Formula:**
```
Gini(p) = 1 - Σ(p_i²) for i in classes

where p_i = proportion of class i in node
```

**Implementation:**
```
FUNCTION gini_impurity(class_counts):
    total = sum(class_counts)
    if total == 0:
        return 0.0

    proportions = [count / total for count in class_counts]
    return 1.0 - sum(p² for p in proportions)

FUNCTION evaluate_gini_split(parent_counts, left_counts, right_counts):
    # Compute impurities
    parent_gini = gini_impurity(parent_counts)
    left_gini = gini_impurity(left_counts)
    right_gini = gini_impurity(right_counts)

    # Weighted average of child impurities
    total_samples = sum(parent_counts)
    left_weight = sum(left_counts) / total_samples
    right_weight = sum(right_counts) / total_samples
    weighted_impurity = left_weight * left_gini + right_weight * right_gini

    # Information gain (returned as negative for maximization)
    gain = parent_gini - weighted_impurity
    return -weighted_impurity  # Negative because we minimize
```

### Entropy (Information Gain)

**Formula:**
```
Entropy(p) = -Σ(p_i * log₂(p_i)) for i in classes

Information Gain = H(parent) - Σ(|child_i|/|parent| * H(child_i))
```

**Implementation:**
```
FUNCTION entropy(class_counts):
    total = sum(class_counts)
    if total == 0:
        return 0.0

    entropy_value = 0.0
    for count in class_counts:
        if count > 0:
            p = count / total
            entropy_value -= p * log2(p)

    return entropy_value

FUNCTION evaluate_entropy_split(parent_counts, left_counts, right_counts):
    parent_entropy = entropy(parent_counts)
    left_entropy = entropy(left_counts)
    right_entropy = entropy(right_counts)

    total_samples = sum(parent_counts)
    left_weight = sum(left_counts) / total_samples
    right_weight = sum(right_counts) / total_samples

    weighted_entropy = left_weight * left_entropy + right_weight * right_entropy
    gain = parent_entropy - weighted_entropy

    return -weighted_entropy  # Negative for consistency with other criteria
```

### Gain Ratio

**Formula:**
```
Gain Ratio = Information Gain / Split Information

Split Information = -Σ(|child_i|/|parent| * log₂(|child_i|/|parent|))
```

**Implementation:**
```
FUNCTION evaluate_gain_ratio_split(parent_counts, left_counts, right_counts):
    # Compute information gain
    parent_entropy = entropy(parent_counts)
    left_entropy = entropy(left_counts)
    right_entropy = entropy(right_counts)

    total_samples = sum(parent_counts)
    left_weight = sum(left_counts) / total_samples
    right_weight = sum(right_counts) / total_samples

    weighted_entropy = left_weight * left_entropy + right_weight * right_entropy
    information_gain = parent_entropy - weighted_entropy

    # Compute split information
    split_info = 0.0
    for weight in [left_weight, right_weight]:
        if weight > 0:
            split_info -= weight * log2(weight)

    # Avoid division by zero
    if split_info == 0:
        return 0.0

    gain_ratio = information_gain / split_info
    return gain_ratio
```

### Binomial Test

**Formula:**
```
Binomial Test evaluates statistical significance of observed vs expected distributions

p-value = 2 * min(P(X ≤ k), P(X ≥ k)) where X ~ Binomial(n, p₀)
```

**Implementation:**
```
from scipy.stats import binom

FUNCTION evaluate_binomial_split(parent_counts, left_counts, right_counts):
    total_positive = parent_counts[1]
    total_samples = sum(parent_counts)
    expected_prob = total_positive / total_samples if total_samples > 0 else 0.5

    # Test left child
    left_samples = sum(left_counts)
    left_positive = left_counts[1]

    if left_samples > 0:
        # Two-tailed binomial test
        p_value_left = 2 * min(
            binom.cdf(left_positive, left_samples, expected_prob),
            1 - binom.cdf(left_positive - 1, left_samples, expected_prob)
        )
    else:
        p_value_left = 1.0

    # Test right child similarly
    right_samples = sum(right_counts)
    right_positive = right_counts[1]

    if right_samples > 0:
        p_value_right = 2 * min(
            binom.cdf(right_positive, right_samples, expected_prob),
            1 - binom.cdf(right_positive - 1, right_samples, expected_prob)
        )
    else:
        p_value_right = 1.0

    # Return negative log p-value (higher values = more significant)
    combined_p_value = min(p_value_left, p_value_right)
    return -math.log(max(combined_p_value, 1e-10))  # Avoid log(0)
```

### Chi-Square Test

**Formula:**
```
χ² = Σ((Observed - Expected)² / Expected) for all cells

Expected frequency = (row_total * column_total) / grand_total
```

**Implementation:**
```
FUNCTION evaluate_chi_square_split(parent_counts, left_counts, right_counts):
    # Create contingency table
    #           Feature=1  Feature=0  Total
    # Class=0   left[0]   right[0]   parent[0]
    # Class=1   left[1]   right[1]   parent[1]
    # Total     sum(left) sum(right) sum(parent)

    left_total = sum(left_counts)
    right_total = sum(right_counts)
    grand_total = sum(parent_counts)

    if grand_total == 0 or left_total == 0 or right_total == 0:
        return 0.0

    chi_square = 0.0

    # For each cell in the 2x2 contingency table
    for class_idx in [0, 1]:
        for feature_val, counts in [(1, left_counts), (0, right_counts)]:
            observed = counts[class_idx]
            feature_total = sum(counts)
            class_total = parent_counts[class_idx]

            expected = (feature_total * class_total) / grand_total

            if expected > 0:
                chi_square += (observed - expected) ** 2 / expected

    return chi_square
```

## Tree Traversal Algorithm

### Prediction Algorithm

```
FUNCTION predict(X_binary, tree_root):
    """Predict class labels for binary input matrix."""

    n_samples, n_features = X_binary.shape
    predictions = zeros(n_samples, dtype=int)

    FOR sample_idx in range(n_samples):
        sample = X_binary[sample_idx]
        node = tree_root

        # Traverse tree to leaf
        WHILE not node.is_leaf:
            feature_value = sample[node.feature_idx]

            # Handle missing values (-1) with majority vote
            if feature_value == -1:
                feature_value = node.majority_vote_for_missing()

            # Navigate based on feature value
            if feature_value == 1:  # Feature present
                node = node.left_child
            else:  # Feature absent (0)
                node = node.right_child

        # Assign leaf prediction
        predictions[sample_idx] = node.predicted_class

    return predictions

FUNCTION predict_proba(X_binary, tree_root):
    """Predict class probabilities for binary input matrix."""

    n_samples, n_features = X_binary.shape
    probabilities = zeros((n_samples, 2), dtype=float)

    FOR sample_idx in range(n_samples):
        sample = X_binary[sample_idx]
        node = tree_root

        # Traverse to leaf
        WHILE not node.is_leaf:
            feature_value = sample[node.feature_idx]

            if feature_value == -1:
                feature_value = node.majority_vote_for_missing()

            if feature_value == 1:
                node = node.left_child
            else:
                node = node.right_child

        # Assign leaf probabilities
        total_samples = sum(node.class_counts)
        probabilities[sample_idx, 0] = node.class_counts[0] / total_samples
        probabilities[sample_idx, 1] = node.class_counts[1] / total_samples

    return probabilities
```

### Missing Value Handling

```
FUNCTION majority_vote_for_missing(node):
    """Determine feature value for missing data using majority vote."""

    left_samples = sum(node.left_child.class_counts) if node.left_child else 0
    right_samples = sum(node.right_child.class_counts) if node.right_child else 0

    # Return majority path (1 for left/present, 0 for right/absent)
    return 1 if left_samples >= right_samples else 0
```

## Pruning Algorithms

### Cost-Complexity Pruning

**Algorithm:**
```
FUNCTION cost_complexity_pruning(tree_root, min_impurity_decrease):
    """Prune tree using cost-complexity algorithm."""

    REPEAT:
        pruning_candidates = []

        # Find all internal nodes
        FOR each internal_node in tree:
            # Compute cost-complexity measure
            current_error = node.n_samples * node.impurity

            # Error if node becomes leaf
            leaf_error = node.n_samples * compute_leaf_impurity(node.class_counts)

            # Error of subtree
            subtree_error = compute_subtree_error(node)

            # Cost-complexity measure
            alpha = (leaf_error - subtree_error) / (count_leaves(node) - 1)

            if alpha <= min_impurity_decrease:
                pruning_candidates.append((alpha, node))

        # Prune node with smallest alpha (least beneficial)
        if pruning_candidates:
            _, node_to_prune = min(pruning_candidates, key=lambda x: x[0])
            prune_to_leaf(node_to_prune)
        else:
            break  # No more pruning candidates

    return tree_root
```

### Validation Pruning

**Algorithm:**
```
FUNCTION validation_pruning(tree_root, X_validation, y_validation):
    """Prune using held-out validation set."""

    best_tree = deep_copy(tree_root)
    best_accuracy = evaluate_accuracy(best_tree, X_validation, y_validation)

    pruning_queue = [tree_root]

    WHILE pruning_queue is not empty:
        current_tree = pruning_queue.pop()

        # Try pruning each internal node
        FOR each internal_node in current_tree:
            pruned_tree = deep_copy(current_tree)
            prune_to_leaf(pruned_tree, internal_node)

            accuracy = evaluate_accuracy(pruned_tree, X_validation, y_validation)

            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                best_tree = pruned_tree
                pruning_queue.append(pruned_tree)

    return best_tree
```

## Feature Importance Algorithm

### Mean Decrease in Impurity (MDI)

**Algorithm:**
```
FUNCTION compute_feature_importance(tree_root, n_features):
    """Compute feature importance using weighted impurity decrease."""

    importance = zeros(n_features)
    total_samples = tree_root.n_samples

    # Traverse all nodes in tree
    FOR each node in tree:
        if not node.is_leaf:
            # Compute impurity decrease for this split
            parent_impurity = node.impurity
            left_child = node.left_child
            right_child = node.right_child

            left_weight = left_child.n_samples / node.n_samples
            right_weight = right_child.n_samples / node.n_samples

            weighted_child_impurity = (left_weight * left_child.impurity +
                                     right_weight * right_child.impurity)

            impurity_decrease = parent_impurity - weighted_child_impurity

            # Weight by sample proportion
            sample_weight = node.n_samples / total_samples
            weighted_importance = sample_weight * impurity_decrease

            # Add to feature importance (ensure non-negative)
            feature_idx = node.feature_idx
            importance[feature_idx] += max(0.0, weighted_importance)

    # Normalize to sum to 1
    total_importance = sum(importance)
    if total_importance > 0:
        importance = importance / total_importance

    return importance
```

## Complexity Analysis

### Training Complexity

- **Time**: O(n_features × n_nodes × log(sketch_size))
  - n_features: Number of binary features
  - n_nodes: Number of tree nodes (≤ 2^depth)
  - log(sketch_size): Sketch intersection cost

- **Space**: O(n_features × n_sketches + n_nodes)
  - Sketch storage: O(n_features × n_sketches)
  - Tree structure: O(n_nodes)

### Inference Complexity

- **Time**: O(n_samples × tree_depth)
  - n_samples: Batch size
  - tree_depth: Average depth of tree (typically O(log n_features))

- **Space**: O(n_samples + tree_depth)
  - Input storage: O(n_samples)
  - Traversal stack: O(tree_depth)

## Algorithm Properties

### Theoretical Guarantees

1. **Convergence**: CART algorithm converges to locally optimal tree
2. **Consistency**: Given sufficient data, splits approach optimal splits
3. **Sketch Accuracy**: Theta sketches provide unbiased cardinality estimates
4. **Pruning Optimality**: Cost-complexity pruning finds optimal subtree sequence

### Performance Characteristics

1. **Scalability**: Linear in number of features and samples
2. **Memory Efficiency**: Sublinear memory growth with dataset size
3. **Robustness**: Handles missing values and noisy data gracefully
4. **Interpretability**: Tree structure provides clear decision rules

### Limitations

1. **Binary Features**: Only supports binary (0/1) feature encoding
2. **Sketch Approximation**: Estimates may have small errors due to sketching
3. **Imbalanced Data**: May require careful tuning for highly imbalanced datasets
4. **Feature Interactions**: Limited ability to capture complex feature interactions

---

## Next Steps

- **API Details**: See [API Reference](05-api-reference.md) for implementation specifics
- **Performance**: Review [Performance Guide](06-performance.md) for optimization techniques
- **Testing**: Check [Testing Guide](07-testing.md) for validation strategies
- **Implementation**: Examine source code for detailed implementations