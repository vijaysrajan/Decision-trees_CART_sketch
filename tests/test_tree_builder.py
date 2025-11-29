"""
Unit tests for tree builder logic.

Tests TreeBuilder class for decision tree construction using theta sketches.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from theta_sketch_tree.tree_builder import TreeBuilder
from theta_sketch_tree.criteria import GiniCriterion, EntropyCriterion
from theta_sketch_tree.tree_structure import TreeNode
from tests.test_mock_sketches import MockThetaSketch


class TestTreeBuilder:
    """Test TreeBuilder class functionality."""

    @pytest.fixture
    def gini_criterion(self):
        """Gini criterion for testing."""
        return GiniCriterion()

    @pytest.fixture
    def entropy_criterion(self):
        """Entropy criterion for testing."""
        return EntropyCriterion()

    @pytest.fixture
    def basic_builder(self, gini_criterion):
        """Basic tree builder with minimal configuration."""
        return TreeBuilder(
            criterion=gini_criterion,
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1
        )

    @pytest.fixture
    def verbose_builder(self, entropy_criterion):
        """Tree builder with verbose output enabled."""
        return TreeBuilder(
            criterion=entropy_criterion,
            max_depth=2,
            min_samples_split=5,
            min_samples_leaf=2,
            verbose=2
        )

    @pytest.fixture
    def simple_sketch_data(self):
        """Simple sketch data for testing."""
        # Create mock sketches for binary features
        pos_feature_a_present = MockThetaSketch(40)  # 40 positive samples with feature A
        pos_feature_a_absent = MockThetaSketch(60)   # 60 positive samples without feature A
        neg_feature_a_present = MockThetaSketch(10)  # 10 negative samples with feature A
        neg_feature_a_absent = MockThetaSketch(90)   # 90 negative samples without feature A

        pos_feature_b_present = MockThetaSketch(30)
        pos_feature_b_absent = MockThetaSketch(70)
        neg_feature_b_present = MockThetaSketch(20)
        neg_feature_b_absent = MockThetaSketch(80)

        return {
            'positive': {
                'feature_A': (pos_feature_a_present, pos_feature_a_absent),
                'feature_B': (pos_feature_b_present, pos_feature_b_absent),
            },
            'negative': {
                'feature_A': (neg_feature_a_present, neg_feature_a_absent),
                'feature_B': (neg_feature_b_present, neg_feature_b_absent),
            }
        }

    @pytest.fixture
    def perfect_split_data(self):
        """Sketch data that creates perfect splits (pure children)."""
        # Feature A perfectly separates classes
        pos_feature_a_present = MockThetaSketch(100)  # All positive samples have feature A
        pos_feature_a_absent = MockThetaSketch(0)     # No positive samples without feature A
        neg_feature_a_present = MockThetaSketch(0)    # No negative samples have feature A
        neg_feature_a_absent = MockThetaSketch(100)   # All negative samples without feature A

        return {
            'positive': {
                'feature_A': (pos_feature_a_present, pos_feature_a_absent),
            },
            'negative': {
                'feature_A': (neg_feature_a_present, neg_feature_a_absent),
            }
        }

    def test_initialization(self, gini_criterion):
        """Test TreeBuilder initialization."""
        builder = TreeBuilder(
            criterion=gini_criterion,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            verbose=1
        )

        assert builder.criterion is gini_criterion
        assert builder.max_depth == 5
        assert builder.min_samples_split == 10
        assert builder.min_samples_leaf == 5
        assert builder.verbose == 1
        assert builder.pruner is None
        assert builder.feature_mapping == {}

    def test_initialization_with_feature_mapping(self, gini_criterion):
        """Test initialization with feature mapping."""
        feature_mapping = {'feature_A': 0, 'feature_B': 1}
        builder = TreeBuilder(
            criterion=gini_criterion,
            feature_mapping=feature_mapping
        )
        assert builder.feature_mapping == feature_mapping

    def test_build_single_node_tree(self, basic_builder):
        """Test building tree with single node (no valid splits)."""
        # Create sketches where no splits are beneficial
        parent_pos = MockThetaSketch(50)
        parent_neg = MockThetaSketch(50)

        # Empty sketch data (no features)
        sketch_data = {'positive': {}, 'negative': {}}

        tree = basic_builder.build_tree(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            sketch_dict=sketch_data,
            feature_names=[],
            already_used=set(),
            depth=0
        )

        # Should create a leaf node
        assert tree.is_leaf
        assert tree.depth == 0
        assert tree.n_samples == 100
        assert tree.prediction in [0, 1]  # Should predict majority class
        assert tree.probabilities is not None

    def test_build_perfect_split_tree(self, basic_builder, perfect_split_data):
        """Test building tree with perfect split."""
        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(100)

        tree = basic_builder.build_tree(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            sketch_dict=perfect_split_data,
            feature_names=['feature_A'],
            already_used=set(),
            depth=0
        )

        # Should create split node
        assert not tree.is_leaf
        assert tree.depth == 0
        assert tree.feature_name == 'feature_A'
        assert tree.left is not None
        assert tree.right is not None

        # Children should be pure leaves
        assert tree.left.is_leaf
        assert tree.right.is_leaf
        assert tree.left.impurity == 0.0
        assert tree.right.impurity == 0.0

    def test_depth_stopping_criterion(self, basic_builder, simple_sketch_data):
        """Test that tree building stops at max depth."""
        # Set max depth to 1
        basic_builder.max_depth = 1

        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(100)

        tree = basic_builder.build_tree(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            sketch_dict=simple_sketch_data,
            feature_names=['feature_A', 'feature_B'],
            already_used=set(),
            depth=0
        )

        # Should split at root level
        if not tree.is_leaf:
            # But children should be leaves (max depth reached)
            assert tree.left.is_leaf
            assert tree.right.is_leaf
            assert tree.left.depth == 1
            assert tree.right.depth == 1

    def test_min_samples_split_criterion(self, gini_criterion, simple_sketch_data):
        """Test min_samples_split stopping criterion."""
        builder = TreeBuilder(
            criterion=gini_criterion,
            min_samples_split=150  # Higher than total samples
        )

        parent_pos = MockThetaSketch(50)
        parent_neg = MockThetaSketch(50)  # Total = 100 < 150

        tree = builder.build_tree(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            sketch_dict=simple_sketch_data,
            feature_names=['feature_A'],
            already_used=set(),
            depth=0
        )

        # Should create leaf due to insufficient samples
        assert tree.is_leaf

    def test_pure_node_stopping_criterion(self, basic_builder):
        """Test that pure nodes become leaves."""
        # Create pure positive class
        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(0)

        sketch_data = {
            'positive': {'feature_A': (MockThetaSketch(50), MockThetaSketch(50))},
            'negative': {'feature_A': (MockThetaSketch(0), MockThetaSketch(0))}
        }

        tree = basic_builder.build_tree(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            sketch_dict=sketch_data,
            feature_names=['feature_A'],
            already_used=set(),
            depth=0
        )

        # Should create leaf due to purity
        assert tree.is_leaf
        assert tree.impurity == 0.0
        assert tree.prediction == 1  # All positive samples

    def test_binary_feature_optimization(self, basic_builder, simple_sketch_data):
        """Test that already used features are skipped."""
        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(100)

        # Mark feature_A as already used
        already_used = {'feature_A'}

        tree = basic_builder.build_tree(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            sketch_dict=simple_sketch_data,
            feature_names=['feature_A', 'feature_B'],
            already_used=already_used,
            depth=0
        )

        # Should only consider feature_B
        if not tree.is_leaf:
            assert tree.feature_name == 'feature_B'

    def test_empty_child_handling(self, basic_builder):
        """Test handling of splits that create empty children."""
        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(100)

        # Create sketches that would result in empty children
        sketch_data = {
            'positive': {
                'bad_feature': (MockThetaSketch(0), MockThetaSketch(100))  # Empty left child
            },
            'negative': {
                'bad_feature': (MockThetaSketch(0), MockThetaSketch(100))  # Empty left child
            }
        }

        tree = basic_builder.build_tree(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            sketch_dict=sketch_data,
            feature_names=['bad_feature'],
            already_used=set(),
            depth=0
        )

        # Should create leaf since split is invalid
        assert tree.is_leaf

    def test_feature_mapping(self, gini_criterion, simple_sketch_data):
        """Test that feature mapping is used correctly."""
        feature_mapping = {'feature_A': 0, 'feature_B': 1}
        builder = TreeBuilder(
            criterion=gini_criterion,
            feature_mapping=feature_mapping
        )

        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(100)

        tree = builder.build_tree(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            sketch_dict=simple_sketch_data,
            feature_names=['feature_A', 'feature_B'],
            already_used=set(),
            depth=0
        )

        # Check that feature index is set correctly
        if not tree.is_leaf:
            expected_idx = feature_mapping.get(tree.feature_name, -1)
            assert tree.feature_idx == expected_idx

    def test_verbose_output(self, verbose_builder, simple_sketch_data, capsys):
        """Test verbose output during tree building."""
        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(100)

        tree = verbose_builder.build_tree(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            sketch_dict=simple_sketch_data,
            feature_names=['feature_A'],
            already_used=set(),
            depth=0
        )

        # Check that verbose output was generated
        captured = capsys.readouterr()
        assert "Depth 0:" in captured.out

    def test_different_criteria(self, simple_sketch_data):
        """Test tree building with different criteria."""
        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(100)

        # Test with Gini
        gini_builder = TreeBuilder(criterion=GiniCriterion())
        gini_tree = gini_builder.build_tree(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            sketch_dict=simple_sketch_data,
            feature_names=['feature_A'],
            already_used=set(),
            depth=0
        )

        # Test with Entropy
        entropy_builder = TreeBuilder(criterion=EntropyCriterion())
        entropy_tree = entropy_builder.build_tree(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            sketch_dict=simple_sketch_data,
            feature_names=['feature_A'],
            already_used=set(),
            depth=0
        )

        # Both should produce valid trees
        assert isinstance(gini_tree, TreeNode)
        assert isinstance(entropy_tree, TreeNode)

    def test_missing_feature_in_sketch_dict(self, basic_builder):
        """Test handling when feature is missing from sketch dict."""
        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(100)

        # Only provide feature_A, but request feature_B as well
        sketch_data = {
            'positive': {'feature_A': (MockThetaSketch(50), MockThetaSketch(50))},
            'negative': {'feature_A': (MockThetaSketch(50), MockThetaSketch(50))}
        }

        tree = basic_builder.build_tree(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            sketch_dict=sketch_data,
            feature_names=['feature_A', 'feature_B'],  # feature_B missing
            already_used=set(),
            depth=0
        )

        # Should still work, just skip missing feature
        assert isinstance(tree, TreeNode)

    def test_recursive_tree_structure(self, basic_builder, simple_sketch_data):
        """Test that recursive tree structure is built correctly."""
        parent_pos = MockThetaSketch(200)
        parent_neg = MockThetaSketch(200)

        tree = basic_builder.build_tree(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            sketch_dict=simple_sketch_data,
            feature_names=['feature_A', 'feature_B'],
            already_used=set(),
            depth=0
        )

        # Verify tree structure
        assert tree.depth == 0

        if not tree.is_leaf:
            # Check children
            assert tree.left.depth == 1
            assert tree.right.depth == 1
            assert tree.left.parent is tree
            assert tree.right.parent is tree

            # Check that already_used propagates correctly
            # (Implementation detail - hard to test directly)


class TestTreeBuilderEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def minimal_builder(self):
        """Minimal builder for edge case testing."""
        return TreeBuilder(criterion=GiniCriterion())

    def test_zero_samples(self, minimal_builder):
        """Test with zero samples."""
        parent_pos = MockThetaSketch(0)
        parent_neg = MockThetaSketch(0)

        tree = minimal_builder.build_tree(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            sketch_dict={'positive': {}, 'negative': {}},
            feature_names=[],
            already_used=set(),
            depth=0
        )

        assert tree.is_leaf
        assert tree.n_samples == 0

    def test_single_class_data(self, minimal_builder):
        """Test with only one class present."""
        # Only positive samples
        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(0)

        sketch_data = {
            'positive': {'feature_A': (MockThetaSketch(50), MockThetaSketch(50))},
            'negative': {'feature_A': (MockThetaSketch(0), MockThetaSketch(0))}
        }

        tree = minimal_builder.build_tree(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            sketch_dict=sketch_data,
            feature_names=['feature_A'],
            already_used=set(),
            depth=0
        )

        assert tree.is_leaf  # Should stop due to single class
        assert tree.prediction == 1

    def test_very_deep_tree(self):
        """Test with very deep tree settings."""
        builder = TreeBuilder(
            criterion=GiniCriterion(),
            max_depth=100  # Very deep
        )

        parent_pos = MockThetaSketch(1000)
        parent_neg = MockThetaSketch(1000)

        # Create sketch data with many features
        sketch_data = {'positive': {}, 'negative': {}}
        feature_names = []

        for i in range(10):
            feature_name = f'feature_{i}'
            feature_names.append(feature_name)
            sketch_data['positive'][feature_name] = (
                MockThetaSketch(500), MockThetaSketch(500)
            )
            sketch_data['negative'][feature_name] = (
                MockThetaSketch(500), MockThetaSketch(500)
            )

        tree = builder.build_tree(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            sketch_dict=sketch_data,
            feature_names=feature_names,
            already_used=set(),
            depth=0
        )

        assert isinstance(tree, TreeNode)



class TestTreeBuilderIntegration:
    """Integration tests for TreeBuilder with realistic scenarios."""

    def test_complete_tree_building_workflow(self):
        """Test complete tree building workflow."""
        builder = TreeBuilder(
            criterion=GiniCriterion(),
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1,
            feature_mapping={'age>30': 0, 'income>50k': 1}
        )

        # Create realistic sketch data
        sketch_data = {
            'positive': {
                'age>30': (MockThetaSketch(40), MockThetaSketch(60)),
                'income>50k': (MockThetaSketch(35), MockThetaSketch(65)),
            },
            'negative': {
                'age>30': (MockThetaSketch(30), MockThetaSketch(70)),
                'income>50k': (MockThetaSketch(25), MockThetaSketch(75)),
            }
        }

        parent_pos = MockThetaSketch(100)
        parent_neg = MockThetaSketch(100)

        tree = builder.build_tree(
            parent_sketch_pos=parent_pos,
            parent_sketch_neg=parent_neg,
            sketch_dict=sketch_data,
            feature_names=['age>30', 'income>50k'],
            already_used=set(),
            depth=0
        )

        # Validate tree structure
        assert isinstance(tree, TreeNode)
        assert tree.depth == 0
        assert tree.n_samples == 200

        # Tree should have meaningful structure
        if not tree.is_leaf:
            assert tree.feature_name in ['age>30', 'income>50k']
            assert tree.feature_idx in [0, 1]