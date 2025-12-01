"""
Test interface protocols and abstract base classes.
"""

import pytest
import numpy as np
from theta_sketch_tree.interfaces import (
    BaseCriterion,
    BaseSplitFinder,
    BaseStoppingCriteria,
    ComponentFactory
)


class TestComponentFactory:
    """Test the component factory."""

    def test_create_criterion_valid(self):
        """Test creating valid criteria."""
        criteria_names = ['gini', 'entropy', 'gain_ratio', 'binomial', 'chi_square']
        for name in criteria_names:
            criterion = ComponentFactory.create_criterion(name)
            assert hasattr(criterion, 'compute_impurity')
            assert hasattr(criterion, 'evaluate_split')

    def test_create_criterion_invalid(self):
        """Test creating invalid criteria."""
        with pytest.raises(ValueError, match="Unknown criterion"):
            ComponentFactory.create_criterion('invalid_criterion')

    def test_create_split_finder_valid(self):
        """Test creating valid split finders."""
        algorithms = ['intersection', 'ratio_based']
        for algorithm in algorithms:
            split_finder = ComponentFactory.create_split_finder(algorithm)
            assert hasattr(split_finder, 'find_best_split')

    def test_create_split_finder_invalid(self):
        """Test creating invalid split finders."""
        with pytest.raises(ValueError, match="Unknown split algorithm"):
            ComponentFactory.create_split_finder('invalid_algorithm')

    def test_create_stopping_criteria_valid(self):
        """Test creating valid stopping criteria."""
        criteria = ComponentFactory.create_stopping_criteria(
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1
        )
        assert hasattr(criteria, 'should_stop')

    def test_create_stopping_criteria_with_params(self):
        """Test creating stopping criteria with different parameters."""
        criteria = ComponentFactory.create_stopping_criteria(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5
        )
        assert criteria.max_depth == 5
        assert criteria.min_samples_split == 10
        assert criteria.min_samples_leaf == 5


class TestAbstractBaseClasses:
    """Test abstract base classes cannot be instantiated."""

    def test_base_criterion_abstract(self):
        """Test BaseCriterion cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseCriterion()

    def test_base_split_finder_abstract(self):
        """Test BaseSplitFinder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseSplitFinder()

    def test_base_stopping_criteria_abstract(self):
        """Test BaseStoppingCriteria cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseStoppingCriteria()


class TestProtocolConformance:
    """Test that implementations conform to protocols."""

    def test_criterion_protocol_conformance(self):
        """Test that concrete criteria conform to protocol."""
        from theta_sketch_tree.criteria import GiniCriterion

        criterion = GiniCriterion()

        # Test required methods exist
        assert callable(getattr(criterion, 'compute_impurity', None))
        assert callable(getattr(criterion, 'evaluate_split', None))

        # Test methods work with basic inputs
        counts = np.array([50.0, 50.0])
        impurity = criterion.compute_impurity(counts)
        assert isinstance(impurity, (int, float))

    def test_split_finder_protocol_conformance(self):
        """Test that concrete split finders conform to protocol."""
        from theta_sketch_tree.split_finder import SplitFinder

        # Create with minimal parameters
        split_finder = SplitFinder(criterion='gini', verbose=0)

        # Test required method exists
        assert callable(getattr(split_finder, 'find_best_split', None))


class TestFactoryErrorHandling:
    """Test factory error handling and edge cases."""

    def test_factory_with_none_inputs(self):
        """Test factory behavior with None inputs."""
        with pytest.raises((ValueError, TypeError)):
            ComponentFactory.create_criterion(None)

        with pytest.raises((ValueError, TypeError)):
            ComponentFactory.create_split_finder(None)

    def test_factory_with_empty_string(self):
        """Test factory behavior with empty strings."""
        with pytest.raises(ValueError):
            ComponentFactory.create_criterion('')

        with pytest.raises(ValueError):
            ComponentFactory.create_split_finder('')

    def test_stopping_criteria_edge_cases(self):
        """Test stopping criteria with edge case values."""
        # Test with None values (should use defaults)
        criteria = ComponentFactory.create_stopping_criteria(
            max_depth=None,
            min_samples_split=1,
            min_samples_leaf=1
        )
        assert criteria.max_depth is None

        # Test with minimum values
        criteria = ComponentFactory.create_stopping_criteria(
            max_depth=1,
            min_samples_split=2,
            min_samples_leaf=1
        )
        assert criteria.max_depth == 1


class TestComponentIntegration:
    """Test that components work together correctly."""

    def test_criterion_and_split_finder_integration(self):
        """Test that criteria and split finders work together."""
        criterion = ComponentFactory.create_criterion('gini')
        split_finder = ComponentFactory.create_split_finder('intersection')

        # Both should be created successfully
        assert criterion is not None
        assert split_finder is not None

        # Split finder should be able to use criterion
        # (This is tested more thoroughly in integration tests)

    def test_all_component_combinations(self):
        """Test creating all valid component combinations."""
        criteria_names = ['gini', 'entropy', 'gain_ratio', 'binomial', 'chi_square']
        split_algorithms = ['intersection', 'ratio_based']

        for criterion_name in criteria_names:
            criterion = ComponentFactory.create_criterion(criterion_name)
            assert criterion is not None

            for algorithm in split_algorithms:
                split_finder = ComponentFactory.create_split_finder(algorithm)
                assert split_finder is not None

                # Test stopping criteria with different combinations
                stopping = ComponentFactory.create_stopping_criteria(
                    max_depth=3,
                    min_samples_split=2,
                    min_samples_leaf=1
                )
                assert stopping is not None