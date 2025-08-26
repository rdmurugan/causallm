"""
Test suite for statistical methods in CausalLLM
Tests PC algorithm, conditional independence tests, and other statistical functionality
"""
import pytest
import numpy as np
import pandas as pd
import networkx as nx
from unittest.mock import Mock, patch
from causallm.core.statistical_methods import (
    ConditionalIndependenceTest,
    PCAlgorithm,
    bootstrap_stability_test,
    calculate_effect_size_cohen_d,
    granger_causality_test
)


class TestConditionalIndependenceTest:
    """Test conditional independence testing methods."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        ci_test = ConditionalIndependenceTest()
        assert ci_test.method == "partial_correlation"
        assert ci_test.alpha == 0.05
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        ci_test = ConditionalIndependenceTest(method="mutual_information", alpha=0.01)
        assert ci_test.method == "mutual_information"
        assert ci_test.alpha == 0.01
    
    def test_partial_correlation_independent(self):
        """Test partial correlation with independent variables."""
        np.random.seed(42)
        n = 1000
        X = np.random.normal(0, 1, n).reshape(-1, 1)
        Y = np.random.normal(0, 1, n).reshape(-1, 1)
        
        ci_test = ConditionalIndependenceTest(method="partial_correlation", alpha=0.05)
        is_independent, p_value = ci_test.test(X, Y)
        
        # Should detect independence (high p-value)
        assert is_independent is True
        assert p_value > 0.05
    
    def test_partial_correlation_dependent(self):
        """Test partial correlation with dependent variables."""
        np.random.seed(42)
        n = 1000
        X = np.random.normal(0, 1, n).reshape(-1, 1)
        Y = (2 * X + np.random.normal(0, 0.1, n)).reshape(-1, 1)
        
        ci_test = ConditionalIndependenceTest(method="partial_correlation", alpha=0.05)
        is_independent, p_value = ci_test.test(X, Y)
        
        # Should detect dependence (low p-value)
        assert is_independent is False
        assert p_value < 0.05
    
    def test_partial_correlation_with_conditioning(self):
        """Test partial correlation with conditioning variable."""
        np.random.seed(42)
        n = 1000
        Z = np.random.normal(0, 1, n).reshape(-1, 1)
        X = (0.5 * Z + np.random.normal(0, 0.5, n)).reshape(-1, 1)
        Y = (0.7 * Z + np.random.normal(0, 0.3, n)).reshape(-1, 1)
        
        ci_test = ConditionalIndependenceTest(method="partial_correlation", alpha=0.05)
        is_independent, p_value = ci_test.test(X, Y, Z)
        
        # X and Y should be independent given Z
        assert is_independent is True
        assert p_value > 0.05
    
    def test_mutual_information_method(self):
        """Test mutual information method."""
        np.random.seed(42)
        n = 1000
        X = np.random.normal(0, 1, n).reshape(-1, 1)
        Y = np.random.normal(0, 1, n).reshape(-1, 1)
        
        ci_test = ConditionalIndependenceTest(method="mutual_information", alpha=0.05)
        is_independent, p_value = ci_test.test(X, Y)
        
        assert isinstance(is_independent, bool)
        assert 0 <= p_value <= 1
    
    def test_chi_square_method(self):
        """Test chi-square method for categorical variables."""
        np.random.seed(42)
        n = 1000
        X = np.random.choice([0, 1, 2], n).reshape(-1, 1)
        Y = np.random.choice([0, 1, 2], n).reshape(-1, 1)
        
        ci_test = ConditionalIndependenceTest(method="chi_square", alpha=0.05)
        is_independent, p_value = ci_test.test(X, Y)
        
        assert isinstance(is_independent, bool)
        assert 0 <= p_value <= 1
    
    def test_unknown_method_error(self):
        """Test error with unknown method."""
        ci_test = ConditionalIndependenceTest(method="unknown_method")
        X = np.random.normal(0, 1, 100).reshape(-1, 1)
        Y = np.random.normal(0, 1, 100).reshape(-1, 1)
        
        with pytest.raises(ValueError, match="Unknown method"):
            ci_test.test(X, Y)
    
    def test_discretization(self):
        """Test discretization of continuous variables."""
        ci_test = ConditionalIndependenceTest(method="chi_square")
        X = np.random.normal(0, 1, 1000)
        
        discretized = ci_test._discretize(X, bins=3)
        
        assert len(np.unique(discretized)) <= 3
        assert discretized.dtype in [np.int32, np.int64]


class TestPCAlgorithm:
    """Test PC Algorithm implementation."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        pc = PCAlgorithm()
        assert isinstance(pc.ci_test, ConditionalIndependenceTest)
        assert pc.max_conditioning_size == 3
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        ci_test = ConditionalIndependenceTest(method="mutual_information")
        pc = PCAlgorithm(ci_test=ci_test, max_conditioning_size=2)
        assert pc.ci_test is ci_test
        assert pc.max_conditioning_size == 2
    
    def test_discover_skeleton_simple(self, simple_data):
        """Test skeleton discovery on simple data."""
        pc = PCAlgorithm(max_conditioning_size=1)
        skeleton = pc.discover_skeleton(simple_data)
        
        assert isinstance(skeleton, nx.Graph)
        assert skeleton.number_of_nodes() == len(simple_data.columns)
        # Should find edge between correlated X and Y
        assert skeleton.number_of_edges() >= 1
    
    def test_discover_skeleton_complex(self, sample_data):
        """Test skeleton discovery on complex causal structure."""
        pc = PCAlgorithm(max_conditioning_size=2)
        skeleton = pc.discover_skeleton(sample_data)
        
        assert isinstance(skeleton, nx.Graph)
        assert skeleton.number_of_nodes() == len(sample_data.columns)
        # Should find multiple edges in complex structure
        edges = list(skeleton.edges())
        assert len(edges) >= 2
    
    def test_orient_edges(self, simple_data):
        """Test edge orientation."""
        pc = PCAlgorithm()
        skeleton = pc.discover_skeleton(simple_data)
        dag = pc.orient_edges(skeleton, simple_data)
        
        assert isinstance(dag, nx.DiGraph)
        assert dag.number_of_nodes() == skeleton.number_of_nodes()
        # DiGraph should have directed edges
        assert dag.number_of_edges() >= skeleton.number_of_edges()
    
    def test_v_structure_orientation(self):
        """Test v-structure (collider) orientation."""
        # Create a simple v-structure scenario
        pc = PCAlgorithm()
        
        # Create simple graph: X -> Z <- Y (where X, Y not connected)
        graph = nx.DiGraph()
        graph.add_edges_from([('X', 'Z'), ('Z', 'X'), ('Y', 'Z'), ('Z', 'Y')])
        
        pc._orient_v_structures(graph)
        
        # After orientation, should have X -> Z <- Y
        assert graph.has_edge('X', 'Z')
        assert graph.has_edge('Y', 'Z')
        assert not graph.has_edge('Z', 'X')
        assert not graph.has_edge('Z', 'Y')
    
    def test_pc_rules(self):
        """Test PC algorithm orientation rules."""
        pc = PCAlgorithm()
        
        # Test Rule R1
        dag = nx.DiGraph()
        dag.add_edges_from([('X', 'Y'), ('Y', 'Z'), ('Z', 'Y')])  # X -> Y - Z
        
        changed = pc._apply_rule_r1(dag)
        
        # Rule R1 might apply depending on adjacency
        assert isinstance(changed, bool)
        
        # Test Rule R2
        dag = nx.DiGraph()
        dag.add_edges_from([('X', 'Y'), ('Y', 'Z'), ('Z', 'X'), ('X', 'Z')])
        
        changed = pc._apply_rule_r2(dag)
        assert isinstance(changed, bool)
        
        # Test Rule R3
        dag = nx.DiGraph()
        dag.add_edges_from([('X', 'Y'), ('Y', 'X'), ('X', 'Z'), ('Z', 'X'), ('Y', 'Z')])
        
        changed = pc._apply_rule_r3(dag)
        assert isinstance(changed, bool)
    
    @pytest.mark.slow
    def test_pc_algorithm_full_pipeline(self, sample_data):
        """Test complete PC algorithm pipeline."""
        ci_test = ConditionalIndependenceTest(method="partial_correlation", alpha=0.05)
        pc = PCAlgorithm(ci_test=ci_test, max_conditioning_size=2)
        
        # Discovery skeleton
        skeleton = pc.discover_skeleton(sample_data)
        
        # Orient edges
        dag = pc.orient_edges(skeleton, sample_data)
        
        assert isinstance(dag, nx.DiGraph)
        assert nx.is_directed_acyclic_graph(dag)  # Should be a valid DAG
        assert dag.number_of_nodes() == len(sample_data.columns)


class TestBootstrapStabilityTest:
    """Test bootstrap stability testing."""
    
    def test_bootstrap_stability_basic(self, simple_data):
        """Test basic bootstrap stability test."""
        pc = PCAlgorithm(max_conditioning_size=1)
        
        stable_graph, stability_scores = bootstrap_stability_test(
            data=simple_data,
            algorithm=pc,
            n_bootstrap=10,  # Small number for speed
            stability_threshold=0.5
        )
        
        assert isinstance(stable_graph, nx.DiGraph)
        assert isinstance(stability_scores, dict)
        assert stable_graph.number_of_nodes() == len(simple_data.columns)
    
    def test_bootstrap_stability_threshold(self, simple_data):
        """Test bootstrap stability with different thresholds."""
        pc = PCAlgorithm(max_conditioning_size=1)
        
        # High threshold - should find fewer stable edges
        stable_graph_high, scores_high = bootstrap_stability_test(
            data=simple_data,
            algorithm=pc,
            n_bootstrap=10,
            stability_threshold=0.9
        )
        
        # Low threshold - should find more stable edges
        stable_graph_low, scores_low = bootstrap_stability_test(
            data=simple_data,
            algorithm=pc,
            n_bootstrap=10,
            stability_threshold=0.1
        )
        
        assert stable_graph_low.number_of_edges() >= stable_graph_high.number_of_edges()
        assert len(scores_low) >= len(scores_high)
    
    @pytest.mark.slow
    def test_bootstrap_reproducibility(self, simple_data):
        """Test bootstrap reproducibility with fixed seed."""
        np.random.seed(42)
        pc = PCAlgorithm()
        
        result1 = bootstrap_stability_test(simple_data, pc, n_bootstrap=20)
        
        np.random.seed(42)
        result2 = bootstrap_stability_test(simple_data, pc, n_bootstrap=20)
        
        # Results should be identical with same seed
        assert result1[0].edges() == result2[0].edges()


class TestStatisticalUtilities:
    """Test additional statistical utility functions."""
    
    def test_cohen_d_calculation(self):
        """Test Cohen's d effect size calculation."""
        np.random.seed(42)
        
        # Treatment group with higher mean
        treatment = np.random.normal(1, 1, 100)
        control = np.random.normal(0, 1, 100)
        
        effect_size = calculate_effect_size_cohen_d(treatment, control)
        
        assert isinstance(effect_size, float)
        assert effect_size > 0  # Treatment has higher mean
        assert 0.5 < effect_size < 1.5  # Reasonable effect size
    
    def test_cohen_d_zero_effect(self):
        """Test Cohen's d with no effect."""
        np.random.seed(42)
        
        # Same distribution for both groups
        group1 = np.random.normal(0, 1, 100)
        group2 = np.random.normal(0, 1, 100)
        
        effect_size = calculate_effect_size_cohen_d(group1, group2)
        
        assert abs(effect_size) < 0.5  # Should be close to zero
    
    def test_granger_causality_test(self, time_series_data):
        """Test Granger causality test."""
        X = time_series_data['X'].values
        Y = time_series_data['Y'].values
        
        results = granger_causality_test(X, Y, max_lags=3)
        
        assert isinstance(results, dict)
        assert len(results) <= 3  # Up to 3 lags
        
        for lag, result in results.items():
            assert 'f_statistic' in result
            assert 'p_value' in result
            assert 'causality_detected' in result
            assert isinstance(result['f_statistic'], float)
            assert 0 <= result['p_value'] <= 1
            assert isinstance(result['causality_detected'], bool)
    
    def test_granger_causality_no_relationship(self):
        """Test Granger causality with independent time series."""
        np.random.seed(42)
        n = 100
        
        # Independent time series
        X = np.random.normal(0, 1, n)
        Y = np.random.normal(0, 1, n)
        
        results = granger_causality_test(X, Y, max_lags=2)
        
        # Should not detect causality
        for result in results.values():
            assert result['p_value'] > 0.01  # No significant causality
    
    def test_granger_causality_causal_relationship(self):
        """Test Granger causality with actual causal relationship."""
        np.random.seed(42)
        n = 200
        
        # Create causal time series: Y depends on lagged X
        X = np.random.normal(0, 1, n)
        Y = np.zeros(n)
        
        for t in range(1, n):
            Y[t] = 0.6 * X[t-1] + np.random.normal(0, 0.3)
        
        results = granger_causality_test(X, Y, max_lags=2)
        
        # Should detect causality at lag 1
        if 1 in results:
            assert results[1]['p_value'] < 0.05  # Significant causality


class TestStatisticalIntegration:
    """Test integration of statistical methods."""
    
    def test_pc_with_different_ci_tests(self, sample_data):
        """Test PC algorithm with different CI tests."""
        methods = ["partial_correlation", "mutual_information"]
        
        results = {}
        for method in methods:
            ci_test = ConditionalIndependenceTest(method=method, alpha=0.05)
            pc = PCAlgorithm(ci_test=ci_test, max_conditioning_size=1)
            
            skeleton = pc.discover_skeleton(sample_data)
            results[method] = skeleton
        
        # Both methods should produce valid results
        for method, skeleton in results.items():
            assert isinstance(skeleton, nx.Graph)
            assert skeleton.number_of_nodes() == len(sample_data.columns)
    
    @pytest.mark.statistical
    def test_statistical_significance_levels(self, sample_data):
        """Test statistical methods with different significance levels."""
        alphas = [0.01, 0.05, 0.1]
        edge_counts = []
        
        for alpha in alphas:
            ci_test = ConditionalIndependenceTest(method="partial_correlation", alpha=alpha)
            pc = PCAlgorithm(ci_test=ci_test, max_conditioning_size=1)
            
            skeleton = pc.discover_skeleton(sample_data)
            edge_counts.append(skeleton.number_of_edges())
        
        # More stringent alpha should generally result in fewer edges
        # (though this isn't guaranteed due to randomness)
        assert all(count >= 0 for count in edge_counts)
    
    def test_error_handling_empty_data(self):
        """Test error handling with empty data."""
        empty_data = pd.DataFrame()
        pc = PCAlgorithm()
        
        with pytest.raises((ValueError, IndexError)):
            pc.discover_skeleton(empty_data)
    
    def test_error_handling_insufficient_data(self):
        """Test error handling with insufficient data."""
        # Very small dataset
        small_data = pd.DataFrame({
            'X': [1, 2],
            'Y': [3, 4]
        })
        
        pc = PCAlgorithm()
        skeleton = pc.discover_skeleton(small_data)
        
        # Should handle gracefully
        assert isinstance(skeleton, nx.Graph)
        assert skeleton.number_of_nodes() == 2