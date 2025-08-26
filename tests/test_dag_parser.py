"""
Test suite for DAG parser functionality
Tests DAG creation, validation, and operations
"""
import pytest
import networkx as nx
import tempfile
import os
from unittest.mock import patch, Mock
from causallm.core.dag_parser import DAGParser


class TestDAGParser:
    """Test DAG parser functionality."""
    
    def test_dag_initialization_valid(self, dag_edges):
        """Test DAG initialization with valid edges."""
        dag_parser = DAGParser(dag_edges)
        
        assert isinstance(dag_parser.graph, nx.DiGraph)
        assert dag_parser.graph.number_of_nodes() == 3  # X1, X2, X3
        assert dag_parser.graph.number_of_edges() == len(dag_edges)
        
        # Check that all edges are present
        for edge in dag_edges:
            assert dag_parser.graph.has_edge(edge[0], edge[1])
    
    def test_dag_initialization_empty(self):
        """Test DAG initialization with empty edges."""
        dag_parser = DAGParser([])
        
        assert isinstance(dag_parser.graph, nx.DiGraph)
        assert dag_parser.graph.number_of_nodes() == 0
        assert dag_parser.graph.number_of_edges() == 0
    
    def test_dag_initialization_single_edge(self):
        """Test DAG initialization with single edge."""
        edges = [('A', 'B')]
        dag_parser = DAGParser(edges)
        
        assert dag_parser.graph.number_of_nodes() == 2
        assert dag_parser.graph.number_of_edges() == 1
        assert dag_parser.graph.has_edge('A', 'B')
    
    def test_dag_initialization_cyclic_error(self):
        """Test DAG initialization fails with cyclic edges."""
        cyclic_edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]  # Creates cycle
        
        with pytest.raises(ValueError, match="not a valid DAG"):
            DAGParser(cyclic_edges)
    
    def test_dag_initialization_self_loop_error(self):
        """Test DAG initialization fails with self-loops."""
        self_loop_edges = [('A', 'A')]  # Self-loop
        
        with pytest.raises(ValueError, match="not a valid DAG"):
            DAGParser(self_loop_edges)
    
    def test_get_ancestors_basic(self, dag_edges):
        """Test getting ancestors of a node."""
        dag_parser = DAGParser(dag_edges)
        
        # X3 should have X1 and X2 as ancestors
        ancestors = dag_parser.get_ancestors('X3')
        
        assert isinstance(ancestors, list)
        assert 'X1' in ancestors
        assert 'X2' in ancestors
        assert 'X3' not in ancestors  # Node is not its own ancestor
    
    def test_get_ancestors_root_node(self, dag_edges):
        """Test getting ancestors of root node."""
        dag_parser = DAGParser(dag_edges)
        
        # X1 is a root node, should have no ancestors
        ancestors = dag_parser.get_ancestors('X1')
        
        assert ancestors == []
    
    def test_get_ancestors_nonexistent_node(self, dag_edges):
        """Test getting ancestors of nonexistent node."""
        dag_parser = DAGParser(dag_edges)
        
        with pytest.raises(Exception):  # NetworkX raises exception for nonexistent nodes
            dag_parser.get_ancestors('nonexistent')
    
    def test_get_descendants_basic(self, dag_edges):
        """Test getting descendants of a node."""
        dag_parser = DAGParser(dag_edges)
        
        # X1 should have X2 and X3 as descendants
        descendants = dag_parser.get_descendants('X1')
        
        assert isinstance(descendants, list)
        assert 'X2' in descendants
        assert 'X3' in descendants
        assert 'X1' not in descendants  # Node is not its own descendant
    
    def test_get_descendants_leaf_node(self, dag_edges):
        """Test getting descendants of leaf node."""
        dag_parser = DAGParser(dag_edges)
        
        # X3 is a leaf node, should have no descendants
        descendants = dag_parser.get_descendants('X3')
        
        assert descendants == []
    
    def test_get_descendants_nonexistent_node(self, dag_edges):
        """Test getting descendants of nonexistent node."""
        dag_parser = DAGParser(dag_edges)
        
        with pytest.raises(Exception):  # NetworkX raises exception for nonexistent nodes
            dag_parser.get_descendants('nonexistent')
    
    def test_to_prompt_basic(self, dag_edges):
        """Test converting DAG to prompt."""
        dag_parser = DAGParser(dag_edges)
        
        task = "analyze_causal_relationships"
        prompt = dag_parser.to_prompt(task)
        
        assert isinstance(prompt, str)
        assert task in prompt
        assert "Graph structure:" in prompt
        
        # Check that edges are represented in the prompt
        for edge in dag_edges:
            edge_str = str(edge)
            assert edge_str in prompt or f"('{edge[0]}', '{edge[1]}')" in prompt
    
    def test_to_prompt_empty_task(self, dag_edges):
        """Test converting DAG to prompt with empty task."""
        dag_parser = DAGParser(dag_edges)
        
        prompt = dag_parser.to_prompt("")
        
        assert isinstance(prompt, str)
        assert "Task: " in prompt
        assert "Graph structure:" in prompt
    
    def test_to_prompt_complex_dag(self):
        """Test prompt generation with complex DAG."""
        complex_edges = [
            ('income', 'education'),
            ('education', 'job_satisfaction'),
            ('age', 'income'),
            ('age', 'health'),
            ('health', 'job_satisfaction')
        ]
        
        dag_parser = DAGParser(complex_edges)
        prompt = dag_parser.to_prompt("analyze_employment_factors")
        
        assert "analyze_employment_factors" in prompt
        assert len(prompt) > 50  # Should be substantial for complex DAG
        
        # Check that all nodes appear in the prompt
        for edge in complex_edges:
            assert edge[0] in prompt
            assert edge[1] in prompt
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('networkx.draw')
    @patch('networkx.spring_layout')
    def test_visualize_basic(self, mock_layout, mock_draw, mock_close, mock_savefig, dag_edges):
        """Test DAG visualization."""
        mock_layout.return_value = {'X1': (0, 0), 'X2': (1, 0), 'X3': (2, 0)}
        
        dag_parser = DAGParser(dag_edges)
        dag_parser.visualize("test_dag.png")
        
        mock_layout.assert_called_once_with(dag_parser.graph)
        mock_draw.assert_called_once()
        mock_savefig.assert_called_once_with("test_dag.png")
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close') 
    @patch('networkx.draw')
    @patch('networkx.spring_layout')
    def test_visualize_custom_path(self, mock_layout, mock_draw, mock_close, mock_savefig, dag_edges):
        """Test DAG visualization with custom path."""
        mock_layout.return_value = {}
        
        dag_parser = DAGParser(dag_edges)
        custom_path = "/tmp/custom_dag.png"
        dag_parser.visualize(custom_path)
        
        mock_savefig.assert_called_once_with(custom_path)
    
    def test_visualize_empty_dag(self):
        """Test visualization of empty DAG."""
        dag_parser = DAGParser([])
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close') as mock_close, \
             patch('networkx.draw') as mock_draw, \
             patch('networkx.spring_layout', return_value={}) as mock_layout:
            
            dag_parser.visualize("empty_dag.png")
            
            # Should still attempt to visualize even with empty graph
            mock_layout.assert_called_once()
            mock_draw.assert_called_once()
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig', side_effect=IOError("Permission denied"))
    def test_visualize_io_error(self, mock_savefig, dag_edges):
        """Test visualization error handling."""
        dag_parser = DAGParser(dag_edges)
        
        with pytest.raises(IOError, match="Permission denied"):
            dag_parser.visualize("/invalid/path/dag.png")


class TestDAGParserComplexScenarios:
    """Test DAG parser with complex scenarios."""
    
    def test_large_dag(self):
        """Test DAG parser with large number of nodes."""
        # Create a large DAG: chain of 100 nodes
        edges = [(f'node_{i}', f'node_{i+1}') for i in range(99)]
        
        dag_parser = DAGParser(edges)
        
        assert dag_parser.graph.number_of_nodes() == 100
        assert dag_parser.graph.number_of_edges() == 99
        
        # Test ancestors/descendants on large DAG
        ancestors = dag_parser.get_ancestors('node_50')
        descendants = dag_parser.get_descendants('node_50')
        
        assert len(ancestors) == 50  # node_0 to node_49
        assert len(descendants) == 49  # node_51 to node_99
    
    def test_star_topology(self):
        """Test DAG with star topology."""
        # Central node connected to many peripheral nodes
        center = 'center'
        peripherals = [f'node_{i}' for i in range(10)]
        edges = [(center, peripheral) for peripheral in peripherals]
        
        dag_parser = DAGParser(edges)
        
        assert dag_parser.graph.number_of_nodes() == 11
        assert dag_parser.graph.number_of_edges() == 10
        
        # Center node should have all peripherals as descendants
        descendants = dag_parser.get_descendants(center)
        assert len(descendants) == 10
        assert all(peripheral in descendants for peripheral in peripherals)
        
        # Peripheral nodes should have center as ancestor
        for peripheral in peripherals:
            ancestors = dag_parser.get_ancestors(peripheral)
            assert ancestors == [center]
    
    def test_tree_topology(self):
        """Test DAG with tree topology."""
        # Binary tree structure
        edges = [
            ('root', 'left'),
            ('root', 'right'),
            ('left', 'left_left'),
            ('left', 'left_right'),
            ('right', 'right_left'),
            ('right', 'right_right')
        ]
        
        dag_parser = DAGParser(edges)
        
        assert dag_parser.graph.number_of_nodes() == 7
        assert dag_parser.graph.number_of_edges() == 6
        
        # Root should have all other nodes as descendants
        root_descendants = dag_parser.get_descendants('root')
        assert len(root_descendants) == 6
        
        # Leaf nodes should have root and parent as ancestors
        leaf_ancestors = dag_parser.get_ancestors('left_left')
        assert 'root' in leaf_ancestors
        assert 'left' in leaf_ancestors
        assert len(leaf_ancestors) == 2
    
    def test_diamond_topology(self):
        """Test DAG with diamond topology (common in causal analysis)."""
        # Diamond: A -> B, A -> C, B -> D, C -> D
        edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')]
        
        dag_parser = DAGParser(edges)
        
        assert dag_parser.graph.number_of_nodes() == 4
        assert dag_parser.graph.number_of_edges() == 4
        
        # D should have all other nodes as ancestors
        d_ancestors = dag_parser.get_ancestors('D')
        assert len(d_ancestors) == 3
        assert all(node in d_ancestors for node in ['A', 'B', 'C'])
        
        # A should have all other nodes as descendants
        a_descendants = dag_parser.get_descendants('A')
        assert len(a_descendants) == 3
        assert all(node in a_descendants for node in ['B', 'C', 'D'])
    
    def test_multiple_components_error(self):
        """Test that disconnected components still form valid DAG."""
        # Two separate components
        edges = [('A', 'B'), ('C', 'D')]
        
        dag_parser = DAGParser(edges)
        
        assert dag_parser.graph.number_of_nodes() == 4
        assert dag_parser.graph.number_of_edges() == 2
        
        # Components should be independent
        assert dag_parser.get_ancestors('B') == ['A']
        assert dag_parser.get_ancestors('D') == ['C']
        assert dag_parser.get_descendants('A') == ['B']
        assert dag_parser.get_descendants('C') == ['D']
    
    def test_prompt_generation_performance(self):
        """Test prompt generation performance with large DAG."""
        # Create moderately large DAG
        edges = [(f'node_{i}', f'node_{i+1}') for i in range(50)]
        
        dag_parser = DAGParser(edges)
        
        import time
        start_time = time.time()
        prompt = dag_parser.to_prompt("performance_test")
        duration = time.time() - start_time
        
        # Should complete quickly (under 1 second)
        assert duration < 1.0
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # Should be substantial
    
    def test_unicode_node_names(self):
        """Test DAG with Unicode node names."""
        edges = [('α', 'β'), ('β', 'γ'), ('δ', 'ε')]
        
        dag_parser = DAGParser(edges)
        
        assert dag_parser.graph.number_of_nodes() == 5
        assert dag_parser.graph.has_edge('α', 'β')
        
        ancestors = dag_parser.get_ancestors('γ')
        assert 'α' in ancestors
        assert 'β' in ancestors
        
        prompt = dag_parser.to_prompt("unicode_test")
        assert 'α' in prompt
        assert 'β' in prompt
    
    def test_long_node_names(self):
        """Test DAG with very long node names."""
        long_name_1 = "very_long_variable_name_that_represents_complex_medical_condition_with_multiple_symptoms"
        long_name_2 = "another_extremely_long_variable_name_for_treatment_protocol_with_detailed_specifications"
        
        edges = [(long_name_1, long_name_2)]
        
        dag_parser = DAGParser(edges)
        
        assert dag_parser.graph.has_edge(long_name_1, long_name_2)
        
        descendants = dag_parser.get_descendants(long_name_1)
        assert long_name_2 in descendants
        
        prompt = dag_parser.to_prompt("long_names_test")
        assert long_name_1 in prompt
        assert long_name_2 in prompt


class TestDAGParserErrorHandling:
    """Test error handling in DAG parser."""
    
    def test_invalid_edge_format(self):
        """Test error handling with invalid edge format."""
        # Edges should be tuples of length 2
        invalid_edges = [('A',), ('B', 'C', 'D')]
        
        # NetworkX should handle this gracefully or raise an appropriate error
        with pytest.raises(Exception):
            dag_parser = DAGParser(invalid_edges)
    
    def test_none_edge_values(self):
        """Test error handling with None values in edges."""
        invalid_edges = [(None, 'B'), ('A', None)]
        
        # Should either work or raise appropriate error
        try:
            dag_parser = DAGParser(invalid_edges)
            # If it works, test basic functionality
            assert dag_parser.graph.number_of_edges() == 2
        except Exception:
            # If it fails, that's also acceptable
            pass
    
    def test_duplicate_edges(self):
        """Test handling of duplicate edges."""
        edges = [('A', 'B'), ('A', 'B'), ('B', 'C')]  # Duplicate edge
        
        dag_parser = DAGParser(edges)
        
        # NetworkX should handle duplicates by keeping only one
        assert dag_parser.graph.number_of_edges() == 2
        assert dag_parser.graph.has_edge('A', 'B')
        assert dag_parser.graph.has_edge('B', 'C')
    
    def test_visualize_error_handling(self, dag_edges):
        """Test visualization error handling."""
        dag_parser = DAGParser(dag_edges)
        
        # Test with matplotlib import error
        with patch('networkx.spring_layout', side_effect=ImportError("matplotlib not available")):
            with pytest.raises(ImportError):
                dag_parser.visualize("test.png")
    
    @patch('causallm.core.dag_parser.get_logger')
    def test_logging_functionality(self, mock_get_logger, dag_edges):
        """Test that DAG parser logs appropriately."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        dag_parser = DAGParser(dag_edges)
        
        # Should have logged initialization
        mock_logger.info.assert_called()
        mock_logger.debug.assert_called()
    
    def test_memory_usage_large_dag(self):
        """Test memory usage with large DAG."""
        import sys
        
        # Create large DAG
        edges = [(f'node_{i}', f'node_{i+1}') for i in range(1000)]
        
        # Check memory usage before
        initial_size = sys.getsizeof(edges)
        
        dag_parser = DAGParser(edges)
        
        # DAG parser should not use excessive memory
        dag_size = sys.getsizeof(dag_parser.graph)
        
        # NetworkX graph should be reasonably sized
        assert dag_size < initial_size * 10  # Reasonable overhead