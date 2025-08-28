"""
Test suite for CausalLLM core functionality
Tests the main CausalLLM class and core methods
"""
import pytest
import pandas as pd
import numpy as np
import warnings
from unittest.mock import Mock, patch, MagicMock
from causallm import CausalLLM
from causallm.core.causal_llm_core import CausalLLMCore
from causallm.enhanced_causallm import EnhancedCausalLLM

# Suppress warnings in tests
warnings.filterwarnings('ignore')


class TestCausalLLMCore:
    """Test CausalLLMCore class functionality."""
    
    def test_initialization_basic(self, variable_descriptions, dag_edges, causal_context):
        """Test basic initialization of CausalLLMCore."""
        core = CausalLLMCore(
            context=causal_context,
            variables=variable_descriptions,
            dag_edges=dag_edges
        )
        
        assert core.context == causal_context
        assert core.variables == variable_descriptions
        assert len(core.dag.graph.edges()) == len(dag_edges)
        assert core.llm_client is not None
    
    def test_initialization_invalid_dag(self, variable_descriptions, causal_context):
        """Test initialization with invalid DAG (contains cycle)."""
        cyclic_edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]  # Creates cycle
        
        with pytest.raises(ValueError, match="not a valid DAG"):
            CausalLLMCore(
                context=causal_context,
                variables=variable_descriptions,
                dag_edges=cyclic_edges
            )
    
    def test_simulate_do_basic(self, variable_descriptions, dag_edges, causal_context):
        """Test basic do-operation simulation."""
        core = CausalLLMCore(
            context=causal_context,
            variables=variable_descriptions,
            dag_edges=dag_edges
        )
        
        intervention = {"X1": "high_dose"}
        result = core.simulate_do(intervention)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "do(X1 := high_dose)" in result
    
    def test_simulate_do_with_question(self, variable_descriptions, dag_edges, causal_context):
        """Test do-operation with specific question."""
        core = CausalLLMCore(
            context=causal_context,
            variables=variable_descriptions,
            dag_edges=dag_edges
        )
        
        intervention = {"X1": "treatment"}
        question = "What is the effect on patient outcomes?"
        result = core.simulate_do(intervention, question)
        
        assert question in result
        assert "do(X1 := treatment)" in result
    
    def test_simulate_do_invalid_variable(self, variable_descriptions, dag_edges, causal_context):
        """Test do-operation with invalid variable."""
        core = CausalLLMCore(
            context=causal_context,
            variables=variable_descriptions,
            dag_edges=dag_edges
        )
        
        intervention = {"invalid_var": "value"}
        
        with pytest.raises(ValueError, match="not in base context"):
            core.simulate_do(intervention)
    
    def test_simulate_counterfactual(self, variable_descriptions, dag_edges, causal_context):
        """Test counterfactual simulation."""
        core = CausalLLMCore(
            context=causal_context,
            variables=variable_descriptions,
            dag_edges=dag_edges
        )
        
        factual = "Patient received standard treatment"
        intervention = "Patient received experimental treatment"
        
        result = core.simulate_counterfactual(factual, intervention)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert factual in result or intervention in result
    
    def test_generate_reasoning_prompt(self, variable_descriptions, dag_edges, causal_context):
        """Test reasoning prompt generation."""
        core = CausalLLMCore(
            context=causal_context,
            variables=variable_descriptions,
            dag_edges=dag_edges
        )
        
        task = "analyze_treatment_effect"
        result = core.generate_reasoning_prompt(task)
        
        assert isinstance(result, str)
        assert task in result
        assert str(dag_edges) in result
    
    @patch('causallm.core.causal_llm_core.get_llm_client')
    def test_custom_llm_client(self, mock_get_client, variable_descriptions, dag_edges, causal_context):
        """Test initialization with custom LLM client."""
        mock_client = Mock()
        
        core = CausalLLMCore(
            context=causal_context,
            variables=variable_descriptions,
            dag_edges=dag_edges,
            llm_client=mock_client
        )
        
        assert core.llm_client is mock_client
        mock_get_client.assert_not_called()
    
    def test_from_mcp_config(self, variable_descriptions, dag_edges, causal_context):
        """Test creation from MCP configuration."""
        mcp_config = {
            "context": causal_context,
            "variables": variable_descriptions,
            "dag_edges": dag_edges
        }
        
        core = CausalLLMCore.from_mcp_config(mcp_config)
        
        assert core.context == causal_context
        assert core.variables == variable_descriptions
        assert len(core.dag.graph.edges()) == len(dag_edges)
    
    def test_from_mcp_config_missing_fields(self):
        """Test MCP config with missing required fields."""
        incomplete_config = {"context": "test"}
        
        with pytest.raises(ValueError, match="missing required"):
            CausalLLMCore.from_mcp_config(incomplete_config)
    
    def test_create_causal_mcp_core(self, variable_descriptions, dag_edges, causal_context):
        """Test creation of MCP core representation."""
        core = CausalLLMCore(
            context=causal_context,
            variables=variable_descriptions,
            dag_edges=dag_edges
        )
        
        mcp_config = core.create_causal_mcp_core()
        
        assert mcp_config["context"] == causal_context
        assert mcp_config["variables"] == variable_descriptions
        assert "capabilities" in mcp_config
        assert mcp_config["capabilities"]["simulate_do"] is True
    
    @pytest.mark.asyncio
    async def test_enhanced_counterfactual_analysis(self, variable_descriptions, dag_edges, causal_context):
        """Test enhanced counterfactual analysis with Tier 1 capabilities."""
        with patch('causallm.core.causal_llm_core.CausalPromptEngine'), \
             patch('causallm.core.causal_llm_core.MultiAgentCausalAnalyzer'), \
             patch('causallm.core.causal_llm_core.create_rag_system'):
            
            core = CausalLLMCore(
                context=causal_context,
                variables=variable_descriptions,
                dag_edges=dag_edges
            )
            
            # Mock the RAG system response
            mock_rag_response = Mock()
            mock_rag_response.enhanced_context = "Enhanced context"
            mock_rag_response.retrieved_documents = []
            mock_rag_response.confidence_score = 0.8
            mock_rag_response.knowledge_gaps = []
            mock_rag_response.recommendations = []
            
            core.rag_system.enhance_query = Mock(return_value=mock_rag_response)
            
            # Mock intelligent prompting
            core.intelligent_prompting.generate_chain_of_thought_prompt = Mock(
                return_value="Enhanced prompt"
            )
            
            # Mock multi-agent analysis
            mock_collaborative_result = Mock()
            mock_collaborative_result.domain_expert_analysis = "Expert analysis"
            mock_collaborative_result.statistical_analysis = "Statistical analysis"
            mock_collaborative_result.skeptic_analysis = "Skeptic analysis"
            mock_collaborative_result.synthesized_conclusion = "Conclusion"
            mock_collaborative_result.confidence_score = 0.75
            mock_collaborative_result.key_assumptions = []
            mock_collaborative_result.recommendations = []
            
            core.multi_agent_analyzer.analyze_counterfactual = Mock(
                return_value=mock_collaborative_result
            )
            
            factual = "Patient received treatment A"
            intervention = "Patient received treatment B"
            
            result = await core.enhanced_counterfactual_analysis(factual, intervention)
            
            assert "factual_scenario" in result
            assert "intervention" in result
            assert "rag_analysis" in result
            assert "multi_agent_analysis" in result
            assert "overall_confidence" in result
            assert result["factual_scenario"] == factual
            assert result["intervention"] == intervention


class TestCausalLLMIntegration:
    """Test CausalLLM main class integration."""
    
    @patch('causallm.core.create_llm_client')
    def test_causallm_initialization(self, mock_create_client):
        """Test CausalLLM initialization."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        causallm = CausalLLM()
        
        assert causallm.llm_client is mock_client
        assert causallm.discovery_engine is not None
        assert causallm.dag_parser is not None
        assert causallm.do_operator is not None
        assert causallm.counterfactual_engine is not None
    
    @patch('causallm.core.create_llm_client')
    def test_causallm_custom_client(self, mock_create_client):
        """Test CausalLLM with custom client."""
        custom_client = Mock()
        
        causallm = CausalLLM(llm_client=custom_client)
        
        assert causallm.llm_client is custom_client
        mock_create_client.assert_not_called()
    
    @pytest.mark.asyncio
    @patch('causallm.core.create_llm_client')
    async def test_discover_causal_relationships(self, mock_create_client, sample_data):
        """Test causal relationship discovery."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        causallm = CausalLLM()
        
        # Mock the discovery engine
        mock_result = Mock()
        mock_result.discovered_edges = []
        mock_result.confidence_summary = {}
        
        causallm.discovery_engine.discover_relationships = Mock(return_value=mock_result)
        
        variables = list(sample_data.columns)
        result = await causallm.discover_causal_relationships(sample_data, variables)
        
        assert result is mock_result
        causallm.discovery_engine.discover_relationships.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('causallm.core.create_llm_client')
    async def test_estimate_causal_effect(self, mock_create_client, sample_data):
        """Test causal effect estimation."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        causallm = CausalLLM()
        
        # Mock the do operator
        mock_result = Mock()
        mock_result.effect_estimate = 0.5
        mock_result.confidence_interval = [0.3, 0.7]
        
        causallm.do_operator.estimate_effect = Mock(return_value=mock_result)
        
        result = await causallm.estimate_causal_effect(
            data=sample_data,
            treatment="X1", 
            outcome="X3"
        )
        
        assert result is mock_result
        causallm.do_operator.estimate_effect.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('causallm.core.create_llm_client')
    async def test_generate_counterfactuals(self, mock_create_client, sample_data):
        """Test counterfactual generation."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        causallm = CausalLLM()
        
        # Mock the counterfactual engine
        mock_result = Mock()
        mock_result.counterfactual_outcomes = ["Outcome A", "Outcome B"]
        
        causallm.counterfactual_engine.generate_counterfactuals = Mock(return_value=mock_result)
        
        intervention = {"X1": "high"}
        result = await causallm.generate_counterfactuals(sample_data, intervention)
        
        assert result is mock_result
        causallm.counterfactual_engine.generate_counterfactuals.assert_called_once()
    
    @patch('causallm.core.create_llm_client')
    def test_parse_causal_graph(self, mock_create_client):
        """Test causal graph parsing."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        causallm = CausalLLM()
        
        # Mock the DAG parser
        mock_result = Mock()
        causallm.dag_parser.parse = Mock(return_value=mock_result)
        
        graph_data = {"edges": [("A", "B"), ("B", "C")]}
        result = causallm.parse_causal_graph(graph_data)
        
        assert result is mock_result
        causallm.dag_parser.parse.assert_called_once_with(graph_data)
    
    @patch('causallm.core.create_llm_client')
    def test_get_enterprise_info_no_license(self, mock_create_client):
        """Test enterprise info when not licensed."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        causallm = CausalLLM()
        info = causallm.get_enterprise_info()
        
        assert info["licensed"] is False
        assert "benefits" in info
        assert len(info["benefits"]) > 0
        assert "https://causallm.com/enterprise" in info["info"]


class TestErrorHandling:
    """Test error handling in core functionality."""
    
    def test_empty_context_error(self, variable_descriptions, dag_edges):
        """Test error with empty context."""
        with pytest.raises(Exception):  # Should fail with empty context
            CausalLLMCore(
                context="",
                variables=variable_descriptions,
                dag_edges=dag_edges
            )
    
    def test_empty_variables_error(self, causal_context, dag_edges):
        """Test error with empty variables."""
        with pytest.raises(Exception):  # Should fail with empty variables
            CausalLLMCore(
                context=causal_context,
                variables={},
                dag_edges=dag_edges
            )
    
    def test_invalid_intervention_variable(self, variable_descriptions, dag_edges, causal_context):
        """Test error with invalid intervention variable."""
        core = CausalLLMCore(
            context=causal_context,
            variables=variable_descriptions,
            dag_edges=dag_edges
        )
        
        # Try to intervene on a variable not in the context
        with pytest.raises(ValueError):
            core.simulate_do({"nonexistent_var": "value"})
    
    def test_malformed_dag_edges(self, variable_descriptions, causal_context):
        """Test error with malformed DAG edges."""
        malformed_edges = [("A",), ("B", "C", "D")]  # Wrong tuple lengths
        
        with pytest.raises(Exception):
            CausalLLMCore(
                context=causal_context,
                variables=variable_descriptions,
                dag_edges=malformed_edges
            )