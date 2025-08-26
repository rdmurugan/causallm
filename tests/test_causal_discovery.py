"""
Test suite for causal discovery algorithms
Tests PC Algorithm, LLM-guided discovery, and hybrid approaches
"""
import pytest
import numpy as np
import pandas as pd
import networkx as nx
import json
from unittest.mock import Mock, patch, AsyncMock
from causallm.core.causal_discovery import (
    DiscoveryMethod,
    ConfidenceLevel,
    CausalEdge,
    DiscoveryResult,
    LLMDiscoveryPrompt,
    PCAlgorithmEngine,
    LLMGuidedDiscoveryEngine,
    HybridLLMDiscoveryEngine,
    AdvancedCausalDiscovery,
    create_discovery_engine,
    discover_causal_structure
)


class TestCausalEdge:
    """Test CausalEdge dataclass."""
    
    def test_causal_edge_creation(self):
        """Test CausalEdge creation and attributes."""
        edge = CausalEdge(
            cause="X",
            effect="Y",
            confidence=0.8,
            confidence_level=ConfidenceLevel.HIGH,
            method=DiscoveryMethod.PC_ALGORITHM,
            reasoning="Strong statistical evidence"
        )
        
        assert edge.cause == "X"
        assert edge.effect == "Y"
        assert edge.confidence == 0.8
        assert edge.confidence_level == ConfidenceLevel.HIGH
        assert edge.method == DiscoveryMethod.PC_ALGORITHM
        assert edge.reasoning == "Strong statistical evidence"
        assert edge.statistical_evidence is None
        assert edge.llm_rationale is None
        assert edge.bidirectional_score is None
    
    def test_causal_edge_with_optional_fields(self):
        """Test CausalEdge with optional fields."""
        edge = CausalEdge(
            cause="treatment",
            effect="outcome",
            confidence=0.75,
            confidence_level=ConfidenceLevel.MEDIUM,
            method=DiscoveryMethod.LLM_GUIDED,
            reasoning="Domain knowledge supports this relationship",
            statistical_evidence={"correlation": 0.6, "p_value": 0.01},
            llm_rationale="Medical literature suggests causal relationship",
            bidirectional_score=0.3
        )
        
        assert edge.statistical_evidence["correlation"] == 0.6
        assert edge.llm_rationale == "Medical literature suggests causal relationship"
        assert edge.bidirectional_score == 0.3


class TestDiscoveryResult:
    """Test DiscoveryResult dataclass."""
    
    def test_discovery_result_creation(self):
        """Test DiscoveryResult creation."""
        edge1 = CausalEdge("X", "Y", 0.8, ConfidenceLevel.HIGH, DiscoveryMethod.PC_ALGORITHM, "Test")
        edge2 = CausalEdge("Y", "Z", 0.6, ConfidenceLevel.MEDIUM, DiscoveryMethod.PC_ALGORITHM, "Test")
        
        result = DiscoveryResult(
            discovered_edges=[edge1, edge2],
            rejected_edges=[],
            method_used=DiscoveryMethod.PC_ALGORITHM,
            confidence_summary={"high": 1, "medium": 1, "low": 0, "uncertain": 0},
            discovery_metrics={"precision": 0.8, "recall": 0.7},
            reasoning_trace=["Step 1", "Step 2"],
            time_taken=1.5,
            data_summary={"n_samples": 1000, "n_variables": 3}
        )
        
        assert len(result.discovered_edges) == 2
        assert len(result.rejected_edges) == 0
        assert result.method_used == DiscoveryMethod.PC_ALGORITHM
        assert result.confidence_summary["high"] == 1
        assert result.discovery_metrics["precision"] == 0.8
        assert result.time_taken == 1.5


class TestPCAlgorithmEngine:
    """Test PC Algorithm Engine."""
    
    def test_pc_engine_initialization(self):
        """Test PC engine initialization."""
        engine = PCAlgorithmEngine(significance_level=0.01)
        assert engine.method == DiscoveryMethod.PC_ALGORITHM
        assert engine.significance_level == 0.01
    
    @pytest.mark.asyncio
    async def test_pc_discover_structure_basic(self, sample_data, variable_descriptions):
        """Test basic structure discovery with PC algorithm."""
        engine = PCAlgorithmEngine(significance_level=0.05)
        
        result = await engine.discover_structure(
            data=sample_data,
            variables=variable_descriptions,
            domain_context="Test domain"
        )
        
        assert isinstance(result, DiscoveryResult)
        assert result.method_used == DiscoveryMethod.PC_ALGORITHM
        assert len(result.discovered_edges) >= 0
        assert result.time_taken > 0
        assert "n_samples" in result.data_summary
        assert result.data_summary["n_samples"] == len(sample_data)
    
    @pytest.mark.asyncio
    async def test_pc_discover_empty_data(self):
        """Test PC algorithm with empty data."""
        engine = PCAlgorithmEngine()
        empty_data = pd.DataFrame({"A": [], "B": []})
        variables = {"A": "Variable A", "B": "Variable B"}
        
        result = await engine.discover_structure(empty_data, variables)
        
        assert isinstance(result, DiscoveryResult)
        assert len(result.discovered_edges) == 0
    
    @pytest.mark.asyncio
    async def test_pc_confidence_levels(self, sample_data, variable_descriptions):
        """Test confidence level assignment in PC algorithm."""
        engine = PCAlgorithmEngine()
        result = await engine.discover_structure(sample_data, variable_descriptions)
        
        for edge in result.discovered_edges:
            assert edge.confidence_level in list(ConfidenceLevel)
            assert 0 <= edge.confidence <= 1
    
    def test_get_confidence_level(self):
        """Test confidence level mapping."""
        engine = PCAlgorithmEngine()
        
        assert engine._get_confidence_level(0.9) == ConfidenceLevel.HIGH
        assert engine._get_confidence_level(0.7) == ConfidenceLevel.MEDIUM
        assert engine._get_confidence_level(0.5) == ConfidenceLevel.LOW
        assert engine._get_confidence_level(0.2) == ConfidenceLevel.UNCERTAIN


class TestLLMGuidedDiscoveryEngine:
    """Test LLM-guided discovery engine."""
    
    def test_llm_engine_initialization(self, mock_llm_client):
        """Test LLM engine initialization."""
        engine = LLMGuidedDiscoveryEngine(mock_llm_client, use_statistical_evidence=True)
        assert engine.method == DiscoveryMethod.LLM_GUIDED
        assert engine.llm_client is mock_llm_client
        assert engine.use_statistical_evidence is True
    
    @pytest.mark.asyncio
    async def test_llm_discover_structure_basic(self, mock_llm_client, sample_data, variable_descriptions):
        """Test basic LLM-guided discovery."""
        # Mock LLM response with valid JSON
        mock_llm_client.generate_response = AsyncMock(return_value="""
        [
            {
                "cause": "X1",
                "effect": "X2",
                "confidence": "high",
                "reasoning": "X1 directly affects X2 based on domain knowledge"
            },
            {
                "cause": "X2", 
                "effect": "X3",
                "confidence": "medium",
                "reasoning": "X2 influences X3 through mediation"
            }
        ]
        """)
        
        engine = LLMGuidedDiscoveryEngine(mock_llm_client)
        result = await engine.discover_structure(
            data=sample_data,
            variables=variable_descriptions,
            domain_context="Healthcare treatment analysis"
        )
        
        assert isinstance(result, DiscoveryResult)
        assert result.method_used == DiscoveryMethod.LLM_GUIDED
        assert len(result.discovered_edges) >= 1
        
        # Check that LLM rationale is preserved
        for edge in result.discovered_edges:
            assert edge.llm_rationale is not None
            assert len(edge.llm_rationale) > 0
    
    @pytest.mark.asyncio
    async def test_llm_discover_with_background_knowledge(self, mock_llm_client, sample_data, variable_descriptions):
        """Test LLM discovery with background knowledge."""
        background_knowledge = [
            "X1 is known to be a strong predictor of X2",
            "X3 is the primary outcome of interest"
        ]
        
        mock_llm_client.generate_response = AsyncMock(return_value='[{"cause": "X1", "effect": "X2", "confidence": "high", "reasoning": "Test"}]')
        
        engine = LLMGuidedDiscoveryEngine(mock_llm_client)
        result = await engine.discover_structure(
            data=sample_data,
            variables=variable_descriptions,
            background_knowledge=background_knowledge
        )
        
        # Check that the prompt included background knowledge
        mock_llm_client.generate_response.assert_called_once()
        call_args = mock_llm_client.generate_response.call_args[0][0]
        assert "BACKGROUND KNOWLEDGE" in call_args
        assert background_knowledge[0] in call_args
    
    def test_create_discovery_prompt(self, mock_llm_client, variable_descriptions):
        """Test discovery prompt creation."""
        engine = LLMGuidedDiscoveryEngine(mock_llm_client)
        
        statistical_evidence = {
            "strong_correlations": [
                {"var1": "X1", "var2": "X2", "correlation": 0.7, "interpretation": "strong"}
            ]
        }
        
        prompt = engine._create_discovery_prompt(
            variables=variable_descriptions,
            domain_context="Test domain",
            background_knowledge=["Test knowledge"],
            statistical_evidence=statistical_evidence
        )
        
        assert "VARIABLES:" in prompt
        assert "DOMAIN CONTEXT" in prompt
        assert "BACKGROUND KNOWLEDGE" in prompt
        assert "STATISTICAL EVIDENCE" in prompt
        assert "X1" in prompt
        assert "strong correlation" in prompt
    
    @pytest.mark.asyncio
    async def test_llm_json_parsing_failure(self, mock_llm_client, sample_data, variable_descriptions):
        """Test handling of malformed JSON response."""
        # Mock malformed JSON response
        mock_llm_client.generate_response = AsyncMock(return_value="This is not valid JSON")
        
        engine = LLMGuidedDiscoveryEngine(mock_llm_client)
        
        with patch.object(engine, '_parse_text_response') as mock_parse:
            mock_parse.return_value = ([], [])
            
            result = await engine.discover_structure(sample_data, variable_descriptions)
            
            # Should fallback to text parsing
            mock_parse.assert_called_once()
    
    def test_parse_text_response(self, mock_llm_client):
        """Test text response parsing fallback."""
        engine = LLMGuidedDiscoveryEngine(mock_llm_client)
        
        response = "X1 causes X2 and there is strong evidence. X2 influences X3."
        reasoning_trace = []
        
        discovered, rejected = engine._parse_text_response(response, reasoning_trace)
        
        assert len(discovered) >= 1
        assert all(isinstance(edge, CausalEdge) for edge in discovered)
    
    @pytest.mark.asyncio
    async def test_statistical_evidence_generation(self, mock_llm_client, sample_data, variable_descriptions):
        """Test statistical evidence generation."""
        engine = LLMGuidedDiscoveryEngine(mock_llm_client, use_statistical_evidence=True)
        reasoning_trace = []
        
        evidence = await engine._generate_statistical_evidence(sample_data, variable_descriptions, reasoning_trace)
        
        assert "correlations" in evidence
        assert "descriptive_stats" in evidence
        assert "strong_correlations" in evidence
        assert len(reasoning_trace) > 0


class TestHybridLLMDiscoveryEngine:
    """Test hybrid LLM + statistical discovery."""
    
    def test_hybrid_engine_initialization(self, mock_llm_client):
        """Test hybrid engine initialization."""
        engine = HybridLLMDiscoveryEngine(mock_llm_client)
        assert engine.method == DiscoveryMethod.HYBRID_LLM
        assert engine.llm_client is mock_llm_client
        assert isinstance(engine.statistical_engine, PCAlgorithmEngine)
        assert isinstance(engine.llm_engine, LLMGuidedDiscoveryEngine)
    
    @pytest.mark.asyncio
    async def test_hybrid_discover_structure(self, mock_llm_client, sample_data, variable_descriptions):
        """Test hybrid discovery structure."""
        # Mock both statistical and LLM results
        mock_stat_result = DiscoveryResult(
            discovered_edges=[CausalEdge("X1", "X2", 0.7, ConfidenceLevel.MEDIUM, DiscoveryMethod.PC_ALGORITHM, "Stat")],
            rejected_edges=[],
            method_used=DiscoveryMethod.PC_ALGORITHM,
            confidence_summary={},
            discovery_metrics={},
            reasoning_trace=["Statistical analysis"],
            time_taken=1.0,
            data_summary={}
        )
        
        mock_llm_result = DiscoveryResult(
            discovered_edges=[CausalEdge("X1", "X2", 0.8, ConfidenceLevel.HIGH, DiscoveryMethod.LLM_GUIDED, "LLM")],
            rejected_edges=[],
            method_used=DiscoveryMethod.LLM_GUIDED,
            confidence_summary={},
            discovery_metrics={},
            reasoning_trace=["LLM analysis"],
            time_taken=1.0,
            data_summary={}
        )
        
        engine = HybridLLMDiscoveryEngine(mock_llm_client)
        
        with patch.object(engine.statistical_engine, 'discover_structure', return_value=mock_stat_result), \
             patch.object(engine.llm_engine, 'discover_structure', return_value=mock_llm_result):
            
            result = await engine.discover_structure(sample_data, variable_descriptions)
            
            assert isinstance(result, DiscoveryResult)
            assert result.method_used == DiscoveryMethod.HYBRID_LLM
            assert len(result.discovered_edges) >= 1
            assert "statistical_edges" in result.discovery_metrics
            assert "llm_edges" in result.discovery_metrics
            assert "agreement_rate" in result.discovery_metrics
    
    @pytest.mark.asyncio
    async def test_hybrid_results_combination(self, mock_llm_client):
        """Test combination of statistical and LLM results."""
        engine = HybridLLMDiscoveryEngine(mock_llm_client)
        
        # Results with overlapping edges
        stat_result = Mock()
        stat_result.discovered_edges = [
            CausalEdge("A", "B", 0.6, ConfidenceLevel.MEDIUM, DiscoveryMethod.PC_ALGORITHM, "Stat"),
            CausalEdge("B", "C", 0.5, ConfidenceLevel.LOW, DiscoveryMethod.PC_ALGORITHM, "Stat")
        ]
        stat_result.rejected_edges = []
        
        llm_result = Mock()
        llm_result.discovered_edges = [
            CausalEdge("A", "B", 0.8, ConfidenceLevel.HIGH, DiscoveryMethod.LLM_GUIDED, "LLM"),
            CausalEdge("C", "D", 0.7, ConfidenceLevel.MEDIUM, DiscoveryMethod.LLM_GUIDED, "LLM")
        ]
        llm_result.rejected_edges = []
        
        reasoning_trace = []
        combined_edges, combined_rejected = await engine._combine_results(stat_result, llm_result, reasoning_trace)
        
        assert len(combined_edges) >= 2  # At least overlapping + unique edges
        
        # Check that overlapping edge has higher confidence
        ab_edges = [e for e in combined_edges if e.cause == "A" and e.effect == "B"]
        assert len(ab_edges) == 1
        assert ab_edges[0].confidence > 0.7  # Should be boosted
    
    def test_calculate_agreement_rate(self, mock_llm_client):
        """Test agreement rate calculation."""
        engine = HybridLLMDiscoveryEngine(mock_llm_client)
        
        stat_result = Mock()
        stat_result.discovered_edges = [
            Mock(cause="A", effect="B"),
            Mock(cause="B", effect="C")
        ]
        
        llm_result = Mock()
        llm_result.discovered_edges = [
            Mock(cause="A", effect="B"),  # Agreement
            Mock(cause="C", effect="D")   # Disagreement
        ]
        
        agreement_rate = engine._calculate_agreement_rate(stat_result, llm_result)
        
        # Should be 1/(2+1) = 0.33 (1 common edge, 3 total unique edges)
        assert 0.2 <= agreement_rate <= 0.4


class TestAdvancedCausalDiscovery:
    """Test advanced causal discovery interface."""
    
    def test_advanced_discovery_initialization(self, mock_llm_client):
        """Test advanced discovery initialization."""
        discovery = AdvancedCausalDiscovery(mock_llm_client)
        
        assert DiscoveryMethod.PC_ALGORITHM in discovery.engines
        assert DiscoveryMethod.LLM_GUIDED in discovery.engines
        assert DiscoveryMethod.HYBRID_LLM in discovery.engines
        assert discovery.llm_client is mock_llm_client
    
    def test_advanced_discovery_no_llm(self):
        """Test advanced discovery without LLM client."""
        discovery = AdvancedCausalDiscovery(llm_client=None)
        
        # Only PC algorithm should be available
        assert DiscoveryMethod.PC_ALGORITHM in discovery.engines
        assert discovery.engines[DiscoveryMethod.LLM_GUIDED] is None
        assert discovery.engines[DiscoveryMethod.HYBRID_LLM] is None
    
    def test_get_available_methods(self, mock_llm_client):
        """Test getting available discovery methods."""
        discovery = AdvancedCausalDiscovery(mock_llm_client)
        methods = discovery.get_available_methods()
        
        assert DiscoveryMethod.PC_ALGORITHM in methods
        assert DiscoveryMethod.LLM_GUIDED in methods
        assert DiscoveryMethod.HYBRID_LLM in methods
    
    @pytest.mark.asyncio
    async def test_discover_method_selection(self, mock_llm_client, sample_data, variable_descriptions):
        """Test discovery with method selection."""
        discovery = AdvancedCausalDiscovery(mock_llm_client)
        
        # Mock PC algorithm result
        with patch.object(discovery.engines[DiscoveryMethod.PC_ALGORITHM], 'discover_structure') as mock_discover:
            mock_result = DiscoveryResult([], [], DiscoveryMethod.PC_ALGORITHM, {}, {}, [], 1.0, {})
            mock_discover.return_value = mock_result
            
            result = await discovery.discover(
                data=sample_data,
                variables=variable_descriptions,
                method=DiscoveryMethod.PC_ALGORITHM
            )
            
            assert result is mock_result
            mock_discover.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_discover_invalid_method(self, mock_llm_client, variable_descriptions):
        """Test discovery with invalid method."""
        discovery = AdvancedCausalDiscovery(mock_llm_client)
        
        # Try LLM method when no LLM client is actually available
        discovery.engines[DiscoveryMethod.LLM_GUIDED] = None
        
        with pytest.raises(ValueError, match="requires LLM client"):
            await discovery.discover(
                variables=variable_descriptions,
                method=DiscoveryMethod.LLM_GUIDED
            )
    
    @pytest.mark.asyncio
    async def test_compare_methods(self, mock_llm_client, sample_data, variable_descriptions):
        """Test comparing multiple discovery methods."""
        discovery = AdvancedCausalDiscovery(mock_llm_client)
        
        # Mock results for different methods
        pc_result = DiscoveryResult([], [], DiscoveryMethod.PC_ALGORITHM, {}, {}, [], 1.0, {})
        
        with patch.object(discovery.engines[DiscoveryMethod.PC_ALGORITHM], 'discover_structure', return_value=pc_result):
            results = await discovery.compare_methods(
                data=sample_data,
                variables=variable_descriptions,
                methods=[DiscoveryMethod.PC_ALGORITHM]
            )
            
            assert DiscoveryMethod.PC_ALGORITHM in results
            assert results[DiscoveryMethod.PC_ALGORITHM] is pc_result
    
    @pytest.mark.asyncio
    async def test_discover_missing_variables(self, mock_llm_client, sample_data):
        """Test discovery with missing variables parameter."""
        discovery = AdvancedCausalDiscovery(mock_llm_client)
        
        with pytest.raises(ValueError, match="Variables dictionary is required"):
            await discovery.discover(data=sample_data)


class TestFactoryFunctions:
    """Test factory and convenience functions."""
    
    def test_create_discovery_engine_pc(self):
        """Test creating PC algorithm engine."""
        engine = create_discovery_engine(method=DiscoveryMethod.PC_ALGORITHM)
        assert isinstance(engine, PCAlgorithmEngine)
        assert engine.method == DiscoveryMethod.PC_ALGORITHM
    
    def test_create_discovery_engine_llm(self, mock_llm_client):
        """Test creating LLM-guided engine."""
        engine = create_discovery_engine(mock_llm_client, DiscoveryMethod.LLM_GUIDED)
        assert isinstance(engine, LLMGuidedDiscoveryEngine)
        assert engine.llm_client is mock_llm_client
    
    def test_create_discovery_engine_hybrid(self, mock_llm_client):
        """Test creating hybrid engine."""
        engine = create_discovery_engine(mock_llm_client, DiscoveryMethod.HYBRID_LLM)
        assert isinstance(engine, HybridLLMDiscoveryEngine)
        assert engine.llm_client is mock_llm_client
    
    def test_create_discovery_engine_llm_no_client(self):
        """Test error when creating LLM engine without client."""
        with pytest.raises(ValueError, match="LLM client required"):
            create_discovery_engine(method=DiscoveryMethod.LLM_GUIDED)
    
    def test_create_discovery_engine_unsupported(self):
        """Test error with unsupported method."""
        with pytest.raises(ValueError, match="Unsupported method"):
            create_discovery_engine(method="unsupported_method")
    
    @pytest.mark.asyncio
    async def test_discover_causal_structure_convenience(self, mock_llm_client, variable_descriptions):
        """Test convenience function for causal structure discovery."""
        with patch('causallm.core.causal_discovery.AdvancedCausalDiscovery') as mock_discovery_class:
            mock_discovery = Mock()
            mock_result = DiscoveryResult([], [], DiscoveryMethod.HYBRID_LLM, {}, {}, [], 1.0, {})
            mock_discovery.discover.return_value = mock_result
            mock_discovery_class.return_value = mock_discovery
            
            result = await discover_causal_structure(
                variables=variable_descriptions,
                llm_client=mock_llm_client,
                domain_context="test"
            )
            
            assert result is mock_result
            mock_discovery.discover.assert_called_once()


class TestDiscoveryIntegration:
    """Test integration scenarios for causal discovery."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_discovery_pipeline(self, sample_data, variable_descriptions):
        """Test complete discovery pipeline without LLM."""
        discovery = AdvancedCausalDiscovery(llm_client=None)
        
        result = await discovery.discover(
            data=sample_data,
            variables=variable_descriptions,
            method=DiscoveryMethod.PC_ALGORITHM,
            domain_context="Healthcare research"
        )
        
        assert isinstance(result, DiscoveryResult)
        assert result.method_used == DiscoveryMethod.PC_ALGORITHM
        assert len(result.reasoning_trace) > 0
        assert result.time_taken > 0
        assert "n_samples" in result.data_summary
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_discovery_performance_large_data(self, large_sample_data):
        """Test discovery performance on larger dataset."""
        variables = {col: f"Variable {col}" for col in large_sample_data.columns}
        discovery = AdvancedCausalDiscovery(llm_client=None)
        
        import time
        start_time = time.time()
        
        result = await discovery.discover(
            data=large_sample_data,
            variables=variables,
            method=DiscoveryMethod.PC_ALGORITHM
        )
        
        duration = time.time() - start_time
        
        # Should complete in reasonable time (less than 30 seconds)
        assert duration < 30
        assert isinstance(result, DiscoveryResult)
        assert result.data_summary["n_samples"] == len(large_sample_data)
    
    @pytest.mark.asyncio
    async def test_discovery_reproducibility(self, sample_data, variable_descriptions):
        """Test discovery reproducibility."""
        discovery = AdvancedCausalDiscovery(llm_client=None)
        
        # Run discovery twice with same parameters
        result1 = await discovery.discover(
            data=sample_data,
            variables=variable_descriptions,
            method=DiscoveryMethod.PC_ALGORITHM
        )
        
        result2 = await discovery.discover(
            data=sample_data,
            variables=variable_descriptions,
            method=DiscoveryMethod.PC_ALGORITHM
        )
        
        # Results should be similar (allowing for some randomness)
        assert len(result1.discovered_edges) == len(result2.discovered_edges)
        assert result1.method_used == result2.method_used