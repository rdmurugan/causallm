"""
Tests for Tier 1 LLM enhancements including intelligent prompting,
multi-agent analysis, and dynamic RAG capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalllm.llm_prompting import (
    CausalPromptEngine, 
    CausalExample, 
    PromptTemplate
)
from causalllm.llm_agents import (
    MultiAgentCausalAnalyzer,
    CollaborativeAnalysis,
    AgentRole
)
from causalllm.causal_rag import (
    DynamicCausalRAG,
    CausalKnowledgeBase,
    CausalDocument,
    CausalRAGResponse,
    create_rag_system
)
from causalllm.core import CausalLLMCore


class TestCausalPromptEngine:
    """Test suite for causal prompt engine system."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = Mock()
        client.generate = AsyncMock(return_value="Test response from LLM")
        return client
    
    @pytest.fixture
    def prompting_system(self, mock_llm_client):
        """Create a CausalPromptEngine instance."""
        return CausalPromptEngine()
    
    def test_initialization(self, prompting_system):
        """Test proper initialization of the prompting system."""
        assert hasattr(prompting_system, 'examples_db')
        assert hasattr(prompting_system, 'templates')
        assert len(prompting_system.examples_db) > 0
        assert len(prompting_system.templates) > 0
    
    def test_causal_example_creation(self):
        """Test CausalExample dataclass creation."""
        example = CausalExample(
            context="Test context",
            factual="Test factual",
            intervention="Test intervention",
            analysis="Test analysis",
            reasoning_steps=["Step 1", "Step 2"],
            domain="test",
            quality_score=0.8
        )
        
        assert example.context == "Test context"
        assert example.quality_score == 0.8
        assert len(example.reasoning_steps) == 2
    
    def test_get_few_shot_examples(self, prompting_system):
        """Test getting few-shot examples for a domain."""
        examples = prompting_system.get_few_shot_examples(
            domain="healthcare",
            task_type="counterfactual",
            max_examples=2
        )
        
        assert isinstance(examples, list)
        assert len(examples) <= 2
    
    def test_generate_chain_of_thought_prompt(self, prompting_system):
        """Test chain-of-thought prompt generation."""
        result = prompting_system.generate_chain_of_thought_prompt(
            task_type="counterfactual",
            domain="healthcare",
            context="Test medical context",
            factual="Patient received treatment A",
            intervention="Patient received treatment B instead"
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_assess_prompt_quality(self, prompting_system):
        """Test prompt quality assessment."""
        prompt = "This is a test prompt for counterfactual analysis of treatment effects."
        examples = [
            CausalExample(
                context="ctx", factual="fact", intervention="int", analysis="anal",
                reasoning_steps=[], domain="test", quality_score=0.9
            )
        ]
        
        quality = prompting_system._assess_prompt_quality(prompt, examples, "healthcare")
        
        assert 0 <= quality <= 1
        assert isinstance(quality, float)


class TestMultiAgentCausalAnalyzer:
    """Test suite for multi-agent analysis system."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = Mock()
        client.generate = AsyncMock(return_value="Agent analysis response")
        return client
    
    @pytest.fixture
    def analyzer(self, mock_llm_client):
        """Create a MultiAgentCausalAnalyzer instance."""
        return MultiAgentCausalAnalyzer(mock_llm_client)
    
    def test_initialization(self, analyzer):
        """Test proper initialization of multi-agent analyzer."""
        assert analyzer.llm_client is not None
        assert hasattr(analyzer, 'agent_prompts')
        assert len(analyzer.agent_prompts) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_counterfactual(self, analyzer):
        """Test counterfactual analysis with multiple agents."""
        result = await analyzer.analyze_counterfactual(
            context="Test context for analysis",
            factual="Current scenario",
            intervention="Alternative scenario"
        )
        
        assert isinstance(result, CollaborativeAnalysis)
        assert len(result.domain_expert_analysis) > 0
        assert len(result.statistical_analysis) > 0
        assert len(result.skeptic_analysis) > 0
        assert len(result.synthesized_conclusion) > 0
        assert 0 <= result.confidence_score <= 1
    
    @pytest.mark.asyncio
    async def test_analyze_treatment_effect(self, analyzer):
        """Test treatment effect analysis."""
        result = await analyzer.analyze_treatment_effect(
            context="Treatment study context",
            treatment="New medication",
            outcome="Patient recovery rate"
        )
        
        assert isinstance(result, CollaborativeAnalysis)
        assert result.confidence_score > 0
        assert len(result.key_assumptions) > 0
        assert len(result.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_get_agent_analysis(self, analyzer):
        """Test individual agent analysis."""
        prompt = "Analyze this scenario from domain expert perspective"
        analysis = await analyzer._get_agent_analysis(AgentRole.DOMAIN_EXPERT, prompt)
        
        assert len(analysis) > 0
        assert isinstance(analysis, str)
    
    def test_synthesize_analyses(self, analyzer):
        """Test synthesis of multiple agent analyses."""
        analyses = {
            AgentRole.DOMAIN_EXPERT: "Domain expert analysis",
            AgentRole.STATISTICIAN: "Statistical analysis", 
            AgentRole.SKEPTIC: "Critical analysis"
        }
        
        synthesis = analyzer._synthesize_analyses(analyses)
        
        assert len(synthesis) > 0
        assert isinstance(synthesis, str)


class TestCausalRAG:
    """Test suite for dynamic causal RAG system."""
    
    @pytest.fixture
    def knowledge_base(self):
        """Create a test knowledge base."""
        return CausalKnowledgeBase()
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = Mock()
        client.generate_response = AsyncMock(return_value="RAG-enhanced response")
        return client
    
    @pytest.fixture
    def rag_system(self, knowledge_base, mock_llm_client):
        """Create a DynamicCausalRAG instance."""
        return DynamicCausalRAG(knowledge_base, mock_llm_client)
    
    def test_knowledge_base_initialization(self, knowledge_base):
        """Test knowledge base initialization with default documents."""
        assert len(knowledge_base.documents) > 0
        
        # Check that we have some expected documents
        doc_titles = [doc.title for doc in knowledge_base.documents.values()]
        assert any("Confounding" in title for title in doc_titles)
        assert any("Counterfactual" in title for title in doc_titles)
    
    def test_document_search(self, knowledge_base):
        """Test document search functionality."""
        results = knowledge_base.search_documents(
            query="What is confounding bias?",
            domain="general",
            top_k=3
        )
        
        assert len(results) <= 3
        assert len(results) > 0
        
        # Check that results are properly scored
        for result in results:
            assert 0 <= result.combined_score <= 1
            assert hasattr(result, 'relevance_score')
            assert hasattr(result, 'document')
    
    @pytest.mark.asyncio
    async def test_enhance_query(self, rag_system):
        """Test query enhancement with RAG."""
        response = await rag_system.enhance_query(
            query="How does confounding affect treatment effect estimation?",
            context="Medical research study",
            domain="healthcare"
        )
        
        assert isinstance(response, CausalRAGResponse)
        assert len(response.enhanced_context) > 0
        assert len(response.reasoning_steps) > 0
        assert 0 <= response.confidence_score <= 1
        assert len(response.retrieved_documents) > 0
    
    def test_causal_document_creation(self):
        """Test CausalDocument creation and embedding."""
        doc = CausalDocument(
            id="test_doc",
            title="Test Document",
            content="This is test content about causal inference",
            doc_type="methodology",
            domain="general",
            causal_concepts=["causality", "inference"]
        )
        
        assert doc.id == "test_doc"
        assert doc.title == "Test Document"
        assert len(doc.causal_concepts) == 2
        assert doc.quality_score == 1.0  # default value
    
    def test_create_rag_system_convenience_function(self):
        """Test the convenience function for creating RAG systems."""
        rag_system = create_rag_system()
        
        assert isinstance(rag_system, DynamicCausalRAG)
        assert rag_system.knowledge_base is not None
        assert len(rag_system.knowledge_base.documents) > 0


class TestTier1Integration:
    """Integration tests for Tier 1 enhanced core functionality."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = Mock()
        client.generate = AsyncMock(return_value="Integrated analysis response")
        client.generate_response = AsyncMock(return_value="Enhanced response")
        return client
    
    @pytest.fixture
    def enhanced_core(self, mock_llm_client):
        """Create an enhanced CausalLLMCore with Tier 1 features."""
        context = "Test context for causal analysis"
        variables = {
            "treatment": "Test treatment variable",
            "outcome": "Test outcome variable"
        }
        dag_edges = [("treatment", "outcome")]
        
        return CausalLLMCore(
            context=context,
            variables=variables,
            dag_edges=dag_edges,
            llm_client=mock_llm_client
        )
    
    def test_tier1_initialization(self, enhanced_core):
        """Test that Tier 1 enhancements are properly initialized."""
        assert hasattr(enhanced_core, 'intelligent_prompting')
        assert hasattr(enhanced_core, 'multi_agent_analyzer')
        assert hasattr(enhanced_core, 'rag_system')
        
        # Check types
        assert isinstance(enhanced_core.intelligent_prompting, IntelligentCausalPrompting)
        assert isinstance(enhanced_core.multi_agent_analyzer, MultiAgentCausalAnalyzer)
        assert isinstance(enhanced_core.rag_system, DynamicCausalRAG)
    
    @pytest.mark.asyncio
    async def test_enhanced_counterfactual_analysis(self, enhanced_core):
        """Test the integrated enhanced counterfactual analysis."""
        result = await enhanced_core.enhanced_counterfactual_analysis(
            factual="Patient received treatment A",
            intervention="Patient received treatment B",
            domain="healthcare"
        )
        
        assert isinstance(result, dict)
        assert "rag_analysis" in result
        assert "intelligent_prompting" in result
        assert "multi_agent_analysis" in result
        assert "overall_confidence" in result
        
        # Check confidence score is reasonable
        assert 0 <= result["overall_confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_enhanced_treatment_effect_analysis(self, enhanced_core):
        """Test the integrated enhanced treatment effect analysis."""
        result = await enhanced_core.enhanced_treatment_effect_analysis(
            treatment="New medication protocol",
            outcome="Patient recovery time",
            domain="healthcare"
        )
        
        assert isinstance(result, dict)
        assert "treatment" in result
        assert "outcome" in result
        assert "rag_enhancement" in result
        assert "collaborative_analysis" in result
        assert "overall_assessment" in result


class TestTier1ErrorHandling:
    """Test error handling and edge cases in Tier 1 features."""
    
    @pytest.fixture
    def failing_llm_client(self):
        """Create a mock LLM client that fails."""
        client = Mock()
        client.generate = AsyncMock(side_effect=Exception("LLM API error"))
        client.generate_response = AsyncMock(side_effect=Exception("LLM API error"))
        return client
    
    @pytest.mark.asyncio
    async def test_prompting_with_failing_llm(self, failing_llm_client):
        """Test intelligent prompting handles LLM failures gracefully."""
        prompting = IntelligentCausalPrompting(failing_llm_client)
        
        with pytest.raises(Exception):
            await prompting.generate_enhanced_counterfactual_prompt(
                context="test", factual="test", intervention="test"
            )
    
    @pytest.mark.asyncio
    async def test_multi_agent_with_failing_llm(self, failing_llm_client):
        """Test multi-agent analysis handles LLM failures gracefully."""
        analyzer = MultiAgentCausalAnalyzer(failing_llm_client)
        
        with pytest.raises(Exception):
            await analyzer.analyze_counterfactual(
                context="test", factual="test", intervention="test"
            )
    
    def test_rag_with_empty_knowledge_base(self):
        """Test RAG system with empty knowledge base."""
        empty_kb = CausalKnowledgeBase()
        empty_kb.documents = {}  # Clear all documents
        
        rag_system = DynamicCausalRAG(empty_kb)
        
        # Should handle empty knowledge base gracefully
        results = empty_kb.search_documents("test query", top_k=5)
        assert len(results) == 0
    
    def test_invalid_domain_handling(self):
        """Test handling of invalid or unknown domains."""
        kb = CausalKnowledgeBase()
        
        results = kb.search_documents(
            query="test query",
            domain="nonexistent_domain",
            top_k=5
        )
        
        # Should still return results (general domain documents)
        assert isinstance(results, list)


# Performance and load tests
class TestTier1Performance:
    """Performance tests for Tier 1 features."""
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis(self):
        """Test concurrent execution of multiple analyses."""
        mock_client = Mock()
        mock_client.generate = AsyncMock(return_value="Response")
        mock_client.generate_response = AsyncMock(return_value="Enhanced response")
        
        core = CausalLLMCore(
            context="Test context",
            variables={"x": "variable x", "y": "variable y"},
            dag_edges=[("x", "y")],
            llm_client=mock_client
        )
        
        # Run multiple analyses concurrently
        tasks = [
            core.enhanced_counterfactual_analysis("fact1", "int1", "healthcare"),
            core.enhanced_counterfactual_analysis("fact2", "int2", "marketing"),
            core.enhanced_treatment_effect_analysis("treatment", "outcome", "healthcare")
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete without exceptions
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, dict)
    
    def test_large_knowledge_base_search(self):
        """Test search performance with larger knowledge base."""
        kb = CausalKnowledgeBase()
        
        # Add many more documents
        for i in range(100):
            doc = CausalDocument(
                id=f"perf_test_doc_{i}",
                title=f"Performance Test Document {i}",
                content=f"This is performance test content {i} about causal inference",
                doc_type="methodology",
                domain="general",
                causal_concepts=["causality", f"concept_{i % 10}"],
                quality_score=0.8
            )
            kb.add_document(doc)
        
        # Search should still be fast
        import time
        start_time = time.time()
        
        results = kb.search_documents("causal inference methodology", top_k=10)
        
        search_time = time.time() - start_time
        
        assert len(results) == 10
        assert search_time < 1.0  # Should complete within 1 second


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])