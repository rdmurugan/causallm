"""
Dynamic RAG system for causal knowledge retrieval and integration.

This module provides retrieval-augmented generation capabilities specifically designed
for causal inference tasks, incorporating domain knowledge, research papers, and case studies
to enhance reasoning quality.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import json
import hashlib
import numpy as np
from pathlib import Path

from causalllm.logging import get_logger


@dataclass
class CausalDocument:
    """Represents a causal knowledge document in the RAG system."""
    
    id: str
    title: str
    content: str
    doc_type: str  # 'research_paper', 'case_study', 'methodology', 'domain_knowledge'
    domain: str  # 'healthcare', 'economics', 'marketing', 'social_science', etc.
    causal_concepts: List[str]  # ['confounding', 'instrumental_variables', 'counterfactuals']
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    quality_score: float = 1.0
    citation: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result from causal knowledge retrieval."""
    
    document: CausalDocument
    relevance_score: float
    concept_match_score: float
    domain_match_score: float
    combined_score: float
    explanation: str


@dataclass
class CausalRAGResponse:
    """Enhanced response from RAG-augmented causal reasoning."""
    
    query: str
    enhanced_context: str
    retrieved_documents: List[RetrievalResult]
    reasoning_steps: List[str]
    confidence_score: float
    knowledge_gaps: List[str]
    recommendations: List[str]
    citations: List[str]


class CausalEmbeddingEngine:
    """Handles embedding generation and similarity computation for causal documents."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.logger = get_logger("causalllm.causal_rag.embedding")
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"Initialized embedding model: {self.model_name}")
        except ImportError:
            self.logger.warning("sentence-transformers not available, using fallback embeddings")
            self.model = None
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            self.model = None
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        if self.model is None:
            # Fallback to simple hash-based embedding
            return self._fallback_embedding(text)
        
        try:
            embedding = self.model.encode(text)
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return self._fallback_embedding(text)
    
    def _fallback_embedding(self, text: str) -> np.ndarray:
        """Generate a simple hash-based embedding as fallback."""
        # Create a deterministic embedding based on text hash
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        # Convert to normalized vector
        np.random.seed(hash_int % (2**32))
        embedding = np.random.normal(0, 1, 384)  # Standard embedding size
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        try:
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            return 0.0


class CausalKnowledgeBase:
    """Manages the causal knowledge repository."""
    
    def __init__(self, data_path: Optional[str] = None):
        self.logger = get_logger("causalllm.causal_rag.knowledge_base")
        self.documents: Dict[str, CausalDocument] = {}
        self.embedding_engine = CausalEmbeddingEngine()
        self.data_path = Path(data_path) if data_path else Path("causal_knowledge_base")
        self._ensure_data_directory()
        self._load_default_knowledge()
    
    def _ensure_data_directory(self):
        """Ensure the data directory exists."""
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Knowledge base directory: {self.data_path}")
    
    def _load_default_knowledge(self):
        """Load default causal knowledge into the system."""
        default_docs = [
            {
                "id": "confounding_basics",
                "title": "Understanding Confounding Variables",
                "content": """
Confounding occurs when a third variable influences both the treatment and outcome variables, 
creating a spurious association. To identify confounders, look for variables that:
1. Are associated with the treatment assignment
2. Affect the outcome independent of treatment
3. Are not on the causal path between treatment and outcome

Common identification strategies:
- Randomized controlled trials
- Instrumental variables
- Regression discontinuity
- Difference-in-differences
- Matching methods
""",
                "doc_type": "methodology",
                "domain": "general",
                "causal_concepts": ["confounding", "bias", "identification"],
                "quality_score": 0.95,
            },
            {
                "id": "counterfactual_framework",
                "title": "Counterfactual Framework for Causal Inference",
                "content": """
The counterfactual framework, also known as the potential outcomes framework, 
defines causation in terms of potential outcomes. For each unit i:
- Y₁ᵢ: potential outcome if treated
- Y₀ᵢ: potential outcome if not treated
- Treatment effect: τᵢ = Y₁ᵢ - Y₀ᵢ

The fundamental problem of causal inference is that we can only observe one 
potential outcome for each unit. Key assumptions:
1. SUTVA (Stable Unit Treatment Value Assumption)
2. Ignorability/Unconfoundedness
3. Positivity/Common Support
""",
                "doc_type": "methodology",
                "domain": "general",
                "causal_concepts": ["counterfactuals", "potential_outcomes", "treatment_effects"],
                "quality_score": 0.98,
            },
            {
                "id": "instrumental_variables",
                "title": "Instrumental Variables Method",
                "content": """
Instrumental Variables (IV) method is used when there are unobserved confounders.
An instrument Z must satisfy:
1. Relevance: Z is correlated with treatment X
2. Exclusion: Z affects outcome Y only through X
3. Exogeneity: Z is uncorrelated with unobserved confounders

Common instruments:
- Natural experiments (weather, policy changes)
- Lottery-based assignments
- Geographic variation
- Institutional rules

Two-Stage Least Squares (2SLS) is the standard estimation method.
""",
                "doc_type": "methodology",
                "domain": "general",
                "causal_concepts": ["instrumental_variables", "endogeneity", "2sls"],
                "quality_score": 0.92,
            },
            {
                "id": "healthcare_rct",
                "title": "Randomized Controlled Trials in Healthcare",
                "content": """
RCTs are the gold standard for causal inference in healthcare. Key considerations:

Design elements:
- Randomization methods (simple, block, stratified)
- Blinding (single, double, triple)
- Control groups (placebo, active, wait-list)
- Outcome measurement timing

Common challenges:
- Recruitment and retention
- Ethical considerations
- Generalizability
- Compliance and adherence
- Missing data and dropouts

Analysis approaches:
- Intention-to-treat (ITT)
- Per-protocol analysis
- Instrumental variable analysis for non-compliance
""",
                "doc_type": "domain_knowledge",
                "domain": "healthcare",
                "causal_concepts": ["randomized_trials", "clinical_research", "medical_statistics"],
                "quality_score": 0.94,
            },
            {
                "id": "marketing_attribution",
                "title": "Marketing Attribution and Lift Testing",
                "content": """
Marketing attribution seeks to identify the causal impact of different touchpoints 
on customer conversion. Challenges include:

Attribution models:
- First-touch attribution
- Last-touch attribution
- Time-decay models
- Data-driven attribution

Causal methods:
- Geo-holdout experiments
- Randomized audience splits
- Synthetic control methods
- Incrementality testing

Key metrics:
- Incremental lift
- Return on ad spend (ROAS)
- Customer lifetime value impact
- Cross-channel effects

Biases to avoid:
- Selection bias in campaign targeting
- Seasonality confounding
- Spillover effects between channels
""",
                "doc_type": "domain_knowledge",
                "domain": "marketing",
                "causal_concepts": ["attribution", "incrementality", "marketing_mix", "lift_testing"],
                "quality_score": 0.89,
            }
        ]
        
        for doc_data in default_docs:
            doc = CausalDocument(**doc_data)
            doc.embedding = self.embedding_engine.embed_text(doc.content)
            self.add_document(doc)
        
        self.logger.info(f"Loaded {len(default_docs)} default knowledge documents")
    
    def add_document(self, document: CausalDocument):
        """Add a document to the knowledge base."""
        if document.embedding is None:
            document.embedding = self.embedding_engine.embed_text(document.content)
        
        self.documents[document.id] = document
        self.logger.debug(f"Added document: {document.title}")
    
    def search_documents(self, query: str, domain: Optional[str] = None, 
                        concepts: Optional[List[str]] = None, 
                        top_k: int = 5) -> List[RetrievalResult]:
        """Search for relevant documents based on query and filters."""
        if not self.documents:
            self.logger.warning("No documents in knowledge base")
            return []
        
        query_embedding = self.embedding_engine.embed_text(query)
        results = []
        
        for doc in self.documents.values():
            # Skip if domain filter doesn't match
            if domain and doc.domain != domain and doc.domain != "general":
                continue
            
            # Compute relevance score (semantic similarity)
            relevance_score = self.embedding_engine.compute_similarity(
                query_embedding, doc.embedding
            )
            
            # Compute concept match score
            concept_match_score = 0.0
            if concepts:
                matched_concepts = set(concepts) & set(doc.causal_concepts)
                concept_match_score = len(matched_concepts) / len(concepts) if concepts else 0.0
            
            # Compute domain match score
            domain_match_score = 1.0 if not domain else (
                1.0 if doc.domain == domain else 0.5 if doc.domain == "general" else 0.0
            )
            
            # Combined score with weights
            combined_score = (
                0.4 * relevance_score +
                0.3 * concept_match_score +
                0.2 * domain_match_score +
                0.1 * doc.quality_score
            )
            
            explanation = f"Relevance: {relevance_score:.2f}, Concepts: {concept_match_score:.2f}, Domain: {domain_match_score:.2f}"
            
            results.append(RetrievalResult(
                document=doc,
                relevance_score=relevance_score,
                concept_match_score=concept_match_score,
                domain_match_score=domain_match_score,
                combined_score=combined_score,
                explanation=explanation
            ))
        
        # Sort by combined score and return top k
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results[:top_k]


class DynamicCausalRAG:
    """Main RAG system for causal inference enhancement."""
    
    def __init__(self, knowledge_base: Optional[CausalKnowledgeBase] = None,
                 llm_client=None):
        self.logger = get_logger("causalllm.causal_rag")
        self.knowledge_base = knowledge_base or CausalKnowledgeBase()
        self.llm_client = llm_client
        self.logger.info("Initialized Dynamic Causal RAG system")
    
    def set_llm_client(self, llm_client):
        """Set the LLM client for enhanced reasoning."""
        self.llm_client = llm_client
        self.logger.debug("LLM client updated")
    
    async def enhance_query(self, query: str, context: str = "", 
                           domain: Optional[str] = None,
                           causal_concepts: Optional[List[str]] = None) -> CausalRAGResponse:
        """Enhance a causal reasoning query with retrieved knowledge."""
        self.logger.debug(f"Enhancing query in domain: {domain}")
        
        # Search for relevant documents
        retrieved_docs = self.knowledge_base.search_documents(
            query=query,
            domain=domain,
            concepts=causal_concepts,
            top_k=3
        )
        
        if not retrieved_docs:
            self.logger.warning("No relevant documents found")
            return CausalRAGResponse(
                query=query,
                enhanced_context=context,
                retrieved_documents=[],
                reasoning_steps=["No relevant knowledge found"],
                confidence_score=0.3,
                knowledge_gaps=["Insufficient domain knowledge"],
                recommendations=["Add more domain-specific knowledge to the system"],
                citations=[]
            )
        
        # Build enhanced context
        enhanced_context = self._build_enhanced_context(context, retrieved_docs)
        
        # Generate reasoning steps
        reasoning_steps = await self._generate_reasoning_steps(
            query, enhanced_context, retrieved_docs
        )
        
        # Assess confidence and identify gaps
        confidence_score = self._assess_confidence(retrieved_docs)
        knowledge_gaps = self._identify_knowledge_gaps(query, retrieved_docs)
        recommendations = self._generate_recommendations(query, retrieved_docs, knowledge_gaps)
        
        # Extract citations
        citations = [doc.document.citation for doc in retrieved_docs if doc.document.citation]
        
        return CausalRAGResponse(
            query=query,
            enhanced_context=enhanced_context,
            retrieved_documents=retrieved_docs,
            reasoning_steps=reasoning_steps,
            confidence_score=confidence_score,
            knowledge_gaps=knowledge_gaps,
            recommendations=recommendations,
            citations=citations
        )
    
    def _build_enhanced_context(self, original_context: str, 
                              retrieved_docs: List[RetrievalResult]) -> str:
        """Build enhanced context by integrating retrieved knowledge."""
        context_parts = []
        
        if original_context.strip():
            context_parts.append(f"Original Context:\n{original_context}")
        
        context_parts.append("\nRelevant Causal Knowledge:")
        
        for i, result in enumerate(retrieved_docs, 1):
            doc = result.document
            context_parts.append(f"\n{i}. {doc.title} (Relevance: {result.relevance_score:.2f})")
            context_parts.append(f"   Domain: {doc.domain}, Type: {doc.doc_type}")
            context_parts.append(f"   Concepts: {', '.join(doc.causal_concepts)}")
            context_parts.append(f"   Content: {doc.content[:300]}...")
            if doc.citation:
                context_parts.append(f"   Citation: {doc.citation}")
        
        return "\n".join(context_parts)
    
    async def _generate_reasoning_steps(self, query: str, enhanced_context: str,
                                      retrieved_docs: List[RetrievalResult]) -> List[str]:
        """Generate structured reasoning steps."""
        steps = [
            f"Query Analysis: {query}",
            f"Retrieved {len(retrieved_docs)} relevant knowledge documents",
        ]
        
        # Add concept-specific steps
        all_concepts = set()
        for result in retrieved_docs:
            all_concepts.update(result.document.causal_concepts)
        
        if all_concepts:
            steps.append(f"Key concepts identified: {', '.join(sorted(all_concepts))}")
        
        # Add domain-specific considerations
        domains = {result.document.domain for result in retrieved_docs}
        if domains and "general" not in domains:
            steps.append(f"Domain-specific considerations from: {', '.join(domains)}")
        
        # If LLM client is available, generate more sophisticated reasoning
        if self.llm_client:
            try:
                llm_reasoning = await self._generate_llm_reasoning(query, enhanced_context)
                steps.extend(llm_reasoning)
            except Exception as e:
                self.logger.warning(f"Failed to generate LLM reasoning: {e}")
                steps.append("Advanced reasoning unavailable due to LLM client error")
        
        return steps
    
    async def _generate_llm_reasoning(self, query: str, enhanced_context: str) -> List[str]:
        """Generate LLM-enhanced reasoning steps."""
        prompt = f"""
Based on the following causal inference query and retrieved knowledge, 
provide 3-5 structured reasoning steps:

Query: {query}

Knowledge Context:
{enhanced_context}

Generate clear, logical reasoning steps that incorporate the retrieved knowledge.
Format as a numbered list.
        """
        
        try:
            if hasattr(self.llm_client, 'generate_response'):
                response = await self.llm_client.generate_response(prompt)
            else:
                response = self.llm_client.generate(prompt)
            
            # Parse response into steps
            lines = response.strip().split('\n')
            steps = []
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering/bullets
                    step = line.lstrip('0123456789.-• ').strip()
                    if step:
                        steps.append(step)
            
            return steps if steps else ["LLM reasoning generated but could not be parsed"]
            
        except Exception as e:
            self.logger.error(f"LLM reasoning generation failed: {e}")
            return ["LLM reasoning unavailable"]
    
    def _assess_confidence(self, retrieved_docs: List[RetrievalResult]) -> float:
        """Assess confidence based on retrieval quality."""
        if not retrieved_docs:
            return 0.1
        
        avg_relevance = sum(doc.relevance_score for doc in retrieved_docs) / len(retrieved_docs)
        avg_quality = sum(doc.document.quality_score for doc in retrieved_docs) / len(retrieved_docs)
        coverage_bonus = min(len(retrieved_docs) / 3, 1.0) * 0.1
        
        confidence = 0.4 * avg_relevance + 0.4 * avg_quality + 0.2 + coverage_bonus
        return min(confidence, 1.0)
    
    def _identify_knowledge_gaps(self, query: str, 
                               retrieved_docs: List[RetrievalResult]) -> List[str]:
        """Identify potential knowledge gaps."""
        gaps = []
        
        if not retrieved_docs:
            gaps.append("No relevant documents found - knowledge base may be incomplete")
        
        # Check if we have domain-specific knowledge
        has_domain_knowledge = any(
            doc.document.domain != "general" for doc in retrieved_docs
        )
        if not has_domain_knowledge:
            gaps.append("Lack of domain-specific knowledge - general principles only")
        
        # Check relevance scores
        if retrieved_docs:
            max_relevance = max(doc.relevance_score for doc in retrieved_docs)
            if max_relevance < 0.7:
                gaps.append("Low semantic similarity - query may involve novel concepts")
        
        # Check for methodological coverage
        method_concepts = {"randomized_trials", "instrumental_variables", "diff_in_diff", 
                          "regression_discontinuity", "matching"}
        covered_methods = set()
        for doc in retrieved_docs:
            covered_methods.update(doc.document.causal_concepts)
        
        missing_methods = method_concepts - covered_methods
        if missing_methods and any(method in query.lower() for method in missing_methods):
            gaps.append(f"Missing methodological knowledge: {', '.join(missing_methods)}")
        
        return gaps if gaps else ["No significant knowledge gaps identified"]
    
    def _generate_recommendations(self, query: str, retrieved_docs: List[RetrievalResult],
                                knowledge_gaps: List[str]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if not retrieved_docs:
            recommendations.append("Add domain-specific causal knowledge to the system")
            recommendations.append("Consider consulting domain experts")
            return recommendations
        
        # Based on retrieved document types
        doc_types = {doc.document.doc_type for doc in retrieved_docs}
        
        if "research_paper" not in doc_types:
            recommendations.append("Consult recent research papers for cutting-edge methods")
        
        if "case_study" not in doc_types:
            recommendations.append("Review similar case studies for practical insights")
        
        # Based on knowledge gaps
        for gap in knowledge_gaps:
            if "domain-specific" in gap.lower():
                recommendations.append("Seek domain expert consultation")
            elif "methodological" in gap.lower():
                recommendations.append("Review methodological literature")
            elif "novel concepts" in gap.lower():
                recommendations.append("Consider experimental design approaches")
        
        # Quality-based recommendations
        avg_quality = sum(doc.document.quality_score for doc in retrieved_docs) / len(retrieved_docs)
        if avg_quality < 0.8:
            recommendations.append("Verify findings with high-quality sources")
        
        return recommendations if recommendations else ["Continue with current approach"]


# Convenience functions for integration
def create_rag_system(knowledge_base_path: Optional[str] = None,
                     llm_client=None) -> DynamicCausalRAG:
    """Create a new RAG system instance."""
    kb = CausalKnowledgeBase(knowledge_base_path)
    return DynamicCausalRAG(kb, llm_client)


async def enhance_causal_query(query: str, context: str = "",
                              domain: Optional[str] = None,
                              concepts: Optional[List[str]] = None,
                              rag_system: Optional[DynamicCausalRAG] = None) -> CausalRAGResponse:
    """Quick function to enhance a causal query with RAG."""
    if rag_system is None:
        rag_system = create_rag_system()
    
    return await rag_system.enhance_query(
        query=query,
        context=context,
        domain=domain,
        causal_concepts=concepts
    )