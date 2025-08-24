"""
Interactive Causal Questioning System

This module provides a dynamic Q&A system for causal exploration, allowing users to
ask natural language questions about causal relationships and receive intelligent
responses based on data analysis and causal reasoning.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import re
from datetime import datetime
import pandas as pd
import numpy as np

from causalllm.logging import get_logger


class QuestionType(Enum):
    """Types of causal questions."""
    CAUSAL_EFFECT = "causal_effect"           # "What is the effect of X on Y?"
    COUNTERFACTUAL = "counterfactual"         # "What would happen if...?"
    MECHANISM = "mechanism"                   # "How does X affect Y?"
    CONFOUNDING = "confounding"               # "What confounds the X-Y relationship?"
    MEDIATION = "mediation"                   # "Does Z mediate X->Y?"
    INTERACTION = "interaction"               # "Does X's effect depend on Z?"
    DISCOVERY = "discovery"                   # "What causes Y?"
    COMPARISON = "comparison"                 # "Which has a stronger effect, X1 or X2?"
    TEMPORAL = "temporal"                     # "When does X affect Y?"
    DOSE_RESPONSE = "dose_response"           # "How much X is needed to affect Y?"


class ConfidenceLevel(Enum):
    """Confidence levels for answers."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class CausalQuestion:
    """Structured representation of a causal question."""
    
    raw_question: str
    question_type: QuestionType
    treatment_variables: List[str]
    outcome_variables: List[str]
    conditioning_variables: List[str]
    temporal_context: Optional[str] = None
    domain_context: Optional[str] = None
    assumptions: List[str] = field(default_factory=list)


@dataclass
class CausalAnswer:
    """Structured answer to a causal question."""
    
    question: CausalQuestion
    main_answer: str
    confidence_level: ConfidenceLevel
    supporting_evidence: List[str]
    statistical_support: Dict[str, Any]
    limitations: List[str]
    alternative_explanations: List[str]
    follow_up_questions: List[str]
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    references: List[str] = field(default_factory=list)


@dataclass
class ConversationContext:
    """Context for ongoing causal conversation."""
    
    domain: str
    available_data: Optional[pd.DataFrame]
    variable_descriptions: Dict[str, str]
    discovered_relationships: Dict[str, Any]
    conversation_history: List[Tuple[CausalQuestion, CausalAnswer]]
    user_assumptions: List[str]
    focus_variables: List[str]


class InteractiveCausalQA:
    """Interactive causal questioning and answering system."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.logger = get_logger("causalllm.interactive_causal_qa")
        
        # Question parsing patterns
        self.question_patterns = {
            QuestionType.CAUSAL_EFFECT: [
                r"what.*effect.*of\s+(\w+).*on\s+(\w+)",
                r"does\s+(\w+).*affect\s+(\w+)",
                r"how.*does\s+(\w+).*influence\s+(\w+)",
                r"impact.*of\s+(\w+).*on\s+(\w+)"
            ],
            QuestionType.COUNTERFACTUAL: [
                r"what.*would.*happen.*if",
                r"what.*if.*(\w+).*were.*different",
                r"suppose.*(\w+).*then.*(\w+)"
            ],
            QuestionType.MECHANISM: [
                r"how.*does\s+(\w+).*work",
                r"what.*mechanism.*(\w+).*(\w+)",
                r"through.*what.*pathway"
            ],
            QuestionType.CONFOUNDING: [
                r"what.*confound.*(\w+).*(\w+)",
                r"what.*variables.*bias",
                r"what.*should.*control.*for"
            ],
            QuestionType.DISCOVERY: [
                r"what.*causes?\s+(\w+)",
                r"what.*determines\s+(\w+)",
                r"what.*factors.*affect\s+(\w+)"
            ]
        }
        
        # Domain-specific question handlers
        self.domain_handlers = {
            "healthcare": self._handle_healthcare_question,
            "business": self._handle_business_question,
            "education": self._handle_education_question,
            "general": self._handle_general_question
        }
    
    async def ask_causal_question(self, 
                                question: str,
                                context: ConversationContext) -> CausalAnswer:
        """
        Answer a causal question using LLM reasoning and available data.
        
        Args:
            question: Natural language causal question
            context: Conversation context with data and history
            
        Returns:
            Structured answer to the causal question
        """
        self.logger.info(f"Processing causal question: {question[:100]}...")
        
        # Step 1: Parse and classify the question
        parsed_question = await self._parse_question(question, context)
        
        # Step 2: Gather relevant information
        relevant_info = await self._gather_relevant_information(parsed_question, context)
        
        # Step 3: Generate answer based on question type
        answer = await self._generate_answer(parsed_question, relevant_info, context)
        
        # Step 4: Enhance with follow-up questions
        answer.follow_up_questions = await self._generate_follow_ups(answer, context)
        
        # Step 5: Update conversation context
        context.conversation_history.append((parsed_question, answer))
        
        self.logger.info("Causal question answered successfully")
        return answer
    
    async def _parse_question(self, 
                            question: str,
                            context: ConversationContext) -> CausalQuestion:
        """Parse natural language question into structured format."""
        
        question_lower = question.lower()
        
        # Detect question type using patterns
        question_type = QuestionType.CAUSAL_EFFECT  # default
        for qtype, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    question_type = qtype
                    break
        
        # Use LLM to extract variables and structure
        extraction_prompt = f"""
        Parse this causal question to extract key components:
        
        Question: "{question}"
        
        Available variables: {list(context.variable_descriptions.keys())}
        Domain: {context.domain}
        
        Extract:
        1. Treatment/intervention variables (what is being manipulated or compared)
        2. Outcome variables (what effects we're interested in)
        3. Conditioning/control variables (variables to hold constant)
        4. Any temporal context mentioned
        
        Respond with JSON:
        {{
            "question_type": "{question_type.value}",
            "treatment_variables": ["var1", "var2"],
            "outcome_variables": ["outcome1"],
            "conditioning_variables": ["control1", "control2"],
            "temporal_context": "before/after/during/etc or null",
            "key_assumptions": ["assumption1", "assumption2"]
        }}
        """
        
        try:
            response = await self.llm_client.generate_response(extraction_prompt)
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
                
                return CausalQuestion(
                    raw_question=question,
                    question_type=QuestionType(parsed_data.get("question_type", question_type.value)),
                    treatment_variables=parsed_data.get("treatment_variables", []),
                    outcome_variables=parsed_data.get("outcome_variables", []),
                    conditioning_variables=parsed_data.get("conditioning_variables", []),
                    temporal_context=parsed_data.get("temporal_context"),
                    domain_context=context.domain,
                    assumptions=parsed_data.get("key_assumptions", [])
                )
                
        except Exception as e:
            self.logger.error(f"Question parsing failed: {e}")
        
        # Fallback parsing
        return CausalQuestion(
            raw_question=question,
            question_type=question_type,
            treatment_variables=[],
            outcome_variables=[],
            conditioning_variables=[],
            domain_context=context.domain
        )
    
    async def _gather_relevant_information(self,
                                         question: CausalQuestion,
                                         context: ConversationContext) -> Dict[str, Any]:
        """Gather relevant information for answering the question."""
        
        relevant_info = {
            "question": question,
            "domain": context.domain,
            "available_variables": list(context.variable_descriptions.keys()),
            "variable_descriptions": context.variable_descriptions,
            "conversation_history": context.conversation_history[-5:],  # Last 5 exchanges
            "statistical_evidence": {},
            "prior_findings": {}
        }
        
        # Add data-specific information if available
        if context.available_data is not None:
            relevant_info["data_summary"] = self._summarize_data_for_question(
                context.available_data, question
            )
            
            # Calculate relevant statistics
            if question.treatment_variables and question.outcome_variables:
                stats = self._calculate_relevant_statistics(
                    context.available_data, question
                )
                relevant_info["statistical_evidence"] = stats
        
        # Add discovered relationships
        if context.discovered_relationships:
            relevant_relationships = self._filter_relevant_relationships(
                context.discovered_relationships, question
            )
            relevant_info["prior_findings"] = relevant_relationships
        
        return relevant_info
    
    def _summarize_data_for_question(self, 
                                   data: pd.DataFrame,
                                   question: CausalQuestion) -> Dict[str, Any]:
        """Summarize data relevant to the question."""
        
        summary = {
            "total_observations": len(data),
            "relevant_variables": {}
        }
        
        # Focus on variables mentioned in question
        relevant_vars = (question.treatment_variables + 
                        question.outcome_variables + 
                        question.conditioning_variables)
        
        for var in relevant_vars:
            if var in data.columns:
                if pd.api.types.is_numeric_dtype(data[var]):
                    summary["relevant_variables"][var] = {
                        "type": "numeric",
                        "mean": float(data[var].mean()) if not data[var].isnull().all() else None,
                        "std": float(data[var].std()) if not data[var].isnull().all() else None,
                        "range": [float(data[var].min()), float(data[var].max())] if not data[var].isnull().all() else None,
                        "missing_pct": float(data[var].isnull().mean() * 100)
                    }
                else:
                    summary["relevant_variables"][var] = {
                        "type": "categorical",
                        "unique_values": int(data[var].nunique()),
                        "categories": data[var].value_counts().head().to_dict(),
                        "missing_pct": float(data[var].isnull().mean() * 100)
                    }
        
        return summary
    
    def _calculate_relevant_statistics(self, 
                                     data: pd.DataFrame,
                                     question: CausalQuestion) -> Dict[str, Any]:
        """Calculate statistics relevant to the question."""
        
        stats = {}
        
        # Correlations between treatment and outcome variables
        if question.treatment_variables and question.outcome_variables:
            correlations = {}
            for treatment in question.treatment_variables:
                if treatment in data.columns:
                    for outcome in question.outcome_variables:
                        if outcome in data.columns:
                            try:
                                # Only for numeric variables
                                if (pd.api.types.is_numeric_dtype(data[treatment]) and 
                                    pd.api.types.is_numeric_dtype(data[outcome])):
                                    
                                    corr_data = data[[treatment, outcome]].dropna()
                                    if len(corr_data) > 10:
                                        from scipy import stats as scipy_stats
                                        corr, p_val = scipy_stats.pearsonr(
                                            corr_data[treatment], corr_data[outcome]
                                        )
                                        correlations[f"{treatment}->{outcome}"] = {
                                            "correlation": float(corr),
                                            "p_value": float(p_val),
                                            "n": len(corr_data)
                                        }
                            except Exception as e:
                                self.logger.warning(f"Could not calculate correlation: {e}")
                                continue
            
            stats["correlations"] = correlations
        
        # Group differences for categorical treatments
        if question.treatment_variables:
            group_comparisons = {}
            for treatment in question.treatment_variables:
                if treatment in data.columns and not pd.api.types.is_numeric_dtype(data[treatment]):
                    for outcome in question.outcome_variables:
                        if outcome in data.columns and pd.api.types.is_numeric_dtype(data[outcome]):
                            try:
                                groups = data.groupby(treatment)[outcome].agg(['mean', 'std', 'count'])
                                group_comparisons[f"{treatment}->{outcome}"] = groups.to_dict()
                            except Exception as e:
                                self.logger.warning(f"Could not calculate group comparison: {e}")
                                continue
            
            stats["group_comparisons"] = group_comparisons
        
        return stats
    
    def _filter_relevant_relationships(self,
                                     all_relationships: Dict[str, Any],
                                     question: CausalQuestion) -> Dict[str, Any]:
        """Filter relationships relevant to the current question."""
        
        relevant = {}
        question_vars = set(question.treatment_variables + 
                           question.outcome_variables + 
                           question.conditioning_variables)
        
        for relationship_id, relationship in all_relationships.items():
            # Check if relationship involves variables in the question
            if hasattr(relationship, 'source') and hasattr(relationship, 'target'):
                if relationship.source in question_vars or relationship.target in question_vars:
                    relevant[relationship_id] = relationship
            elif isinstance(relationship, dict):
                if (relationship.get('source') in question_vars or 
                    relationship.get('target') in question_vars):
                    relevant[relationship_id] = relationship
        
        return relevant
    
    async def _generate_answer(self,
                             question: CausalQuestion,
                             relevant_info: Dict[str, Any],
                             context: ConversationContext) -> CausalAnswer:
        """Generate an answer based on the question type and available information."""
        
        # Use domain-specific handler
        domain_handler = self.domain_handlers.get(
            context.domain, self.domain_handlers["general"]
        )
        
        return await domain_handler(question, relevant_info, context)
    
    async def _handle_general_question(self,
                                     question: CausalQuestion,
                                     relevant_info: Dict[str, Any],
                                     context: ConversationContext) -> CausalAnswer:
        """Handle general domain questions."""
        
        # Build comprehensive prompt
        answer_prompt = self._build_answer_prompt(question, relevant_info, context)
        
        try:
            response = await self.llm_client.generate_response(answer_prompt, max_tokens=2000)
            return self._parse_answer_response(response, question, relevant_info)
            
        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}")
            return self._create_fallback_answer(question, relevant_info)
    
    async def _handle_healthcare_question(self,
                                        question: CausalQuestion,
                                        relevant_info: Dict[str, Any],
                                        context: ConversationContext) -> CausalAnswer:
        """Handle healthcare-specific questions."""
        
        healthcare_context = """
        Healthcare-specific considerations:
        - Clinical significance vs. statistical significance
        - Patient safety and ethics
        - Confounding by indication
        - Temporal relationships (exposure -> disease)
        - Dose-response relationships
        - Biological plausibility
        """
        
        modified_info = relevant_info.copy()
        modified_info["domain_context"] = healthcare_context
        
        return await self._handle_general_question(question, modified_info, context)
    
    async def _handle_business_question(self,
                                      question: CausalQuestion,
                                      relevant_info: Dict[str, Any],
                                      context: ConversationContext) -> CausalAnswer:
        """Handle business-specific questions."""
        
        business_context = """
        Business-specific considerations:
        - Economic significance vs. statistical significance
        - ROI and cost-benefit analysis
        - Market conditions and external factors
        - Competitive responses
        - Time-to-effect and persistence
        - Scalability of interventions
        """
        
        modified_info = relevant_info.copy()
        modified_info["domain_context"] = business_context
        
        return await self._handle_general_question(question, modified_info, context)
    
    async def _handle_education_question(self,
                                       question: CausalQuestion,
                                       relevant_info: Dict[str, Any],
                                       context: ConversationContext) -> CausalAnswer:
        """Handle education-specific questions."""
        
        education_context = """
        Education-specific considerations:
        - Educational significance vs. statistical significance
        - Student heterogeneity and subgroup effects
        - Long-term vs. short-term learning outcomes
        - Transfer effects and skill generalization
        - Implementation fidelity in real classrooms
        - Equity and accessibility considerations
        """
        
        modified_info = relevant_info.copy()
        modified_info["domain_context"] = education_context
        
        return await self._handle_general_question(question, modified_info, context)
    
    def _build_answer_prompt(self,
                           question: CausalQuestion,
                           relevant_info: Dict[str, Any],
                           context: ConversationContext) -> str:
        """Build comprehensive prompt for answer generation."""
        
        # Format statistical evidence
        stats_section = ""
        if relevant_info.get("statistical_evidence"):
            stats_section = f"""
            STATISTICAL EVIDENCE:
            {json.dumps(relevant_info['statistical_evidence'], indent=2)}
            """
        
        # Format conversation history
        history_section = ""
        if relevant_info.get("conversation_history"):
            recent_qa = []
            for prev_q, prev_a in relevant_info["conversation_history"]:
                recent_qa.append(f"Q: {prev_q.raw_question}")
                recent_qa.append(f"A: {prev_a.main_answer[:200]}...")
            history_section = f"""
            RECENT CONVERSATION:
            {chr(10).join(recent_qa[-6:])}  # Last 3 Q&As
            """
        
        # Domain context
        domain_context = relevant_info.get("domain_context", "")
        
        prompt = f"""
        You are a causal inference expert answering a specific question about causal relationships.
        
        QUESTION: {question.raw_question}
        QUESTION TYPE: {question.question_type.value}
        DOMAIN: {question.domain_context}
        
        RELEVANT VARIABLES:
        Treatment variables: {question.treatment_variables}
        Outcome variables: {question.outcome_variables}
        Conditioning variables: {question.conditioning_variables}
        
        VARIABLE DESCRIPTIONS:
        {json.dumps(relevant_info.get('variable_descriptions', {}), indent=2)}
        
        {stats_section}
        {history_section}
        {domain_context}
        
        AVAILABLE DATA SUMMARY:
        {json.dumps(relevant_info.get('data_summary', {}), indent=2)}
        
        PRIOR FINDINGS:
        {json.dumps(relevant_info.get('prior_findings', {}), indent=2)}
        
        Please provide a comprehensive answer addressing:
        1. Direct answer to the question
        2. Confidence level and reasoning
        3. Supporting evidence from the data
        4. Key limitations and uncertainties
        5. Alternative explanations to consider
        
        Respond in JSON format:
        {{
            "main_answer": "Clear, direct answer to the question",
            "confidence_level": "very_low|low|moderate|high|very_high",
            "supporting_evidence": ["Evidence point 1", "Evidence point 2"],
            "statistical_support": {{"key": "value"}},
            "limitations": ["Limitation 1", "Limitation 2"],
            "alternative_explanations": ["Alternative 1", "Alternative 2"],
            "practical_implications": "What this means in practice",
            "recommendations": ["Next step 1", "Next step 2"]
        }}
        """
        
        return prompt
    
    def _parse_answer_response(self,
                             response: str,
                             question: CausalQuestion,
                             relevant_info: Dict[str, Any]) -> CausalAnswer:
        """Parse LLM response into structured answer."""
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
                
                confidence_mapping = {
                    "very_low": ConfidenceLevel.VERY_LOW,
                    "low": ConfidenceLevel.LOW,
                    "moderate": ConfidenceLevel.MODERATE,
                    "high": ConfidenceLevel.HIGH,
                    "very_high": ConfidenceLevel.VERY_HIGH
                }
                
                return CausalAnswer(
                    question=question,
                    main_answer=parsed_data.get("main_answer", "Unable to determine answer"),
                    confidence_level=confidence_mapping.get(
                        parsed_data.get("confidence_level", "moderate"),
                        ConfidenceLevel.MODERATE
                    ),
                    supporting_evidence=parsed_data.get("supporting_evidence", []),
                    statistical_support=parsed_data.get("statistical_support", {}),
                    limitations=parsed_data.get("limitations", []),
                    alternative_explanations=parsed_data.get("alternative_explanations", []),
                    follow_up_questions=[]  # Will be filled later
                )
                
        except json.JSONDecodeError:
            pass
        
        # Fallback parsing
        return self._create_fallback_answer(question, relevant_info)
    
    def _create_fallback_answer(self,
                              question: CausalQuestion,
                              relevant_info: Dict[str, Any]) -> CausalAnswer:
        """Create fallback answer when parsing fails."""
        
        return CausalAnswer(
            question=question,
            main_answer="Unable to provide a complete analysis due to parsing errors. Please rephrase your question.",
            confidence_level=ConfidenceLevel.LOW,
            supporting_evidence=["Analysis incomplete due to technical issues"],
            statistical_support={},
            limitations=["Unable to complete full analysis"],
            alternative_explanations=[],
            follow_up_questions=[]
        )
    
    async def _generate_follow_ups(self,
                                 answer: CausalAnswer,
                                 context: ConversationContext) -> List[str]:
        """Generate relevant follow-up questions."""
        
        follow_up_prompt = f"""
        Based on this causal question and answer, suggest 3-5 natural follow-up questions 
        that would deepen understanding of the causal relationship.
        
        ORIGINAL QUESTION: {answer.question.raw_question}
        ANSWER: {answer.main_answer}
        CONFIDENCE: {answer.confidence_level.value}
        
        Generate follow-up questions that explore:
        1. Mechanisms and pathways
        2. Boundary conditions and moderators
        3. Alternative explanations
        4. Practical applications
        5. Related causal relationships
        
        Return as a JSON array of strings:
        ["Question 1", "Question 2", "Question 3"]
        """
        
        try:
            response = await self.llm_client.generate_response(follow_up_prompt)
            
            # Parse JSON array
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                follow_ups = json.loads(json_match.group())
                return follow_ups[:5]  # Limit to 5 questions
                
        except Exception as e:
            self.logger.error(f"Follow-up generation failed: {e}")
        
        # Default follow-ups
        return [
            "What variables might confound this relationship?",
            "How strong is this causal effect?",
            "What are the practical implications of this finding?"
        ]
    
    def start_conversation(self,
                          domain: str,
                          data: Optional[pd.DataFrame] = None,
                          variable_descriptions: Optional[Dict[str, str]] = None) -> ConversationContext:
        """Start a new causal conversation session."""
        
        return ConversationContext(
            domain=domain,
            available_data=data,
            variable_descriptions=variable_descriptions or {},
            discovered_relationships={},
            conversation_history=[],
            user_assumptions=[],
            focus_variables=[]
        )
    
    async def suggest_questions(self, context: ConversationContext) -> List[str]:
        """Suggest interesting causal questions based on available data and context."""
        
        suggestion_prompt = f"""
        Suggest interesting causal questions that could be explored with the available data and context.
        
        DOMAIN: {context.domain}
        AVAILABLE VARIABLES: {list(context.variable_descriptions.keys())}
        VARIABLE DESCRIPTIONS: {context.variable_descriptions}
        
        DATA SUMMARY:
        - Number of observations: {len(context.available_data) if context.available_data is not None else 'Unknown'}
        - Variable types: {context.available_data.dtypes.to_dict() if context.available_data is not None else 'Unknown'}
        
        Generate 5-7 diverse causal questions covering:
        1. Direct causal effects
        2. Confounding relationships
        3. Mediating mechanisms
        4. Counterfactual scenarios
        5. Comparative effects
        
        Make questions specific to the available variables and domain.
        
        Return as JSON array: ["Question 1", "Question 2", ...]
        """
        
        try:
            response = await self.llm_client.generate_response(suggestion_prompt)
            
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
                return questions
                
        except Exception as e:
            self.logger.error(f"Question suggestion failed: {e}")
        
        # Fallback suggestions
        return [
            "What are the main causes of [outcome variable]?",
            "What is the effect of [treatment] on [outcome]?",
            "What variables might confound the relationship between X and Y?",
            "What would happen if we intervened on [variable]?"
        ]
    
    def export_conversation(self, context: ConversationContext, filepath: str):
        """Export conversation history to file."""
        
        export_data = {
            "domain": context.domain,
            "timestamp": datetime.now().isoformat(),
            "conversation_history": []
        }
        
        for question, answer in context.conversation_history:
            export_data["conversation_history"].append({
                "question": {
                    "raw_question": question.raw_question,
                    "question_type": question.question_type.value,
                    "treatment_variables": question.treatment_variables,
                    "outcome_variables": question.outcome_variables
                },
                "answer": {
                    "main_answer": answer.main_answer,
                    "confidence_level": answer.confidence_level.value,
                    "supporting_evidence": answer.supporting_evidence,
                    "limitations": answer.limitations
                }
            })
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)


# Convenience functions
def create_causal_qa_system(llm_client) -> InteractiveCausalQA:
    """Create an interactive causal Q&A system."""
    return InteractiveCausalQA(llm_client)


async def ask_causal_question_simple(question: str,
                                   data: pd.DataFrame,
                                   variable_descriptions: Dict[str, str],
                                   llm_client,
                                   domain: str = "general") -> CausalAnswer:
    """Simple function to ask a single causal question."""
    
    qa_system = create_causal_qa_system(llm_client)
    context = qa_system.start_conversation(domain, data, variable_descriptions)
    
    return await qa_system.ask_causal_question(question, context)