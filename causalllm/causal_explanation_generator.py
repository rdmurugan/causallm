"""
Advanced causal explanation generation system for Tier 2 capabilities.

This module provides sophisticated explanation generation for causal relationships,
counterfactual scenarios, and intervention outcomes using LLM-powered narrative
generation, multi-modal explanations, and adaptive explanation strategies.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import asyncio
import json
import time
from datetime import datetime
import textwrap

from causalllm.logging import get_logger


class ExplanationType(Enum):
    """Types of causal explanations."""
    MECHANISM = "mechanism"              # How does X cause Y?
    COUNTERFACTUAL = "counterfactual"   # What if X were different?
    CONTRASTIVE = "contrastive"         # Why X rather than Y?
    NECESSITY = "necessity"             # Is X necessary for Y?
    SUFFICIENCY = "sufficiency"         # Is X sufficient for Y?
    MEDIATION = "mediation"             # What mediates X → Y?
    MODERATION = "moderation"           # When does X cause Y?


class ExplanationAudience(Enum):
    """Target audience for explanations."""
    EXPERT = "expert"                   # Domain experts, researchers
    PRACTITIONER = "practitioner"       # Professionals using results
    GENERAL = "general"                 # General public
    STUDENT = "student"                 # Educational context
    STAKEHOLDER = "stakeholder"         # Decision-makers, funders


class ExplanationModality(Enum):
    """Modalities for explanations."""
    TEXTUAL = "textual"                 # Natural language explanation
    VISUAL = "visual"                   # Diagrams, graphs, charts
    INTERACTIVE = "interactive"         # Step-by-step exploration
    ANALOGICAL = "analogical"           # Comparisons, metaphors
    MATHEMATICAL = "mathematical"       # Equations, formulas
    NARRATIVE = "narrative"             # Story-like explanation


class CertaintyLevel(Enum):
    """Levels of certainty in explanations."""
    CERTAIN = "certain"                 # High confidence, strong evidence
    LIKELY = "likely"                   # Good evidence, some uncertainty
    POSSIBLE = "possible"               # Moderate evidence, more uncertainty  
    SPECULATIVE = "speculative"         # Limited evidence, high uncertainty
    UNKNOWN = "unknown"                 # Insufficient evidence


@dataclass
class ExplanationRequest:
    """Request for causal explanation."""
    
    explanation_type: ExplanationType
    cause_variable: str
    effect_variable: str
    audience: ExplanationAudience
    modality: ExplanationModality
    context: str
    specific_question: Optional[str] = None
    background_knowledge: Dict[str, str] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)  # e.g., "avoid technical jargon"
    desired_length: str = "medium"  # "brief", "medium", "detailed"


@dataclass
class ExplanationEvidence:
    """Evidence supporting a causal explanation."""
    
    evidence_type: str  # "statistical", "experimental", "observational", "theoretical"
    description: str
    strength: float  # 0-1 scale
    source: Optional[str] = None
    limitations: List[str] = field(default_factory=list)


@dataclass
class CausalExplanation:
    """A generated causal explanation."""
    
    request: ExplanationRequest
    main_explanation: str
    supporting_details: List[str]
    evidence: List[ExplanationEvidence]
    certainty_level: CertaintyLevel
    confidence_score: float
    alternative_explanations: List[str]
    limitations: List[str]
    follow_up_questions: List[str]
    visual_suggestions: List[str]  # Suggestions for visual aids
    analogies: List[str]
    key_concepts: Dict[str, str]  # Concept definitions
    reasoning_trace: List[str]
    generation_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplanationTemplate:
    """Template for generating explanations."""
    
    template_name: str
    explanation_type: ExplanationType
    audience: ExplanationAudience
    structure: List[str]  # Ordered sections
    prompt_template: str
    example_explanations: List[str] = field(default_factory=list)
    quality_criteria: List[str] = field(default_factory=list)


class CausalExplanationEngine(ABC):
    """Abstract base class for causal explanation generation."""
    
    def __init__(self):
        self.logger = get_logger("causalllm.causal_explanation_engine")
    
    @abstractmethod
    async def generate_explanation(self, request: ExplanationRequest,
                                 causal_data: Dict[str, Any]) -> CausalExplanation:
        """Generate causal explanation based on request."""
        pass


class LLMExplanationEngine(CausalExplanationEngine):
    """LLM-powered causal explanation generator."""
    
    def __init__(self, llm_client):
        super().__init__()
        self.llm_client = llm_client
        self.explanation_templates = self._initialize_templates()
        self.explanation_examples = self._initialize_examples()
    
    def _initialize_templates(self) -> Dict[Tuple[ExplanationType, ExplanationAudience], ExplanationTemplate]:
        """Initialize explanation templates for different types and audiences."""
        templates = {}
        
        # Mechanism explanation for practitioners
        templates[(ExplanationType.MECHANISM, ExplanationAudience.PRACTITIONER)] = ExplanationTemplate(
            template_name="mechanism_practitioner",
            explanation_type=ExplanationType.MECHANISM,
            audience=ExplanationAudience.PRACTITIONER,
            structure=["overview", "mechanism_steps", "evidence", "implications", "limitations"],
            prompt_template="""
            Explain HOW {cause} causes {effect} for a practitioner audience.
            Focus on actionable insights and practical mechanisms.
            Structure: Overview → Mechanism Steps → Evidence → Practical Implications → Limitations
            """,
            quality_criteria=["Clear mechanism description", "Practical relevance", "Evidence-based"]
        )
        
        # Counterfactual explanation for general audience  
        templates[(ExplanationType.COUNTERFACTUAL, ExplanationAudience.GENERAL)] = ExplanationTemplate(
            template_name="counterfactual_general",
            explanation_type=ExplanationType.COUNTERFACTUAL,
            audience=ExplanationAudience.GENERAL,
            structure=["scenario_setup", "counterfactual_contrast", "explanation", "real_world_examples"],
            prompt_template="""
            Explain what would happen if {cause} were different, affecting {effect}.
            Use simple language and relatable examples for a general audience.
            Structure: Scenario Setup → Counterfactual Contrast → Explanation → Real-world Examples
            """,
            quality_criteria=["Simple language", "Relatable examples", "Clear contrast"]
        )
        
        # Necessity explanation for experts
        templates[(ExplanationType.NECESSITY, ExplanationAudience.EXPERT)] = ExplanationTemplate(
            template_name="necessity_expert",
            explanation_type=ExplanationType.NECESSITY,
            audience=ExplanationAudience.EXPERT,
            structure=["necessity_analysis", "supporting_evidence", "alternative_causes", "methodology"],
            prompt_template="""
            Analyze whether {cause} is NECESSARY for {effect} to occur.
            Provide rigorous analysis suitable for domain experts.
            Structure: Necessity Analysis → Supporting Evidence → Alternative Causes → Methodological Considerations
            """,
            quality_criteria=["Rigorous analysis", "Comprehensive evidence", "Technical accuracy"]
        )
        
        return templates
    
    def _initialize_examples(self) -> Dict[str, List[str]]:
        """Initialize high-quality explanation examples for few-shot learning."""
        return {
            "mechanism_healthcare": [
                """
                How does exercise cause improved cardiovascular health?
                
                **Overview**: Regular exercise strengthens the cardiovascular system through multiple physiological adaptations that improve heart function and blood vessel health.
                
                **Mechanism Steps**:
                1. **Cardiac Strengthening**: Exercise increases heart muscle workload, leading to cardiac muscle hypertrophy and improved pumping efficiency
                2. **Vascular Adaptation**: Physical activity stimulates nitric oxide production, improving blood vessel flexibility and reducing arterial stiffness
                3. **Metabolic Improvements**: Exercise enhances lipid metabolism, reducing harmful cholesterol and increasing beneficial HDL cholesterol
                4. **Blood Pressure Regulation**: Regular activity improves baroreceptor sensitivity and reduces peripheral resistance
                
                **Evidence**: Multiple randomized controlled trials show 20-30% reduction in cardiovascular events with regular moderate exercise. Meta-analyses consistently demonstrate dose-response relationships.
                
                **Practical Implications**: 150 minutes of moderate exercise per week provides substantial cardiovascular benefits. Effects are measurable within 8-12 weeks of starting a program.
                
                **Limitations**: Individual genetic factors affect response magnitude. Benefits require sustained activity and diminish if exercise is discontinued.
                """
            ],
            "counterfactual_economics": [
                """
                What if interest rates had not been lowered during the 2008 financial crisis?
                
                **Scenario Setup**: In reality, central banks worldwide dramatically lowered interest rates to near-zero levels during 2008-2010 to stimulate economic recovery.
                
                **Counterfactual Contrast**: If interest rates had remained at pre-crisis levels (around 5-6%), the economic trajectory would have been dramatically different.
                
                **Explanation**: Higher interest rates would have:
                - Made borrowing more expensive, further reducing business investment and consumer spending
                - Increased mortgage payments, accelerating foreclosures and housing market collapse
                - Strengthened the currency but hurt exports
                - Led to deeper recession with unemployment potentially reaching 15-20%
                - Caused more bank failures as businesses and consumers defaulted on loans
                
                **Real-world Examples**: Similar scenarios occurred during the Great Depression when monetary policy remained restrictive, prolonging economic hardship. Countries that lowered rates faster (like Australia) experienced milder recessions.
                """
            ]
        }
    
    async def generate_explanation(self, request: ExplanationRequest,
                                 causal_data: Dict[str, Any]) -> CausalExplanation:
        """Generate causal explanation using LLM."""
        self.logger.info(f"Generating {request.explanation_type.value} explanation for {request.audience.value} audience")
        
        start_time = time.time()
        reasoning_trace = []
        
        # Step 1: Select appropriate template
        template = self._select_template(request)
        reasoning_trace.append(f"Selected template: {template.template_name if template else 'default'}")
        
        # Step 2: Gather relevant evidence
        evidence = await self._gather_evidence(request, causal_data, reasoning_trace)
        
        # Step 3: Generate explanation using LLM
        explanation = await self._generate_llm_explanation(request, template, evidence, reasoning_trace)
        
        # Step 4: Enhance with additional content
        explanation = await self._enhance_explanation(explanation, request, reasoning_trace)
        
        # Step 5: Validate and refine
        explanation = await self._validate_explanation(explanation, reasoning_trace)
        
        # Add metadata
        explanation.generation_metadata = {
            "template_used": template.template_name if template else "default",
            "evidence_sources": len(evidence),
            "generation_time": time.time() - start_time,
            "reasoning_steps": len(reasoning_trace)
        }
        explanation.reasoning_trace = reasoning_trace
        
        self.logger.info(f"Generated explanation with {len(explanation.main_explanation)} characters")
        return explanation
    
    def _select_template(self, request: ExplanationRequest) -> Optional[ExplanationTemplate]:
        """Select appropriate template based on request."""
        template_key = (request.explanation_type, request.audience)
        return self.explanation_templates.get(template_key)
    
    async def _gather_evidence(self, request: ExplanationRequest,
                             causal_data: Dict[str, Any],
                             reasoning_trace: List[str]) -> List[ExplanationEvidence]:
        """Gather evidence to support the causal explanation."""
        reasoning_trace.append("Gathering evidence for explanation")
        
        evidence = []
        
        # Extract evidence from causal data
        if "statistical_evidence" in causal_data:
            stat_evidence = causal_data["statistical_evidence"]
            evidence.append(ExplanationEvidence(
                evidence_type="statistical",
                description=f"Statistical analysis shows correlation of {stat_evidence.get('correlation', 'unknown')}",
                strength=min(abs(stat_evidence.get('correlation', 0)), 1.0),
                limitations=["Correlation does not imply causation"]
            ))
        
        if "experimental_data" in causal_data:
            exp_data = causal_data["experimental_data"]
            evidence.append(ExplanationEvidence(
                evidence_type="experimental",
                description=f"Experimental study with {exp_data.get('sample_size', 'unknown')} participants",
                strength=exp_data.get('effect_size', 0.5),
                limitations=exp_data.get('limitations', [])
            ))
        
        if "domain_knowledge" in causal_data:
            domain_knowledge = causal_data["domain_knowledge"]
            evidence.append(ExplanationEvidence(
                evidence_type="theoretical",
                description=domain_knowledge.get('description', 'Domain knowledge supports this relationship'),
                strength=domain_knowledge.get('confidence', 0.7),
                source=domain_knowledge.get('source', 'Domain expertise')
            ))
        
        reasoning_trace.append(f"Gathered {len(evidence)} pieces of evidence")
        return evidence
    
    async def _generate_llm_explanation(self, request: ExplanationRequest,
                                      template: Optional[ExplanationTemplate],
                                      evidence: List[ExplanationEvidence],
                                      reasoning_trace: List[str]) -> CausalExplanation:
        """Generate core explanation using LLM."""
        reasoning_trace.append("Generating explanation with LLM")
        
        # Build prompt
        prompt = self._build_explanation_prompt(request, template, evidence)
        
        try:
            # Generate explanation
            if hasattr(self.llm_client, 'generate_response'):
                response = await self.llm_client.generate_response(prompt)
            else:
                response = await asyncio.to_thread(self.llm_client.generate, prompt)
            
            # Parse response
            explanation = self._parse_llm_response(response, request, evidence)
            reasoning_trace.append("Successfully generated LLM explanation")
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"LLM explanation generation failed: {e}")
            reasoning_trace.append(f"LLM generation failed: {e}")
            
            # Fallback to template-based explanation
            return self._generate_fallback_explanation(request, evidence)
    
    def _build_explanation_prompt(self, request: ExplanationRequest,
                                 template: Optional[ExplanationTemplate],
                                 evidence: List[ExplanationEvidence]) -> str:
        """Build comprehensive prompt for LLM explanation generation."""
        
        prompt_parts = [
            "You are an expert in causal explanation generation.",
            f"Generate a {request.explanation_type.value} explanation for a {request.audience.value} audience.",
            ""
        ]
        
        # Add specific question if provided
        if request.specific_question:
            prompt_parts.append(f"Question to answer: {request.specific_question}")
        else:
            prompt_parts.append(f"Explain the causal relationship between {request.cause_variable} and {request.effect_variable}")
        
        # Add context
        if request.context:
            prompt_parts.extend([
                "",
                f"Context: {request.context}"
            ])
        
        # Add background knowledge
        if request.background_knowledge:
            prompt_parts.extend([
                "",
                "Background Knowledge:"
            ])
            for concept, definition in request.background_knowledge.items():
                prompt_parts.append(f"- {concept}: {definition}")
        
        # Add evidence
        if evidence:
            prompt_parts.extend([
                "",
                "Available Evidence:"
            ])
            for i, ev in enumerate(evidence, 1):
                prompt_parts.append(f"{i}. {ev.evidence_type}: {ev.description} (strength: {ev.strength:.2f})")
                if ev.limitations:
                    prompt_parts.append(f"   Limitations: {', '.join(ev.limitations)}")
        
        # Add template structure if available
        if template:
            prompt_parts.extend([
                "",
                f"Structure your explanation with these sections: {' → '.join(template.structure)}"
            ])
        
        # Add audience-specific guidelines
        audience_guidelines = {
            ExplanationAudience.EXPERT: "Use technical terminology. Focus on methodological rigor and comprehensive analysis.",
            ExplanationAudience.PRACTITIONER: "Focus on actionable insights. Balance technical accuracy with practical utility.",
            ExplanationAudience.GENERAL: "Use simple language. Avoid jargon. Include relatable examples and analogies.",
            ExplanationAudience.STUDENT: "Be educational. Build understanding step-by-step. Include learning objectives.",
            ExplanationAudience.STAKEHOLDER: "Focus on implications and decisions. Highlight costs, benefits, and risks."
        }
        
        if request.audience in audience_guidelines:
            prompt_parts.extend([
                "",
                f"Audience guidelines: {audience_guidelines[request.audience]}"
            ])
        
        # Add constraints
        if request.constraints:
            prompt_parts.extend([
                "",
                "Constraints:",
                *[f"- {constraint}" for constraint in request.constraints]
            ])
        
        # Add few-shot examples if available
        example_key = f"{request.explanation_type.value}_{request.context.split()[0] if request.context else 'general'}"
        if example_key in self.explanation_examples:
            prompt_parts.extend([
                "",
                "Example of high-quality explanation:",
                self.explanation_examples[example_key][0]
            ])
        
        # Add output format requirements
        prompt_parts.extend([
            "",
            "Provide a JSON response with:",
            "{",
            '  "main_explanation": "core explanation text",',
            '  "supporting_details": ["detail 1", "detail 2"],',
            '  "certainty_level": "certain/likely/possible/speculative/unknown",',
            '  "confidence_score": 0.8,',
            '  "alternative_explanations": ["alternative 1"],',
            '  "limitations": ["limitation 1"],',
            '  "follow_up_questions": ["question 1"],',
            '  "key_concepts": {"concept": "definition"},',
            '  "analogies": ["analogy 1"]',
            "}"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_llm_response(self, response: str, request: ExplanationRequest,
                           evidence: List[ExplanationEvidence]) -> CausalExplanation:
        """Parse LLM response into CausalExplanation object."""
        
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # Map certainty level
                certainty_str = data.get("certainty_level", "possible").lower()
                certainty_level = CertaintyLevel.POSSIBLE
                for level in CertaintyLevel:
                    if level.value == certainty_str:
                        certainty_level = level
                        break
                
                explanation = CausalExplanation(
                    request=request,
                    main_explanation=data.get("main_explanation", "Explanation could not be generated."),
                    supporting_details=data.get("supporting_details", []),
                    evidence=evidence,
                    certainty_level=certainty_level,
                    confidence_score=data.get("confidence_score", 0.5),
                    alternative_explanations=data.get("alternative_explanations", []),
                    limitations=data.get("limitations", []),
                    follow_up_questions=data.get("follow_up_questions", []),
                    visual_suggestions=[],  # Will be added in enhancement step
                    analogies=data.get("analogies", []),
                    key_concepts=data.get("key_concepts", {}),
                    reasoning_trace=[]
                )
                
                return explanation
                
            else:
                # Fallback: use raw text as main explanation
                return CausalExplanation(
                    request=request,
                    main_explanation=response[:2000],  # Limit length
                    supporting_details=[],
                    evidence=evidence,
                    certainty_level=CertaintyLevel.POSSIBLE,
                    confidence_score=0.4,  # Lower confidence for unparsed response
                    alternative_explanations=[],
                    limitations=["Explanation could not be fully structured"],
                    follow_up_questions=[],
                    visual_suggestions=[],
                    analogies=[],
                    key_concepts={},
                    reasoning_trace=[]
                )
                
        except json.JSONDecodeError as e:
            self.logger.warning(f"Could not parse LLM JSON response: {e}")
            return self._generate_fallback_explanation(request, evidence)
    
    def _generate_fallback_explanation(self, request: ExplanationRequest,
                                     evidence: List[ExplanationEvidence]) -> CausalExplanation:
        """Generate fallback explanation when LLM fails."""
        
        fallback_explanations = {
            ExplanationType.MECHANISM: f"The mechanism by which {request.cause_variable} causes {request.effect_variable} involves multiple pathways that require further investigation.",
            ExplanationType.COUNTERFACTUAL: f"If {request.cause_variable} were different, we would expect {request.effect_variable} to change accordingly, though the exact nature of this change requires more analysis.",
            ExplanationType.NECESSITY: f"The necessity of {request.cause_variable} for {request.effect_variable} depends on various contextual factors and alternative causal pathways.",
            ExplanationType.SUFFICIENCY: f"Whether {request.cause_variable} is sufficient to cause {request.effect_variable} requires examination of additional contributing factors."
        }
        
        main_explanation = fallback_explanations.get(
            request.explanation_type,
            f"The relationship between {request.cause_variable} and {request.effect_variable} requires careful analysis to understand fully."
        )
        
        return CausalExplanation(
            request=request,
            main_explanation=main_explanation,
            supporting_details=[],
            evidence=evidence,
            certainty_level=CertaintyLevel.UNKNOWN,
            confidence_score=0.2,
            alternative_explanations=[],
            limitations=["Explanation generation failed - fallback used"],
            follow_up_questions=[],
            visual_suggestions=[],
            analogies=[],
            key_concepts={},
            reasoning_trace=[]
        )
    
    async def _enhance_explanation(self, explanation: CausalExplanation,
                                 request: ExplanationRequest,
                                 reasoning_trace: List[str]) -> CausalExplanation:
        """Enhance explanation with additional content."""
        reasoning_trace.append("Enhancing explanation with visual suggestions and analogies")
        
        # Add visual suggestions based on explanation type and audience
        visual_suggestions = self._generate_visual_suggestions(request)
        explanation.visual_suggestions = visual_suggestions
        
        # Enhance analogies if audience is general or student
        if request.audience in [ExplanationAudience.GENERAL, ExplanationAudience.STUDENT]:
            if not explanation.analogies:
                explanation.analogies = await self._generate_analogies(request)
        
        # Add domain-specific key concepts
        if not explanation.key_concepts:
            explanation.key_concepts = self._extract_key_concepts(request, explanation)
        
        return explanation
    
    def _generate_visual_suggestions(self, request: ExplanationRequest) -> List[str]:
        """Generate suggestions for visual aids."""
        visual_suggestions = []
        
        if request.explanation_type == ExplanationType.MECHANISM:
            visual_suggestions.extend([
                "Flow diagram showing causal pathway",
                "Process chart with intermediate steps", 
                "Network diagram showing causal connections"
            ])
        
        elif request.explanation_type == ExplanationType.COUNTERFACTUAL:
            visual_suggestions.extend([
                "Side-by-side comparison charts",
                "Timeline showing alternative scenarios",
                "Decision tree visualization"
            ])
        
        elif request.explanation_type == ExplanationType.MEDIATION:
            visual_suggestions.extend([
                "Mediation path diagram",
                "Causal chain visualization",
                "Statistical path model"
            ])
        
        # Add audience-specific suggestions
        if request.audience == ExplanationAudience.GENERAL:
            visual_suggestions.append("Simple infographic with icons")
            visual_suggestions.append("Before/after comparison images")
        
        elif request.audience == ExplanationAudience.EXPERT:
            visual_suggestions.append("Statistical plots and confidence intervals")
            visual_suggestions.append("Technical diagrams with parameters")
        
        return visual_suggestions
    
    async def _generate_analogies(self, request: ExplanationRequest) -> List[str]:
        """Generate analogies to help explain causal relationships."""
        # This could be enhanced with LLM generation, but providing simple fallbacks
        analogy_templates = {
            ExplanationType.MECHANISM: [
                f"{request.cause_variable} acts like a key that unlocks {request.effect_variable}",
                f"Think of {request.cause_variable} as a domino that causes {request.effect_variable} to fall"
            ],
            ExplanationType.COUNTERFACTUAL: [
                f"Imagine {request.cause_variable} as a fork in the road - changing it leads down a different path to {request.effect_variable}",
                f"Like a recipe - changing {request.cause_variable} changes the final dish ({request.effect_variable})"
            ]
        }
        
        return analogy_templates.get(request.explanation_type, [])
    
    def _extract_key_concepts(self, request: ExplanationRequest, 
                            explanation: CausalExplanation) -> Dict[str, str]:
        """Extract and define key concepts from the explanation."""
        key_concepts = {}
        
        # Add basic causal concepts
        key_concepts["Causation"] = "A relationship where one event (cause) brings about another event (effect)"
        
        if request.explanation_type == ExplanationType.COUNTERFACTUAL:
            key_concepts["Counterfactual"] = "A hypothetical scenario describing what would have happened under different circumstances"
        
        elif request.explanation_type == ExplanationType.MEDIATION:
            key_concepts["Mediation"] = "A causal pathway where the effect of the cause on the outcome operates through an intermediate variable"
        
        elif request.explanation_type == ExplanationType.MODERATION:
            key_concepts["Moderation"] = "When the strength or direction of a causal relationship depends on the level of another variable"
        
        return key_concepts
    
    async def _validate_explanation(self, explanation: CausalExplanation,
                                  reasoning_trace: List[str]) -> CausalExplanation:
        """Validate and potentially improve the explanation."""
        reasoning_trace.append("Validating explanation quality")
        
        # Check explanation length appropriateness
        length_requirements = {
            "brief": (50, 200),
            "medium": (200, 800),
            "detailed": (800, 2000)
        }
        
        desired_length = explanation.request.desired_length
        if desired_length in length_requirements:
            min_len, max_len = length_requirements[desired_length]
            actual_len = len(explanation.main_explanation)
            
            if actual_len < min_len:
                explanation.limitations.append(f"Explanation may be too brief ({actual_len} chars, target: {min_len}+)")
            elif actual_len > max_len:
                explanation.limitations.append(f"Explanation may be too lengthy ({actual_len} chars, target: <{max_len})")
        
        # Validate audience appropriateness
        if explanation.request.audience == ExplanationAudience.GENERAL:
            # Check for technical jargon (simplified check)
            technical_terms = ["coefficient", "regression", "parameter", "statistical significance", "p-value"]
            found_jargon = [term for term in technical_terms if term in explanation.main_explanation.lower()]
            if found_jargon:
                explanation.limitations.append(f"May contain technical jargon: {', '.join(found_jargon)}")
        
        # Adjust confidence based on validation results
        if len(explanation.limitations) > 3:
            explanation.confidence_score = max(0.1, explanation.confidence_score - 0.2)
        
        return explanation


class AdaptiveExplanationGenerator:
    """Adaptive explanation generator that learns from user feedback."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.base_engine = LLMExplanationEngine(llm_client)
        self.logger = get_logger("causalllm.adaptive_explanation_generator")
        
        # Learning components
        self.user_feedback_history = []
        self.successful_explanations = []
        self.audience_preferences = {}
        self.explanation_performance = {}
    
    async def generate_adaptive_explanation(self, request: ExplanationRequest,
                                          causal_data: Dict[str, Any],
                                          user_context: Optional[Dict[str, Any]] = None) -> CausalExplanation:
        """Generate explanation adapted to user preferences and feedback."""
        self.logger.info("Generating adaptive explanation")
        
        # Adapt request based on learned preferences
        adapted_request = await self._adapt_request(request, user_context)
        
        # Generate base explanation
        explanation = await self.base_engine.generate_explanation(adapted_request, causal_data)
        
        # Apply adaptive enhancements
        explanation = await self._apply_adaptive_enhancements(explanation, user_context)
        
        return explanation
    
    async def _adapt_request(self, request: ExplanationRequest,
                           user_context: Optional[Dict[str, Any]]) -> ExplanationRequest:
        """Adapt explanation request based on learned user preferences."""
        
        adapted_request = request
        
        # Check audience preferences
        audience_key = request.audience.value
        if audience_key in self.audience_preferences:
            prefs = self.audience_preferences[audience_key]
            
            # Adapt length preference
            if "preferred_length" in prefs:
                adapted_request.desired_length = prefs["preferred_length"]
            
            # Adapt constraints based on feedback
            if "avoid_jargon" in prefs and prefs["avoid_jargon"]:
                if "avoid technical jargon" not in adapted_request.constraints:
                    adapted_request.constraints.append("avoid technical jargon")
        
        return adapted_request
    
    async def _apply_adaptive_enhancements(self, explanation: CausalExplanation,
                                         user_context: Optional[Dict[str, Any]]) -> CausalExplanation:
        """Apply enhancements based on successful patterns."""
        
        # If we have successful examples for this type/audience combination
        combo_key = (explanation.request.explanation_type.value, explanation.request.audience.value)
        
        if combo_key in self.explanation_performance:
            performance_data = self.explanation_performance[combo_key]
            
            # Boost confidence if this combination has been successful
            if performance_data.get("success_rate", 0) > 0.8:
                explanation.confidence_score = min(0.95, explanation.confidence_score + 0.1)
            
            # Add successful patterns
            if "successful_analogies" in performance_data:
                explanation.analogies.extend(performance_data["successful_analogies"])
        
        return explanation
    
    def record_user_feedback(self, explanation: CausalExplanation, 
                           feedback: Dict[str, Any]):
        """Record user feedback to improve future explanations."""
        feedback_entry = {
            "timestamp": datetime.now(),
            "explanation_id": id(explanation),
            "explanation_type": explanation.request.explanation_type.value,
            "audience": explanation.request.audience.value,
            "feedback": feedback
        }
        
        self.user_feedback_history.append(feedback_entry)
        
        # Update preferences
        self._update_preferences(explanation, feedback)
        
        # Update performance metrics
        self._update_performance_metrics(explanation, feedback)
        
        self.logger.info(f"Recorded feedback: {feedback.get('rating', 'no rating')}")
    
    def _update_preferences(self, explanation: CausalExplanation, feedback: Dict[str, Any]):
        """Update audience preferences based on feedback."""
        audience_key = explanation.request.audience.value
        
        if audience_key not in self.audience_preferences:
            self.audience_preferences[audience_key] = {}
        
        prefs = self.audience_preferences[audience_key]
        
        # Update length preferences
        if "length_feedback" in feedback:
            length_feedback = feedback["length_feedback"]
            if length_feedback == "too_short":
                prefs["preferred_length"] = "detailed" if explanation.request.desired_length == "medium" else "medium"
            elif length_feedback == "too_long":
                prefs["preferred_length"] = "brief" if explanation.request.desired_length == "medium" else "medium"
        
        # Update jargon preferences
        if "jargon_feedback" in feedback:
            prefs["avoid_jargon"] = feedback["jargon_feedback"] == "too_technical"
    
    def _update_performance_metrics(self, explanation: CausalExplanation, feedback: Dict[str, Any]):
        """Update performance metrics for explanation types and audiences."""
        combo_key = (explanation.request.explanation_type.value, explanation.request.audience.value)
        
        if combo_key not in self.explanation_performance:
            self.explanation_performance[combo_key] = {
                "total_explanations": 0,
                "successful_explanations": 0,
                "success_rate": 0.0,
                "successful_analogies": []
            }
        
        performance = self.explanation_performance[combo_key]
        performance["total_explanations"] += 1
        
        # Consider feedback rating > 3 as successful
        if feedback.get("rating", 0) > 3:
            performance["successful_explanations"] += 1
            
            # Record successful patterns
            if explanation.analogies and feedback.get("analogies_helpful", False):
                performance["successful_analogies"].extend(explanation.analogies)
        
        performance["success_rate"] = performance["successful_explanations"] / performance["total_explanations"]
    
    def get_explanation_statistics(self) -> Dict[str, Any]:
        """Get statistics about explanation performance."""
        return {
            "total_explanations_generated": sum(p["total_explanations"] for p in self.explanation_performance.values()),
            "overall_success_rate": np.mean([p["success_rate"] for p in self.explanation_performance.values()]) if self.explanation_performance else 0,
            "feedback_entries": len(self.user_feedback_history),
            "audience_preferences": dict(self.audience_preferences),
            "performance_by_type": dict(self.explanation_performance)
        }


# Convenience functions
def create_explanation_engine(llm_client) -> LLMExplanationEngine:
    """Create a causal explanation engine."""
    return LLMExplanationEngine(llm_client)


async def generate_causal_explanation(cause_variable: str,
                                    effect_variable: str,
                                    explanation_type: ExplanationType,
                                    audience: ExplanationAudience,
                                    context: str,
                                    causal_data: Dict[str, Any],
                                    llm_client,
                                    **kwargs) -> CausalExplanation:
    """Quick function to generate causal explanation."""
    
    request = ExplanationRequest(
        explanation_type=explanation_type,
        cause_variable=cause_variable,
        effect_variable=effect_variable,
        audience=audience,
        modality=ExplanationModality.TEXTUAL,
        context=context,
        **kwargs
    )
    
    engine = create_explanation_engine(llm_client)
    return await engine.generate_explanation(request, causal_data)