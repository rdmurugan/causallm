"""
Multi-Agent Causal Reasoning System.

This module implements a collaborative multi-agent approach to causal analysis:
- Domain Expert Agent: Provides context-specific insights
- Statistician Agent: Validates statistical assumptions and methods  
- Skeptic Agent: Challenges claims and identifies weaknesses
- Synthesizer Agent: Combines perspectives into coherent analysis
- Coordinator Agent: Orchestrates the analysis workflow

Each agent has specialized prompts and reasoning patterns optimized for their role.
"""

import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor
from causalllm.logging import get_logger, get_structured_logger
from causalllm.llm_client import BaseLLMClient, get_llm_client


class AgentRole(Enum):
    """Agent roles in the multi-agent system."""
    DOMAIN_EXPERT = "domain_expert"
    STATISTICIAN = "statistician"
    SKEPTIC = "skeptic"
    SYNTHESIZER = "synthesizer"
    COORDINATOR = "coordinator"


@dataclass
class AgentResponse:
    """Response from an individual agent."""
    agent_role: AgentRole
    analysis: str
    confidence: float
    key_points: List[str]
    concerns: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CollaborativeAnalysis:
    """Final collaborative analysis result."""
    primary_conclusion: str
    consensus_points: List[str]
    disagreements: List[str]
    confidence_level: float
    agent_responses: List[AgentResponse]
    synthesis_reasoning: str
    recommendations: List[str]
    limitations: List[str]


class CausalAgent:
    """
    Base class for causal analysis agents.
    
    Each agent has specialized prompts and reasoning patterns
    optimized for their specific role in the analysis process.
    """
    
    def __init__(self, 
                 role: AgentRole,
                 llm_client: BaseLLMClient,
                 domain: str = "general",
                 temperature: float = 0.7):
        """
        Initialize causal agent.
        
        Args:
            role: Agent's role in the analysis
            llm_client: LLM client for generating responses
            domain: Domain specialization (healthcare, finance, marketing, etc.)
            temperature: Temperature for LLM generation
        """
        self.role = role
        self.llm_client = llm_client
        self.domain = domain
        self.temperature = temperature
        
        self.logger = get_logger(f"causalllm.llm_agents.{role.value}")
        self.struct_logger = get_structured_logger(f"agent_{role.value}")
        
        # Agent-specific system prompts
        self.system_prompt = self._get_system_prompt()
        
        self.logger.info(f"Initialized {role.value} agent for {domain} domain")
    
    def _get_system_prompt(self) -> str:
        """Get role-specific system prompt."""
        
        if self.role == AgentRole.DOMAIN_EXPERT:
            return f"""You are a {self.domain} domain expert with deep knowledge of causal relationships in this field.

Your role is to:
- Provide context-specific insights and domain knowledge
- Identify relevant mechanisms and pathways
- Suggest domain-appropriate interpretations
- Flag domain-specific confounders and considerations
- Draw on established research and best practices in {self.domain}

Focus on practical, real-world insights that statistical methods alone might miss.
Always ground your analysis in domain expertise and established knowledge."""

        elif self.role == AgentRole.STATISTICIAN:
            return """You are a biostatistician and causal inference expert with deep knowledge of statistical methods.

Your role is to:
- Validate statistical assumptions and methods
- Identify potential sources of bias
- Suggest appropriate statistical techniques
- Assess the strength of causal evidence
- Flag violations of causal inference assumptions
- Recommend sensitivity analyses and robustness checks

Focus on methodological rigor and statistical validity.
Always consider the limitations and assumptions of the proposed analysis."""

        elif self.role == AgentRole.SKEPTIC:
            return """You are a critical thinker whose role is to challenge causal claims and identify weaknesses.

Your role is to:
- Question causal assumptions and interpretations
- Identify alternative explanations for observed patterns
- Point out potential confounders and biases
- Challenge the strength of evidence presented
- Suggest competing hypotheses
- Flag overstated or unsupported conclusions

Focus on intellectual rigor and scientific skepticism.
Always ask "What else could explain these results?" and "What could go wrong?"."""

        elif self.role == AgentRole.SYNTHESIZER:
            return """You are a synthesis expert who integrates multiple perspectives into coherent analysis.

Your role is to:
- Combine insights from domain experts, statisticians, and skeptics
- Identify areas of consensus and disagreement  
- Weigh competing arguments and evidence
- Create balanced, nuanced conclusions
- Acknowledge uncertainty and limitations
- Provide actionable recommendations

Focus on creating comprehensive, balanced analysis that acknowledges multiple viewpoints.
Always strive for nuanced conclusions that reflect the complexity of causal inference."""

        else:  # COORDINATOR
            return """You are a research coordinator who orchestrates collaborative causal analysis.

Your role is to:
- Define clear analysis objectives and scope
- Coordinate between different expert perspectives
- Ensure comprehensive coverage of key issues
- Facilitate productive disagreement and debate
- Synthesize findings into actionable insights
- Manage the overall analysis workflow

Focus on ensuring thorough, systematic analysis that leverages each agent's strengths."""
    
    async def analyze(self, 
                     context: str,
                     factual: str = None,
                     intervention: str = None,
                     question: str = None,
                     previous_analyses: List[AgentResponse] = None) -> AgentResponse:
        """
        Perform role-specific analysis.
        
        Args:
            context: Background context for analysis
            factual: Factual scenario (for counterfactuals)
            intervention: Intervention description
            question: Specific question to address
            previous_analyses: Previous agent analyses to consider
            
        Returns:
            AgentResponse with role-specific analysis
        """
        self.logger.info(f"Starting {self.role.value} analysis")
        
        # Build role-specific prompt
        prompt = self._build_analysis_prompt(
            context, factual, intervention, question, previous_analyses
        )
        
        # Generate response
        try:
            response = self.llm_client.chat(prompt, temperature=self.temperature)
            
            # Parse and structure response
            parsed_response = self._parse_response(response)
            
            self.struct_logger.log_interaction(
                f"{self.role.value}_analysis",
                {
                    "context_length": len(context),
                    "response_length": len(response),
                    "confidence": parsed_response.confidence,
                    "n_key_points": len(parsed_response.key_points),
                    "n_concerns": len(parsed_response.concerns)
                }
            )
            
            self.logger.info(f"Completed {self.role.value} analysis with confidence {parsed_response.confidence:.2f}")
            return parsed_response
            
        except Exception as e:
            self.logger.error(f"{self.role.value} analysis failed: {e}")
            self.struct_logger.log_error(e, {"role": self.role.value})
            raise RuntimeError(f"Agent analysis failed: {e}")
    
    def _build_analysis_prompt(self, 
                              context: str,
                              factual: Optional[str],
                              intervention: Optional[str], 
                              question: Optional[str],
                              previous_analyses: Optional[List[AgentResponse]]) -> str:
        """Build role-specific analysis prompt."""
        
        # Base prompt with system instructions
        prompt = f"{self.system_prompt}\n\n"
        
        # Add context
        prompt += f"ANALYSIS CONTEXT:\n{context}\n\n"
        
        # Add scenario details
        if factual and intervention:
            prompt += f"FACTUAL SCENARIO: {factual}\n"
            prompt += f"INTERVENTION: {intervention}\n\n"
        
        if question:
            prompt += f"SPECIFIC QUESTION: {question}\n\n"
        
        # Add previous analyses if available
        if previous_analyses:
            prompt += "PREVIOUS AGENT ANALYSES:\n"
            for i, analysis in enumerate(previous_analyses, 1):
                prompt += f"{i}. {analysis.agent_role.value.upper()} PERSPECTIVE:\n"
                prompt += f"Analysis: {analysis.analysis[:500]}...\n"  # Truncate for brevity
                prompt += f"Key Points: {', '.join(analysis.key_points[:3])}\n"
                prompt += f"Concerns: {', '.join(analysis.concerns[:3])}\n\n"
        
        # Add role-specific instructions
        prompt += self._get_role_specific_instructions()
        
        # Add structured output format
        prompt += self._get_output_format_instructions()
        
        return prompt
    
    def _get_role_specific_instructions(self) -> str:
        """Get role-specific analysis instructions."""
        
        if self.role == AgentRole.DOMAIN_EXPERT:
            return """
As a domain expert, focus on:
1. What mechanisms in this domain could explain the causal relationship?
2. What domain-specific confounders might be present?
3. How do these findings align with established knowledge in the field?
4. What practical considerations affect implementation or interpretation?
5. What domain-specific data or evidence would strengthen the analysis?
"""

        elif self.role == AgentRole.STATISTICIAN:
            return """
As a statistician, focus on:
1. What statistical assumptions are required for causal inference?
2. What are the potential sources of bias and how serious are they?
3. What is the appropriate statistical method for estimation?
4. How strong is the evidence for causality versus correlation?
5. What sensitivity analyses should be conducted?
"""

        elif self.role == AgentRole.SKEPTIC:
            return """
As a skeptic, focus on:
1. What alternative explanations could account for these findings?
2. What assumptions might be violated or questionable?
3. What evidence contradicts or weakens the causal claim?
4. What are the most serious threats to validity?
5. Where might the analysis be oversimplified or overconfident?
"""

        elif self.role == AgentRole.SYNTHESIZER:
            return """
As a synthesizer, focus on:
1. Where do the different perspectives agree and disagree?
2. How can competing viewpoints be reconciled?
3. What is the overall strength of evidence for causality?
4. What are the most important limitations and uncertainties?
5. What actionable recommendations emerge from this analysis?
"""

        else:  # COORDINATOR
            return """
As a coordinator, focus on:
1. Have all important perspectives been considered?
2. Are there gaps in the analysis that need addressing?
3. What additional information or analysis is needed?
4. How should the findings be interpreted and communicated?
5. What are the next steps for this research question?
"""
    
    def _get_output_format_instructions(self) -> str:
        """Get structured output format instructions."""
        return """
Please provide your analysis in the following structured format:

ANALYSIS:
[Your detailed analysis here]

CONFIDENCE LEVEL: [0.0 to 1.0]

KEY POINTS:
- [First key point]
- [Second key point]
- [Third key point]

CONCERNS/LIMITATIONS:
- [First concern]
- [Second concern] 
- [Third concern]

RECOMMENDATIONS:
- [First recommendation]
- [Second recommendation]
- [Third recommendation]
"""
    
    def _parse_response(self, response: str) -> AgentResponse:
        """Parse structured response from the agent."""
        
        # Initialize default values
        analysis = response
        confidence = 0.7
        key_points = []
        concerns = []
        recommendations = []
        
        try:
            # Extract sections using simple text parsing
            lines = response.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('ANALYSIS:'):
                    current_section = 'analysis'
                    analysis = line.replace('ANALYSIS:', '').strip()
                elif line.startswith('CONFIDENCE LEVEL:'):
                    confidence_text = line.replace('CONFIDENCE LEVEL:', '').strip()
                    try:
                        confidence = float(confidence_text)
                    except:
                        confidence = 0.7
                elif line.startswith('KEY POINTS:'):
                    current_section = 'key_points'
                elif line.startswith('CONCERNS') or line.startswith('LIMITATIONS'):
                    current_section = 'concerns'
                elif line.startswith('RECOMMENDATIONS:'):
                    current_section = 'recommendations'
                elif line.startswith('- ') and current_section:
                    item = line[2:].strip()
                    if current_section == 'key_points':
                        key_points.append(item)
                    elif current_section == 'concerns':
                        concerns.append(item)
                    elif current_section == 'recommendations':
                        recommendations.append(item)
                elif current_section == 'analysis' and line:
                    analysis += ' ' + line
        
        except Exception as e:
            self.logger.warning(f"Failed to parse structured response: {e}")
            # Fallback to basic parsing
            key_points = ["Analysis completed"]
            concerns = ["Response parsing incomplete"]
            recommendations = ["Review full response"]
        
        return AgentResponse(
            agent_role=self.role,
            analysis=analysis,
            confidence=max(0.0, min(1.0, confidence)),
            key_points=key_points[:5],  # Limit to 5 items
            concerns=concerns[:5],
            recommendations=recommendations[:5]
        )


class MultiAgentCausalAnalyzer:
    """
    Orchestrates multi-agent collaborative causal analysis.
    
    Coordinates domain experts, statisticians, skeptics, and synthesizers
    to provide comprehensive causal analysis from multiple perspectives.
    """
    
    def __init__(self, 
                 domain: str = "general",
                 llm_client: Optional[BaseLLMClient] = None):
        """
        Initialize multi-agent analyzer.
        
        Args:
            domain: Domain specialization
            llm_client: LLM client (if None, creates default)
        """
        self.domain = domain
        self.llm_client = llm_client or get_llm_client("grok")
        
        self.logger = get_logger("causalllm.llm_agents.multi_agent")
        self.struct_logger = get_structured_logger("multi_agent_analyzer")
        
        # Initialize agents
        self.agents = self._create_agents()
        
        self.logger.info(f"Initialized multi-agent analyzer for {domain} domain with {len(self.agents)} agents")
    
    def _create_agents(self) -> Dict[AgentRole, CausalAgent]:
        """Create specialized agents for collaborative analysis."""
        agents = {}
        
        # Create each type of agent
        for role in [AgentRole.DOMAIN_EXPERT, AgentRole.STATISTICIAN, AgentRole.SKEPTIC, AgentRole.SYNTHESIZER]:
            
            # Adjust temperature by role
            if role == AgentRole.STATISTICIAN:
                temperature = 0.3  # Lower temperature for statistical precision
            elif role == AgentRole.SKEPTIC:
                temperature = 0.6  # Medium temperature for critical thinking
            elif role == AgentRole.SYNTHESIZER:
                temperature = 0.4  # Lower temperature for balanced synthesis
            else:  # DOMAIN_EXPERT
                temperature = 0.7  # Higher temperature for creative insights
            
            agents[role] = CausalAgent(
                role=role,
                llm_client=self.llm_client,
                domain=self.domain,
                temperature=temperature
            )
        
        return agents
    
    async def analyze_counterfactual(self,
                                   context: str,
                                   factual: str,
                                   intervention: str,
                                   include_agents: Optional[List[AgentRole]] = None) -> CollaborativeAnalysis:
        """
        Perform collaborative counterfactual analysis.
        
        Args:
            context: Background context
            factual: Factual scenario
            intervention: Counterfactual intervention
            include_agents: Specific agents to include (if None, uses all)
            
        Returns:
            CollaborativeAnalysis with multi-agent insights
        """
        self.logger.info("Starting collaborative counterfactual analysis")
        
        if include_agents is None:
            include_agents = [AgentRole.DOMAIN_EXPERT, AgentRole.STATISTICIAN, AgentRole.SKEPTIC]
        
        # Phase 1: Individual agent analyses (parallel)
        self.logger.info(f"Phase 1: Individual analyses from {len(include_agents)} agents")
        
        tasks = []
        for role in include_agents:
            if role in self.agents:
                task = self.agents[role].analyze(
                    context=context,
                    factual=factual,
                    intervention=intervention
                )
                tasks.append(task)
        
        # Run analyses in parallel
        agent_responses = await asyncio.gather(*tasks)
        
        # Phase 2: Synthesis
        self.logger.info("Phase 2: Synthesizing multi-agent perspectives")
        
        synthesis_response = await self.agents[AgentRole.SYNTHESIZER].analyze(
            context=context,
            factual=factual,
            intervention=intervention,
            previous_analyses=agent_responses
        )
        
        # Create collaborative analysis
        collaborative_analysis = self._create_collaborative_analysis(
            agent_responses + [synthesis_response],
            synthesis_response
        )
        
        self.struct_logger.log_interaction(
            "collaborative_counterfactual_analysis",
            {
                "domain": self.domain,
                "n_agents": len(include_agents),
                "consensus_points": len(collaborative_analysis.consensus_points),
                "disagreements": len(collaborative_analysis.disagreements),
                "final_confidence": collaborative_analysis.confidence_level
            }
        )
        
        self.logger.info(f"Completed collaborative analysis with {collaborative_analysis.confidence_level:.2f} confidence")
        return collaborative_analysis
    
    async def analyze_treatment_effect(self,
                                     treatment: str,
                                     outcome: str,
                                     context: str,
                                     data_description: Optional[str] = None) -> CollaborativeAnalysis:
        """
        Perform collaborative treatment effect analysis.
        
        Args:
            treatment: Treatment description
            outcome: Outcome description
            context: Study context
            data_description: Optional data description
            
        Returns:
            CollaborativeAnalysis with treatment effect insights
        """
        self.logger.info("Starting collaborative treatment effect analysis")
        
        # Combine context with treatment/outcome info
        full_context = f"{context}\n\nTreatment: {treatment}\nOutcome: {outcome}"
        if data_description:
            full_context += f"\nData: {data_description}"
        
        question = f"What is the causal effect of {treatment} on {outcome}?"
        
        # Phase 1: Individual analyses
        tasks = [
            self.agents[AgentRole.DOMAIN_EXPERT].analyze(context=full_context, question=question),
            self.agents[AgentRole.STATISTICIAN].analyze(context=full_context, question=question),
            self.agents[AgentRole.SKEPTIC].analyze(context=full_context, question=question)
        ]
        
        agent_responses = await asyncio.gather(*tasks)
        
        # Phase 2: Synthesis
        synthesis_response = await self.agents[AgentRole.SYNTHESIZER].analyze(
            context=full_context,
            question=question,
            previous_analyses=agent_responses
        )
        
        collaborative_analysis = self._create_collaborative_analysis(
            agent_responses + [synthesis_response],
            synthesis_response
        )
        
        self.logger.info("Completed collaborative treatment effect analysis")
        return collaborative_analysis
    
    def _create_collaborative_analysis(self,
                                     all_responses: List[AgentResponse],
                                     synthesis: AgentResponse) -> CollaborativeAnalysis:
        """Create final collaborative analysis from agent responses."""
        
        # Extract consensus and disagreement points
        consensus_points = []
        disagreements = []
        
        # Simple consensus detection: points mentioned by multiple agents
        all_points = []
        for response in all_responses:
            all_points.extend(response.key_points)
        
        # Find common themes (simplified approach)
        point_counts = {}
        for point in all_points:
            point_lower = point.lower()
            point_counts[point_lower] = point_counts.get(point_lower, 0) + 1
        
        # Points mentioned by multiple agents are consensus
        for point, count in point_counts.items():
            if count >= 2:
                consensus_points.append(point)
        
        # Disagreements are points with low confidence or conflicting views
        for response in all_responses:
            if response.confidence < 0.6:
                for concern in response.concerns:
                    if concern not in disagreements:
                        disagreements.append(concern)
        
        # Calculate overall confidence (weighted average)
        total_confidence = sum(r.confidence for r in all_responses)
        avg_confidence = total_confidence / len(all_responses) if all_responses else 0.5
        
        # Combine recommendations
        all_recommendations = []
        for response in all_responses:
            all_recommendations.extend(response.recommendations)
        
        # Deduplicate recommendations
        unique_recommendations = list(set(all_recommendations))
        
        # Extract limitations
        limitations = []
        for response in all_responses:
            limitations.extend(response.concerns)
        
        return CollaborativeAnalysis(
            primary_conclusion=synthesis.analysis,
            consensus_points=consensus_points[:10],  # Top 10
            disagreements=disagreements[:5],  # Top 5
            confidence_level=avg_confidence,
            agent_responses=all_responses,
            synthesis_reasoning=synthesis.analysis,
            recommendations=unique_recommendations[:10],
            limitations=list(set(limitations))[:10]
        )
    
    def get_agent_summary(self) -> Dict[str, Any]:
        """Get summary of available agents and their capabilities."""
        return {
            "domain": self.domain,
            "available_agents": [role.value for role in self.agents.keys()],
            "agent_descriptions": {
                role.value: agent._get_system_prompt()[:100] + "..."
                for role, agent in self.agents.items()
            }
        }


# Convenience functions for common multi-agent analysis patterns
async def quick_multi_agent_analysis(context: str,
                                    factual: str,
                                    intervention: str,
                                    domain: str = "general") -> CollaborativeAnalysis:
    """
    Perform quick multi-agent counterfactual analysis.
    
    Args:
        context: Background context
        factual: Factual scenario
        intervention: Counterfactual intervention  
        domain: Domain for specialized analysis
        
    Returns:
        CollaborativeAnalysis with multi-agent insights
    """
    analyzer = MultiAgentCausalAnalyzer(domain=domain)
    return await analyzer.analyze_counterfactual(context, factual, intervention)


async def expert_panel_analysis(treatment: str,
                               outcome: str,
                               context: str,
                               domain: str = "general") -> CollaborativeAnalysis:
    """
    Simulate an expert panel discussion on treatment effects.
    
    Args:
        treatment: Treatment description
        outcome: Outcome description
        context: Study context
        domain: Domain for specialized analysis
        
    Returns:
        CollaborativeAnalysis representing expert panel conclusions
    """
    analyzer = MultiAgentCausalAnalyzer(domain=domain)
    return await analyzer.analyze_treatment_effect(treatment, outcome, context)


def compare_agent_perspectives(responses: List[AgentResponse]) -> Dict[str, Any]:
    """
    Compare and analyze differences between agent perspectives.
    
    Args:
        responses: List of agent responses to compare
        
    Returns:
        Comparison analysis showing agreements and disagreements
    """
    if not responses:
        return {"error": "No responses to compare"}
    
    comparison = {
        "agents": [r.agent_role.value for r in responses],
        "confidence_range": (
            min(r.confidence for r in responses),
            max(r.confidence for r in responses)
        ),
        "avg_confidence": sum(r.confidence for r in responses) / len(responses),
        "common_points": [],
        "unique_perspectives": {},
        "major_disagreements": []
    }
    
    # Find common points
    all_points = [point.lower() for response in responses for point in response.key_points]
    point_counts = {}
    for point in all_points:
        point_counts[point] = point_counts.get(point, 0) + 1
    
    comparison["common_points"] = [
        point for point, count in point_counts.items() 
        if count >= len(responses) // 2
    ]
    
    # Identify unique perspectives
    for response in responses:
        unique_points = [
            point for point in response.key_points 
            if point.lower() not in comparison["common_points"]
        ]
        if unique_points:
            comparison["unique_perspectives"][response.agent_role.value] = unique_points
    
    # Major disagreements (high confidence differences)
    confidences = [r.confidence for r in responses]
    if max(confidences) - min(confidences) > 0.3:
        comparison["major_disagreements"].append(
            f"Confidence levels vary significantly: {min(confidences):.2f} to {max(confidences):.2f}"
        )
    
    return comparison