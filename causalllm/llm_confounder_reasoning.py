"""
LLM-based confounder reasoning and detection system.

This module provides LLM-enhanced identification of potential confounders,
natural language explanation of confounding mechanisms, backdoor criterion
validation, and intelligent suggestions for adjustment strategies.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import asyncio
import json
import itertools
from collections import defaultdict

from causalllm.logging import get_logger


class ConfounderType(Enum):
    """Types of confounding relationships."""
    CLASSIC_CONFOUNDER = "classic_confounder"    # X ← C → Y
    COLLIDER = "collider"                        # X → C ← Y  
    MEDIATOR = "mediator"                        # X → M → Y
    INSTRUMENTAL_VARIABLE = "instrumental"        # I → X, I ⊥ Y|X
    SELECTION_BIAS = "selection_bias"            # Selection affects both X and Y
    TIME_VARYING_CONFOUNDER = "time_varying"     # Confounding changes over time


class ConfounderStrength(Enum):
    """Strength of confounding effect."""
    WEAK = "weak"
    MODERATE = "moderate" 
    STRONG = "strong"
    CRITICAL = "critical"


class AdjustmentStrategy(Enum):
    """Strategies for handling confounders."""
    CONTROL_DIRECTLY = "control_directly"
    STRATIFICATION = "stratification"
    MATCHING = "matching"
    INSTRUMENTAL_VARIABLES = "instrumental_variables"
    REGRESSION_ADJUSTMENT = "regression_adjustment"
    PROPENSITY_SCORES = "propensity_scores"
    DIFFERENCE_IN_DIFFERENCES = "difference_in_differences"
    RANDOMIZATION = "randomization"


@dataclass
class ConfounderCandidate:
    """A potential confounding variable identified by the LLM."""
    
    variable_name: str
    confounder_type: ConfounderType
    strength: ConfounderStrength
    reasoning: str
    mechanism_description: str
    affects_treatment: bool
    affects_outcome: bool
    measurement_available: bool = True
    adjustment_strategies: List[AdjustmentStrategy] = field(default_factory=list)
    evidence_level: str = "theoretical"  # theoretical, empirical, established


@dataclass
class BackdoorAnalysis:
    """Results from backdoor criterion analysis."""
    
    valid_adjustment_sets: List[List[str]]
    minimal_adjustment_sets: List[List[str]]
    invalid_sets: List[Tuple[List[str], str]]  # (set, reason_invalid)
    recommended_adjustment_set: List[str]
    reasoning: str
    assumptions_required: List[str]


@dataclass
class ConfounderAssessment:
    """Comprehensive confounder assessment results."""
    
    identified_confounders: List[ConfounderCandidate]
    colliders_detected: List[ConfounderCandidate] 
    mediators_detected: List[ConfounderCandidate]
    backdoor_analysis: BackdoorAnalysis
    adjustment_recommendations: List[str]
    residual_confounding_risk: str
    sensitivity_analysis_suggestions: List[str]
    data_collection_recommendations: List[str]


class LLMConfounderReasoning:
    """LLM-enhanced confounder identification and reasoning system."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.logger = get_logger("causalllm.llm_confounder_reasoning")
        
        # Domain-specific confounder knowledge
        self.domain_knowledge = {
            "healthcare": {
                "common_confounders": [
                    "age", "gender", "socioeconomic_status", "comorbidities", 
                    "disease_severity", "healthcare_access", "adherence"
                ],
                "mechanisms": {
                    "age": "Age affects both treatment selection and health outcomes",
                    "socioeconomic_status": "SES influences treatment access and health behaviors"
                }
            },
            "education": {
                "common_confounders": [
                    "socioeconomic_background", "prior_achievement", "motivation",
                    "school_quality", "family_support", "student_characteristics"
                ],
                "mechanisms": {
                    "socioeconomic_background": "Family background affects both educational opportunities and outcomes",
                    "prior_achievement": "Previous performance influences both intervention selection and future success"
                }
            },
            "business": {
                "common_confounders": [
                    "market_conditions", "company_size", "industry_sector",
                    "management_quality", "resource_availability", "competitive_position"
                ],
                "mechanisms": {
                    "market_conditions": "Economic conditions affect both business decisions and outcomes",
                    "company_size": "Size influences both strategy choices and performance capacity"
                }
            }
        }
    
    async def identify_potential_confounders(self, 
                                           treatment_variable: str,
                                           outcome_variable: str,
                                           available_variables: Dict[str, str],
                                           domain: str = "general",
                                           context: str = "") -> List[ConfounderCandidate]:
        """
        Use LLM to identify potential confounders based on domain knowledge and causal reasoning.
        
        Args:
            treatment_variable: Name of treatment/exposure variable
            outcome_variable: Name of outcome variable
            available_variables: Dictionary of available variables with descriptions
            domain: Domain context (healthcare, education, business, etc.)
            context: Additional context about the study/analysis
            
        Returns:
            List of potential confounder candidates with reasoning
        """
        self.logger.info("Identifying potential confounders using LLM reasoning")
        
        # Generate LLM prompt for confounder identification
        confounder_candidates = await self._llm_identify_confounders(
            treatment_variable, outcome_variable, available_variables, domain, context
        )
        
        # Enhance with domain-specific knowledge
        enhanced_candidates = self._enhance_with_domain_knowledge(
            confounder_candidates, domain
        )
        
        # Classify confounders by type and strength
        classified_candidates = await self._classify_confounders(
            enhanced_candidates, treatment_variable, outcome_variable, context
        )
        
        self.logger.info(f"Identified {len(classified_candidates)} potential confounders")
        return classified_candidates
    
    async def _llm_identify_confounders(self, treatment: str, outcome: str,
                                      variables: Dict[str, str], domain: str,
                                      context: str) -> List[Dict[str, Any]]:
        """Use LLM to identify confounders through causal reasoning."""
        
        # Create domain-specific guidance
        domain_guidance = ""
        if domain in self.domain_knowledge:
            common_confounders = self.domain_knowledge[domain]["common_confounders"]
            domain_guidance = f"\nCommon confounders in {domain}: {', '.join(common_confounders)}"
        
        prompt = f"""
        You are an expert in causal inference. Identify potential confounding variables 
        that could bias the causal relationship between {treatment} and {outcome}.
        
        TREATMENT VARIABLE: {treatment}
        OUTCOME VARIABLE: {outcome}
        DOMAIN: {domain}
        CONTEXT: {context}
        
        AVAILABLE VARIABLES:
        """
        
        for var, desc in variables.items():
            prompt += f"\n- {var}: {desc}"
        
        prompt += domain_guidance
        
        prompt += f"""
        
        For each potential confounder, determine:
        1. Does it affect the treatment assignment/exposure?
        2. Does it affect the outcome independently of treatment?
        3. What is the mechanism by which it confounds?
        4. How strong is the confounding effect likely to be?
        
        IMPORTANT: Only include variables that satisfy BOTH criteria:
        - Affects treatment selection/assignment
        - Affects outcome independently of treatment
        
        Respond with JSON array:
        [
          {{
            "variable": "variable_name",
            "affects_treatment": true/false,
            "affects_outcome": true/false,  
            "mechanism": "explain the confounding mechanism",
            "strength": "weak/moderate/strong/critical",
            "evidence": "theoretical/empirical/established",
            "reasoning": "detailed reasoning for why this is a confounder"
          }}
        ]
        
        Do not include mediators (X → M → Y) or colliders (X → C ← Y) as confounders.
        Focus on classic confounders (X ← C → Y).
        """
        
        try:
            if hasattr(self.llm_client, 'generate_response'):
                response = await self.llm_client.generate_response(prompt)
            else:
                response = await asyncio.to_thread(self.llm_client.generate, prompt)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                candidates_data = json.loads(json_match.group())
                return candidates_data
            else:
                self.logger.warning("Could not parse confounder identification response")
                return []
                
        except Exception as e:
            self.logger.error(f"LLM confounder identification failed: {e}")
            return []
    
    def _enhance_with_domain_knowledge(self, candidates: List[Dict[str, Any]], 
                                     domain: str) -> List[Dict[str, Any]]:
        """Enhance LLM-identified confounders with domain-specific knowledge."""
        
        if domain not in self.domain_knowledge:
            return candidates
        
        domain_info = self.domain_knowledge[domain]
        
        for candidate in candidates:
            var_name = candidate["variable"]
            
            # Add domain-specific mechanism if available
            if var_name in domain_info.get("mechanisms", {}):
                domain_mechanism = domain_info["mechanisms"][var_name]
                candidate["mechanism"] += f" ({domain_mechanism})"
            
            # Enhance reasoning with domain knowledge
            if var_name in domain_info.get("common_confounders", []):
                candidate["evidence"] = "established"
                candidate["reasoning"] += f" This is a well-established confounder in {domain} research."
        
        return candidates
    
    async def _classify_confounders(self, candidates: List[Dict[str, Any]],
                                  treatment: str, outcome: str, 
                                  context: str) -> List[ConfounderCandidate]:
        """Classify confounders by type and create structured objects."""
        
        classified_candidates = []
        
        for candidate_data in candidates:
            # Determine confounder type
            confounder_type = ConfounderType.CLASSIC_CONFOUNDER  # Default
            
            # Map strength
            strength_mapping = {
                "weak": ConfounderStrength.WEAK,
                "moderate": ConfounderStrength.MODERATE,
                "strong": ConfounderStrength.STRONG,
                "critical": ConfounderStrength.CRITICAL
            }
            strength = strength_mapping.get(
                candidate_data.get("strength", "moderate").lower(),
                ConfounderStrength.MODERATE
            )
            
            # Suggest adjustment strategies
            adjustment_strategies = self._suggest_adjustment_strategies(
                candidate_data, strength
            )
            
            candidate = ConfounderCandidate(
                variable_name=candidate_data["variable"],
                confounder_type=confounder_type,
                strength=strength,
                reasoning=candidate_data.get("reasoning", ""),
                mechanism_description=candidate_data.get("mechanism", ""),
                affects_treatment=candidate_data.get("affects_treatment", True),
                affects_outcome=candidate_data.get("affects_outcome", True),
                adjustment_strategies=adjustment_strategies,
                evidence_level=candidate_data.get("evidence", "theoretical")
            )
            
            classified_candidates.append(candidate)
        
        return classified_candidates
    
    def _suggest_adjustment_strategies(self, candidate_data: Dict[str, Any],
                                     strength: ConfounderStrength) -> List[AdjustmentStrategy]:
        """Suggest appropriate adjustment strategies for a confounder."""
        
        strategies = []
        
        # Always suggest direct control if variable is measured
        strategies.append(AdjustmentStrategy.CONTROL_DIRECTLY)
        strategies.append(AdjustmentStrategy.REGRESSION_ADJUSTMENT)
        
        # For strong confounders, suggest multiple strategies
        if strength in [ConfounderStrength.STRONG, ConfounderStrength.CRITICAL]:
            strategies.extend([
                AdjustmentStrategy.STRATIFICATION,
                AdjustmentStrategy.MATCHING,
                AdjustmentStrategy.PROPENSITY_SCORES
            ])
        
        return strategies[:3]  # Limit to top 3 strategies
    
    async def detect_colliders_and_mediators(self, 
                                           treatment_variable: str,
                                           outcome_variable: str,
                                           available_variables: Dict[str, str],
                                           context: str = "") -> Tuple[List[ConfounderCandidate], List[ConfounderCandidate]]:
        """
        Detect colliders (X → C ← Y) and mediators (X → M → Y) that should NOT be adjusted for.
        
        Returns:
            Tuple of (colliders, mediators)
        """
        self.logger.info("Detecting colliders and mediators using LLM reasoning")
        
        prompt = f"""
        Identify variables that are either COLLIDERS or MEDIATORS in the causal relationship 
        between {treatment_variable} and {outcome_variable}.
        
        TREATMENT: {treatment_variable}
        OUTCOME: {outcome_variable}
        CONTEXT: {context}
        
        AVAILABLE VARIABLES:
        """
        
        for var, desc in available_variables.items():
            if var not in [treatment_variable, outcome_variable]:
                prompt += f"\n- {var}: {desc}"
        
        prompt += f"""
        
        DEFINITIONS:
        - COLLIDER: A variable that is causally affected by BOTH treatment and outcome (X → C ← Y)
        - MEDIATOR: A variable on the causal path from treatment to outcome (X → M → Y)
        
        IMPORTANT: 
        - Adjusting for colliders introduces bias
        - Adjusting for mediators blocks the causal path
        - Both should generally NOT be included in adjustment sets
        
        Respond with JSON:
        {{
          "colliders": [
            {{
              "variable": "variable_name",
              "reasoning": "why this is a collider",
              "mechanism": "X → C ← Y pathway description"
            }}
          ],
          "mediators": [
            {{
              "variable": "variable_name", 
              "reasoning": "why this is a mediator",
              "mechanism": "X → M → Y pathway description"
            }}
          ]
        }}
        """
        
        try:
            if hasattr(self.llm_client, 'generate_response'):
                response = await self.llm_client.generate_response(prompt)
            else:
                response = await asyncio.to_thread(self.llm_client.generate, prompt)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                colliders = []
                for collider_data in data.get("colliders", []):
                    collider = ConfounderCandidate(
                        variable_name=collider_data["variable"],
                        confounder_type=ConfounderType.COLLIDER,
                        strength=ConfounderStrength.CRITICAL,  # Colliders are critical to avoid
                        reasoning=collider_data["reasoning"],
                        mechanism_description=collider_data["mechanism"],
                        affects_treatment=False,  # Colliders are affected by treatment
                        affects_outcome=False,   # Not confounders in traditional sense
                        adjustment_strategies=[]  # Should NOT be adjusted for
                    )
                    colliders.append(collider)
                
                mediators = []
                for mediator_data in data.get("mediators", []):
                    mediator = ConfounderCandidate(
                        variable_name=mediator_data["variable"],
                        confounder_type=ConfounderType.MEDIATOR,
                        strength=ConfounderStrength.CRITICAL,  # Critical to identify
                        reasoning=mediator_data["reasoning"],
                        mechanism_description=mediator_data["mechanism"],
                        affects_treatment=False,
                        affects_outcome=True,
                        adjustment_strategies=[]  # Should NOT be adjusted for in total effect
                    )
                    mediators.append(mediator)
                
                return colliders, mediators
            
        except Exception as e:
            self.logger.error(f"Collider/mediator detection failed: {e}")
        
        return [], []
    
    async def validate_backdoor_criterion(self, 
                                        treatment_variable: str,
                                        outcome_variable: str,
                                        potential_adjustment_sets: List[List[str]],
                                        identified_confounders: List[ConfounderCandidate],
                                        colliders: List[ConfounderCandidate]) -> BackdoorAnalysis:
        """
        Validate potential adjustment sets against the backdoor criterion using LLM reasoning.
        
        Args:
            treatment_variable: Treatment variable name
            outcome_variable: Outcome variable name  
            potential_adjustment_sets: List of potential adjustment sets to validate
            identified_confounders: Known confounders
            colliders: Known colliders to avoid
            
        Returns:
            BackdoorAnalysis with valid/invalid adjustment sets and recommendations
        """
        self.logger.info("Validating backdoor criterion using LLM reasoning")
        
        # Create confounder and collider lists for prompt
        confounder_names = [c.variable_name for c in identified_confounders]
        collider_names = [c.variable_name for c in colliders]
        
        prompt = f"""
        Evaluate these potential adjustment sets for estimating the causal effect of 
        {treatment_variable} on {outcome_variable} using the backdoor criterion.
        
        IDENTIFIED CONFOUNDERS: {confounder_names}
        IDENTIFIED COLLIDERS (DO NOT ADJUST): {collider_names}
        
        BACKDOOR CRITERION REQUIREMENTS:
        1. No variable in the set is a descendant of {treatment_variable}
        2. The set blocks all backdoor paths from {treatment_variable} to {outcome_variable}
        3. Do not include colliders (variables caused by both treatment and outcome)
        
        POTENTIAL ADJUSTMENT SETS:
        """
        
        for i, adj_set in enumerate(potential_adjustment_sets):
            prompt += f"\nSet {i+1}: {adj_set}"
        
        prompt += f"""
        
        For each adjustment set, determine:
        1. Is it VALID according to backdoor criterion?
        2. What backdoor paths does it block/leave open?
        3. Does it include any problematic variables (colliders, mediators)?
        4. Is it minimal (no unnecessary variables)?
        
        Also suggest the BEST adjustment set from the available confounders.
        
        Respond with JSON:
        {{
          "analysis": [
            {{
              "set": [list of variables],
              "valid": true/false,
              "reasoning": "detailed reasoning",
              "problems": ["any issues with this set"],
              "blocked_paths": ["paths this set blocks"],
              "minimal": true/false
            }}
          ],
          "recommended_set": [list of variables],
          "recommendation_reasoning": "why this set is recommended",
          "assumptions_required": ["assumption 1", "assumption 2"]
        }}
        """
        
        try:
            if hasattr(self.llm_client, 'generate_response'):
                response = await self.llm_client.generate_response(prompt)
            else:
                response = await asyncio.to_thread(self.llm_client.generate, prompt)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                valid_sets = []
                minimal_sets = []
                invalid_sets = []
                
                for analysis in data.get("analysis", []):
                    adj_set = analysis["set"]
                    if analysis.get("valid", False):
                        valid_sets.append(adj_set)
                        if analysis.get("minimal", False):
                            minimal_sets.append(adj_set)
                    else:
                        reason = analysis.get("reasoning", "Does not satisfy backdoor criterion")
                        invalid_sets.append((adj_set, reason))
                
                return BackdoorAnalysis(
                    valid_adjustment_sets=valid_sets,
                    minimal_adjustment_sets=minimal_sets,
                    invalid_sets=invalid_sets,
                    recommended_adjustment_set=data.get("recommended_set", []),
                    reasoning=data.get("recommendation_reasoning", ""),
                    assumptions_required=data.get("assumptions_required", [])
                )
            
        except Exception as e:
            self.logger.error(f"Backdoor criterion validation failed: {e}")
        
        # Fallback analysis
        return BackdoorAnalysis(
            valid_adjustment_sets=[confounder_names] if confounder_names else [],
            minimal_adjustment_sets=[],
            invalid_sets=[],
            recommended_adjustment_set=confounder_names[:3],  # Top 3 confounders
            reasoning="LLM analysis failed - using identified confounders",
            assumptions_required=["No unmeasured confounding", "Correct causal structure"]
        )
    
    async def comprehensive_confounder_assessment(self, 
                                                treatment_variable: str,
                                                outcome_variable: str,
                                                available_variables: Dict[str, str],
                                                domain: str = "general",
                                                context: str = "") -> ConfounderAssessment:
        """
        Conduct comprehensive confounder assessment combining all LLM reasoning capabilities.
        
        Args:
            treatment_variable: Treatment/exposure variable
            outcome_variable: Outcome variable
            available_variables: Available variables with descriptions
            domain: Domain context
            context: Study context
            
        Returns:
            Comprehensive confounder assessment
        """
        self.logger.info("Conducting comprehensive confounder assessment")
        
        # Step 1: Identify potential confounders
        confounders = await self.identify_potential_confounders(
            treatment_variable, outcome_variable, available_variables, domain, context
        )
        
        # Step 2: Detect colliders and mediators
        colliders, mediators = await self.detect_colliders_and_mediators(
            treatment_variable, outcome_variable, available_variables, context
        )
        
        # Step 3: Generate potential adjustment sets
        potential_sets = self._generate_adjustment_sets(confounders)
        
        # Step 4: Validate adjustment sets
        backdoor_analysis = await self.validate_backdoor_criterion(
            treatment_variable, outcome_variable, potential_sets, confounders, colliders
        )
        
        # Step 5: Generate final recommendations
        recommendations = await self._generate_adjustment_recommendations(
            confounders, colliders, mediators, backdoor_analysis, domain
        )
        
        # Step 6: Assess residual confounding risk
        residual_risk = await self._assess_residual_confounding_risk(
            confounders, backdoor_analysis.recommended_adjustment_set, context
        )
        
        # Step 7: Suggest sensitivity analyses
        sensitivity_suggestions = self._suggest_sensitivity_analyses(confounders, colliders, mediators)
        
        # Step 8: Data collection recommendations
        data_recommendations = self._suggest_data_collection(confounders, available_variables, domain)
        
        return ConfounderAssessment(
            identified_confounders=confounders,
            colliders_detected=colliders,
            mediators_detected=mediators,
            backdoor_analysis=backdoor_analysis,
            adjustment_recommendations=recommendations,
            residual_confounding_risk=residual_risk,
            sensitivity_analysis_suggestions=sensitivity_suggestions,
            data_collection_recommendations=data_recommendations
        )
    
    def _generate_adjustment_sets(self, confounders: List[ConfounderCandidate]) -> List[List[str]]:
        """Generate potential adjustment sets from identified confounders."""
        
        confounder_names = [c.variable_name for c in confounders]
        
        # Generate sets of different sizes
        adjustment_sets = []
        
        # Individual confounders
        for name in confounder_names:
            adjustment_sets.append([name])
        
        # Pairs of strong confounders
        strong_confounders = [
            c.variable_name for c in confounders 
            if c.strength in [ConfounderStrength.STRONG, ConfounderStrength.CRITICAL]
        ]
        
        if len(strong_confounders) >= 2:
            for pair in itertools.combinations(strong_confounders, 2):
                adjustment_sets.append(list(pair))
        
        # All confounders (might be over-adjustment)
        if len(confounder_names) <= 5:  # Only for manageable sets
            adjustment_sets.append(confounder_names)
        
        return adjustment_sets[:10]  # Limit to 10 sets for analysis
    
    async def _generate_adjustment_recommendations(self, 
                                                 confounders: List[ConfounderCandidate],
                                                 colliders: List[ConfounderCandidate],
                                                 mediators: List[ConfounderCandidate],
                                                 backdoor_analysis: BackdoorAnalysis,
                                                 domain: str) -> List[str]:
        """Generate final adjustment recommendations."""
        
        recommendations = []
        
        # Primary recommendation
        if backdoor_analysis.recommended_adjustment_set:
            vars_str = ", ".join(backdoor_analysis.recommended_adjustment_set)
            recommendations.append(f"Adjust for: {vars_str}")
        
        # Warnings about colliders
        if colliders:
            collider_names = [c.variable_name for c in colliders]
            recommendations.append(f"DO NOT adjust for colliders: {', '.join(collider_names)}")
        
        # Warnings about mediators
        if mediators:
            mediator_names = [c.variable_name for c in mediators]
            recommendations.append(f"Consider NOT adjusting for mediators: {', '.join(mediator_names)} (blocks causal path)")
        
        # Strong confounders that must be addressed
        critical_confounders = [
            c.variable_name for c in confounders 
            if c.strength == ConfounderStrength.CRITICAL
        ]
        if critical_confounders:
            recommendations.append(f"Critical confounders requiring attention: {', '.join(critical_confounders)}")
        
        # Domain-specific recommendations
        if domain == "healthcare":
            recommendations.append("Consider unmeasured confounders: genetic factors, lifestyle, disease severity")
        elif domain == "education":
            recommendations.append("Consider unmeasured confounders: student motivation, family support, peer effects")
        
        return recommendations[:5]  # Top 5 recommendations
    
    async def _assess_residual_confounding_risk(self, 
                                              confounders: List[ConfounderCandidate],
                                              adjustment_set: List[str],
                                              context: str) -> str:
        """Assess risk of residual confounding after adjustment."""
        
        prompt = f"""
        Assess the risk of residual (unmeasured) confounding after adjusting for: {adjustment_set}
        
        IDENTIFIED CONFOUNDERS: {[c.variable_name for c in confounders]}
        CONTEXT: {context}
        
        Consider:
        1. What important confounders might be unmeasured?
        2. How strong could the residual bias be?
        3. What are the key assumptions being made?
        4. How sensitive are results likely to be?
        
        Provide a risk assessment (low/moderate/high) with reasoning.
        """
        
        try:
            if hasattr(self.llm_client, 'generate_response'):
                response = await self.llm_client.generate_response(prompt)
            else:
                response = await asyncio.to_thread(self.llm_client.generate, prompt)
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Residual confounding assessment failed: {e}")
            return "Moderate risk - additional sensitivity analyses recommended"
    
    def _suggest_sensitivity_analyses(self, 
                                    confounders: List[ConfounderCandidate],
                                    colliders: List[ConfounderCandidate],
                                    mediators: List[ConfounderCandidate]) -> List[str]:
        """Suggest sensitivity analyses based on confounder assessment."""
        
        suggestions = []
        
        # E-value analysis for unmeasured confounding
        suggestions.append("Calculate E-values to assess sensitivity to unmeasured confounding")
        
        # Test different adjustment sets
        if len(confounders) > 1:
            suggestions.append("Test robustness across different adjustment sets")
        
        # Negative control analysis
        suggestions.append("Use negative control outcomes to test for residual bias")
        
        # Instrumental variable analysis if possible
        suggestions.append("Consider instrumental variable analysis if valid instruments available")
        
        # Test collider bias
        if colliders:
            suggestions.append("Test for collider bias by comparing results with/without collider adjustment")
        
        return suggestions
    
    def _suggest_data_collection(self, 
                               confounders: List[ConfounderCandidate],
                               available_variables: Dict[str, str],
                               domain: str) -> List[str]:
        """Suggest additional data collection to improve confounding control."""
        
        suggestions = []
        
        # Missing strong confounders
        theoretical_confounders = [
            c.variable_name for c in confounders 
            if c.evidence_level == "theoretical" and c.variable_name not in available_variables
        ]
        
        if theoretical_confounders:
            suggestions.append(f"Collect data on theoretical confounders: {', '.join(theoretical_confounders)}")
        
        # Domain-specific suggestions
        if domain == "healthcare":
            suggestions.append("Consider collecting: genetic markers, lifestyle factors, disease history")
        elif domain == "education":
            suggestions.append("Consider collecting: student motivation measures, family characteristics, peer influences")
        elif domain == "business":
            suggestions.append("Consider collecting: market conditions, competitive factors, organizational characteristics")
        
        # General recommendations
        suggestions.append("Consider longitudinal data to address time-varying confounding")
        suggestions.append("Collect data on potential instrumental variables")
        
        return suggestions[:5]


# Convenience functions
def create_confounder_reasoner(llm_client) -> LLMConfounderReasoning:
    """Create an LLM confounder reasoning system."""
    return LLMConfounderReasoning(llm_client)


async def identify_confounders(treatment_variable: str,
                             outcome_variable: str,
                             available_variables: Dict[str, str],
                             llm_client,
                             domain: str = "general",
                             context: str = "") -> List[ConfounderCandidate]:
    """Quick function to identify confounders using LLM reasoning."""
    
    reasoner = create_confounder_reasoner(llm_client)
    return await reasoner.identify_potential_confounders(
        treatment_variable, outcome_variable, available_variables, domain, context
    )


async def assess_backdoor_criterion(treatment_variable: str,
                                  outcome_variable: str,
                                  adjustment_sets: List[List[str]],
                                  llm_client,
                                  context: str = "") -> BackdoorAnalysis:
    """Quick function to validate adjustment sets using backdoor criterion."""
    
    reasoner = create_confounder_reasoner(llm_client)
    
    # Need to identify confounders and colliders first for proper validation
    # This is a simplified version - in practice, should do full assessment
    return await reasoner.validate_backdoor_criterion(
        treatment_variable, outcome_variable, adjustment_sets, [], []
    )