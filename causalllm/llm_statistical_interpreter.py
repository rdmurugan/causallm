"""
LLM-based statistical interpretation and validation module.

This module provides LLM-enhanced interpretation of statistical results,
natural language explanations of effect sizes and confidence intervals,
and intelligent detection of statistical issues in causal analysis.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import asyncio
import json
import warnings
from scipy import stats
import math

from causalllm.logging import get_logger


class StatisticalConcern(Enum):
    """Types of statistical concerns that can be detected."""
    MULTIPLE_TESTING = "multiple_testing"
    SMALL_SAMPLE_SIZE = "small_sample_size"
    EFFECT_SIZE_MISMATCH = "effect_size_mismatch"
    WIDE_CONFIDENCE_INTERVALS = "wide_confidence_intervals"
    PUBLICATION_BIAS = "publication_bias"
    OUTLIER_INFLUENCE = "outlier_influence"
    ASSUMPTION_VIOLATION = "assumption_violation"
    POWER_INSUFFICIENCY = "power_insufficiency"


class EffectSizeMagnitude(Enum):
    """Categories for effect size magnitude."""
    NEGLIGIBLE = "negligible"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    VERY_LARGE = "very_large"


@dataclass
class StatisticalResult:
    """Container for statistical analysis results."""
    
    effect_size: float
    confidence_interval: Tuple[float, float]
    p_value: Optional[float] = None
    sample_size: Optional[int] = None
    test_statistic: Optional[float] = None
    degrees_of_freedom: Optional[int] = None
    effect_size_type: str = "unknown"
    test_type: str = "unknown"
    assumptions_met: Optional[Dict[str, bool]] = None


@dataclass
class StatisticalInterpretation:
    """LLM-generated interpretation of statistical results."""
    
    effect_magnitude: EffectSizeMagnitude
    practical_significance: str
    statistical_significance: str
    confidence_interpretation: str
    concerns_detected: List[StatisticalConcern]
    recommendations: List[str]
    plain_language_summary: str
    technical_details: str
    context_specific_insights: List[str]
    follow_up_analyses: List[str]
    limitations: List[str]


@dataclass
class PowerAnalysisResult:
    """Results from LLM-enhanced power analysis."""
    
    observed_power: float
    required_sample_size: int
    minimum_detectable_effect: float
    power_interpretation: str
    sample_size_recommendations: List[str]
    design_suggestions: List[str]


@dataclass
class SensitivityAnalysisResult:
    """Results from LLM-guided sensitivity analysis."""
    
    robustness_assessment: str
    critical_assumptions: List[str]
    sensitivity_to_confounders: str
    alternative_explanations: List[str]
    robustness_recommendations: List[str]


class LLMStatisticalInterpreter:
    """LLM-enhanced statistical interpretation system."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.logger = get_logger("causalllm.llm_statistical_interpreter")
        
        # Statistical interpretation knowledge base
        self.effect_size_thresholds = {
            "cohen_d": {"small": 0.2, "medium": 0.5, "large": 0.8},
            "correlation": {"small": 0.1, "medium": 0.3, "large": 0.5},
            "eta_squared": {"small": 0.01, "medium": 0.06, "large": 0.14},
            "odds_ratio": {"small": 1.5, "medium": 2.5, "large": 4.0}
        }
        
        self.statistical_tests_info = {
            "t_test": {
                "assumptions": ["normality", "independence", "equal_variances"],
                "effect_size": "cohen_d",
                "interpretation_context": "mean differences"
            },
            "correlation": {
                "assumptions": ["linearity", "normality", "homoscedasticity"],
                "effect_size": "correlation_coefficient", 
                "interpretation_context": "relationship strength"
            },
            "regression": {
                "assumptions": ["linearity", "independence", "homoscedasticity", "normality"],
                "effect_size": "R_squared",
                "interpretation_context": "variance explained"
            }
        }
    
    async def interpret_statistical_results(self, 
                                          results: StatisticalResult,
                                          context: str = "",
                                          domain: str = "general") -> StatisticalInterpretation:
        """
        Generate comprehensive LLM-based interpretation of statistical results.
        
        Args:
            results: Statistical results to interpret
            context: Context of the analysis
            domain: Domain-specific context (healthcare, business, etc.)
            
        Returns:
            Comprehensive statistical interpretation
        """
        self.logger.info("Generating LLM-based statistical interpretation")
        
        # Step 1: Classify effect size magnitude
        effect_magnitude = self._classify_effect_size(results.effect_size, results.effect_size_type)
        
        # Step 2: Detect statistical concerns
        concerns = await self._detect_statistical_concerns(results, context)
        
        # Step 3: Generate LLM interpretation
        llm_interpretation = await self._generate_llm_interpretation(
            results, effect_magnitude, concerns, context, domain
        )
        
        # Step 4: Add technical details
        technical_details = self._generate_technical_summary(results)
        
        # Step 5: Generate recommendations
        recommendations = await self._generate_recommendations(results, concerns, domain)
        
        interpretation = StatisticalInterpretation(
            effect_magnitude=effect_magnitude,
            practical_significance=llm_interpretation.get("practical_significance", ""),
            statistical_significance=llm_interpretation.get("statistical_significance", ""),
            confidence_interpretation=llm_interpretation.get("confidence_interpretation", ""),
            concerns_detected=concerns,
            recommendations=recommendations,
            plain_language_summary=llm_interpretation.get("plain_language_summary", ""),
            technical_details=technical_details,
            context_specific_insights=llm_interpretation.get("context_insights", []),
            follow_up_analyses=llm_interpretation.get("follow_up_analyses", []),
            limitations=llm_interpretation.get("limitations", [])
        )
        
        self.logger.info("Statistical interpretation completed")
        return interpretation
    
    def _classify_effect_size(self, effect_size: float, effect_type: str) -> EffectSizeMagnitude:
        """Classify effect size magnitude using established thresholds."""
        
        abs_effect = abs(effect_size)
        
        # Get thresholds for the effect type
        thresholds = self.effect_size_thresholds.get(effect_type, self.effect_size_thresholds["cohen_d"])
        
        if abs_effect < thresholds["small"]:
            return EffectSizeMagnitude.NEGLIGIBLE
        elif abs_effect < thresholds["medium"]:
            return EffectSizeMagnitude.SMALL
        elif abs_effect < thresholds["large"]:
            return EffectSizeMagnitude.MEDIUM
        elif abs_effect < thresholds["large"] * 1.5:
            return EffectSizeMagnitude.LARGE
        else:
            return EffectSizeMagnitude.VERY_LARGE
    
    async def _detect_statistical_concerns(self, results: StatisticalResult, 
                                         context: str) -> List[StatisticalConcern]:
        """Detect potential statistical concerns in the results."""
        
        concerns = []
        
        # Small sample size
        if results.sample_size and results.sample_size < 30:
            concerns.append(StatisticalConcern.SMALL_SAMPLE_SIZE)
        
        # Wide confidence intervals
        if results.confidence_interval:
            ci_width = results.confidence_interval[1] - results.confidence_interval[0]
            effect_size = abs(results.effect_size)
            if effect_size > 0 and (ci_width / effect_size) > 2:
                concerns.append(StatisticalConcern.WIDE_CONFIDENCE_INTERVALS)
        
        # Multiple testing (detected through context analysis)
        if "multiple" in context.lower() or "several" in context.lower():
            concerns.append(StatisticalConcern.MULTIPLE_TESTING)
        
        # Effect size and p-value mismatch
        if results.p_value:
            if results.p_value < 0.05 and abs(results.effect_size) < 0.1:
                concerns.append(StatisticalConcern.EFFECT_SIZE_MISMATCH)
        
        # Power insufficiency (rough heuristic)
        if results.sample_size and results.sample_size < 50 and abs(results.effect_size) < 0.5:
            concerns.append(StatisticalConcern.POWER_INSUFFICIENCY)
        
        return concerns
    
    async def _generate_llm_interpretation(self, results: StatisticalResult,
                                         effect_magnitude: EffectSizeMagnitude,
                                         concerns: List[StatisticalConcern],
                                         context: str, domain: str) -> Dict[str, Any]:
        """Generate LLM-based interpretation of the results."""
        
        prompt = f"""
        You are an expert statistician providing interpretation of causal analysis results.
        
        STATISTICAL RESULTS:
        - Effect size: {results.effect_size:.4f} ({results.effect_size_type})
        - Confidence interval: [{results.confidence_interval[0]:.4f}, {results.confidence_interval[1]:.4f}]
        - P-value: {results.p_value if results.p_value else 'Not provided'}
        - Sample size: {results.sample_size if results.sample_size else 'Not provided'}
        - Test type: {results.test_type}
        - Effect magnitude classification: {effect_magnitude.value}
        
        CONTEXT: {context}
        DOMAIN: {domain}
        
        DETECTED CONCERNS: {[concern.value for concern in concerns]}
        
        Provide a comprehensive interpretation addressing:
        
        1. PRACTICAL SIGNIFICANCE: What does this effect size mean in real-world terms for this domain?
        2. STATISTICAL SIGNIFICANCE: Interpret the p-value and confidence interval
        3. CONFIDENCE INTERPRETATION: What do the confidence intervals tell us?
        4. PLAIN LANGUAGE SUMMARY: Explain the results in simple terms
        5. CONTEXT INSIGHTS: Domain-specific implications
        6. FOLLOW-UP ANALYSES: What additional analyses would be helpful?
        7. LIMITATIONS: What are the key limitations to consider?
        
        Format your response as JSON:
        {{
            "practical_significance": "explanation of real-world importance",
            "statistical_significance": "interpretation of p-value and significance",
            "confidence_interpretation": "what confidence intervals mean",
            "plain_language_summary": "simple explanation for non-statisticians",
            "context_insights": ["domain-specific insight 1", "insight 2"],
            "follow_up_analyses": ["suggested analysis 1", "analysis 2"],
            "limitations": ["limitation 1", "limitation 2"]
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
                return json.loads(json_match.group())
            else:
                self.logger.warning("Could not parse LLM interpretation response")
                return self._generate_fallback_interpretation(results, effect_magnitude)
                
        except Exception as e:
            self.logger.error(f"LLM interpretation failed: {e}")
            return self._generate_fallback_interpretation(results, effect_magnitude)
    
    def _generate_fallback_interpretation(self, results: StatisticalResult,
                                        effect_magnitude: EffectSizeMagnitude) -> Dict[str, Any]:
        """Generate fallback interpretation when LLM fails."""
        
        return {
            "practical_significance": f"Effect size of {results.effect_size:.3f} is {effect_magnitude.value} in magnitude",
            "statistical_significance": f"P-value of {results.p_value if results.p_value else 'unknown'} indicates statistical significance" if results.p_value and results.p_value < 0.05 else "Results may not be statistically significant",
            "confidence_interpretation": f"95% confidence interval: [{results.confidence_interval[0]:.3f}, {results.confidence_interval[1]:.3f}]",
            "plain_language_summary": f"The analysis shows a {effect_magnitude.value} effect",
            "context_insights": ["Further domain-specific analysis recommended"],
            "follow_up_analyses": ["Replication study", "Sensitivity analysis"],
            "limitations": ["Limited interpretation due to missing context"]
        }
    
    def _generate_technical_summary(self, results: StatisticalResult) -> str:
        """Generate technical summary of statistical results."""
        
        summary_parts = [
            f"Effect size: {results.effect_size:.4f} ({results.effect_size_type})",
            f"95% CI: [{results.confidence_interval[0]:.4f}, {results.confidence_interval[1]:.4f}]"
        ]
        
        if results.p_value:
            summary_parts.append(f"p = {results.p_value:.4f}")
        
        if results.sample_size:
            summary_parts.append(f"n = {results.sample_size}")
        
        if results.test_statistic and results.degrees_of_freedom:
            summary_parts.append(f"t({results.degrees_of_freedom}) = {results.test_statistic:.3f}")
        
        return "; ".join(summary_parts)
    
    async def _generate_recommendations(self, results: StatisticalResult,
                                      concerns: List[StatisticalConcern],
                                      domain: str) -> List[str]:
        """Generate recommendations based on results and concerns."""
        
        recommendations = []
        
        # Address specific concerns
        for concern in concerns:
            if concern == StatisticalConcern.SMALL_SAMPLE_SIZE:
                recommendations.append("Consider increasing sample size or using appropriate small-sample methods")
            elif concern == StatisticalConcern.MULTIPLE_TESTING:
                recommendations.append("Apply multiple testing corrections (e.g., Bonferroni, FDR)")
            elif concern == StatisticalConcern.WIDE_CONFIDENCE_INTERVALS:
                recommendations.append("Precision could be improved with larger sample size or reduced measurement error")
            elif concern == StatisticalConcern.POWER_INSUFFICIENCY:
                recommendations.append("Conduct power analysis to determine adequate sample size")
            elif concern == StatisticalConcern.EFFECT_SIZE_MISMATCH:
                recommendations.append("Consider practical significance alongside statistical significance")
        
        # Domain-specific recommendations
        if domain == "healthcare":
            recommendations.append("Consider clinical significance thresholds for this intervention")
        elif domain == "business":
            recommendations.append("Evaluate economic significance and cost-benefit implications")
        elif domain == "education":
            recommendations.append("Assess educational significance and long-term learning outcomes")
        
        # General recommendations
        if abs(results.effect_size) < 0.2:
            recommendations.append("Small effect size may require replication or meta-analysis")
        
        if not recommendations:
            recommendations.append("Results appear robust - consider replication to confirm findings")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    async def conduct_power_analysis(self, effect_size: float,
                                   alpha: float = 0.05,
                                   desired_power: float = 0.8,
                                   current_n: Optional[int] = None) -> PowerAnalysisResult:
        """Conduct LLM-enhanced power analysis."""
        
        self.logger.info("Conducting LLM-enhanced power analysis")
        
        # Calculate required sample size (simplified)
        # For two-sample t-test: n ≈ (z_α/2 + z_β)² × 2σ² / δ²
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(desired_power)
        
        # Simplified calculation assuming σ = 1
        required_n = math.ceil(2 * ((z_alpha + z_beta) / effect_size) ** 2)
        
        # Calculate observed power if current n provided
        observed_power = 0.0
        if current_n:
            ncp = effect_size * math.sqrt(current_n / 2)  # Non-centrality parameter
            observed_power = 1 - stats.norm.cdf(z_alpha - ncp)
        
        # Calculate minimum detectable effect
        if current_n:
            min_detectable_effect = (z_alpha + z_beta) * math.sqrt(2 / current_n)
        else:
            min_detectable_effect = effect_size
        
        # Generate LLM interpretation
        power_interpretation = await self._interpret_power_analysis(
            effect_size, alpha, desired_power, required_n, observed_power, current_n
        )
        
        return PowerAnalysisResult(
            observed_power=observed_power,
            required_sample_size=required_n,
            minimum_detectable_effect=min_detectable_effect,
            power_interpretation=power_interpretation,
            sample_size_recommendations=[],  # Will be filled by LLM
            design_suggestions=[]  # Will be filled by LLM
        )
    
    async def _interpret_power_analysis(self, effect_size: float, alpha: float,
                                      desired_power: float, required_n: int,
                                      observed_power: float, current_n: Optional[int]) -> str:
        """Generate LLM interpretation of power analysis."""
        
        prompt = f"""
        Interpret this power analysis for a causal study:
        
        - Target effect size: {effect_size:.3f}
        - Significance level (α): {alpha}
        - Desired power: {desired_power}
        - Required sample size: {required_n}
        - Current sample size: {current_n if current_n else 'Not specified'}
        - Observed power: {observed_power:.3f}
        
        Provide a clear explanation of what these results mean for the study design,
        including practical implications for data collection and result interpretation.
        Focus on actionable insights for researchers.
        """
        
        try:
            if hasattr(self.llm_client, 'generate_response'):
                response = await self.llm_client.generate_response(prompt)
            else:
                response = await asyncio.to_thread(self.llm_client.generate, prompt)
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Power analysis interpretation failed: {e}")
            return f"Required sample size: {required_n}. Current power: {observed_power:.3f}"
    
    async def assess_sensitivity(self, results: StatisticalResult,
                               potential_confounders: List[str],
                               context: str) -> SensitivityAnalysisResult:
        """Conduct LLM-guided sensitivity analysis."""
        
        self.logger.info("Conducting LLM-guided sensitivity analysis")
        
        prompt = f"""
        Assess the sensitivity and robustness of this causal analysis:
        
        RESULTS:
        - Effect size: {results.effect_size:.4f}
        - Confidence interval: [{results.confidence_interval[0]:.4f}, {results.confidence_interval[1]:.4f}]
        - Sample size: {results.sample_size}
        
        CONTEXT: {context}
        
        POTENTIAL CONFOUNDERS: {potential_confounders}
        
        Provide a sensitivity analysis addressing:
        1. How robust are these results to unmeasured confounding?
        2. What are the critical assumptions?
        3. How sensitive is the effect to potential confounders?
        4. What alternative explanations should be considered?
        5. What would strengthen confidence in these results?
        
        Format as JSON:
        {{
            "robustness_assessment": "overall assessment of result robustness",
            "critical_assumptions": ["assumption 1", "assumption 2"],
            "sensitivity_to_confounders": "assessment of confounder sensitivity",
            "alternative_explanations": ["explanation 1", "explanation 2"],
            "robustness_recommendations": ["recommendation 1", "recommendation 2"]
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
                analysis_data = json.loads(json_match.group())
                
                return SensitivityAnalysisResult(
                    robustness_assessment=analysis_data.get("robustness_assessment", ""),
                    critical_assumptions=analysis_data.get("critical_assumptions", []),
                    sensitivity_to_confounders=analysis_data.get("sensitivity_to_confounders", ""),
                    alternative_explanations=analysis_data.get("alternative_explanations", []),
                    robustness_recommendations=analysis_data.get("robustness_recommendations", [])
                )
            
        except Exception as e:
            self.logger.error(f"Sensitivity analysis failed: {e}")
        
        # Fallback
        return SensitivityAnalysisResult(
            robustness_assessment="Sensitivity analysis could not be completed",
            critical_assumptions=["Assume no unmeasured confounding"],
            sensitivity_to_confounders="Unknown - requires manual assessment",
            alternative_explanations=["Alternative causal mechanisms possible"],
            robustness_recommendations=["Conduct additional robustness checks"]
        )
    
    async def explain_statistical_concepts(self, concept: str, 
                                         context: str = "") -> str:
        """Provide LLM explanation of statistical concepts in context."""
        
        prompt = f"""
        Explain the statistical concept "{concept}" in the context of causal analysis.
        
        Context: {context}
        
        Provide a clear explanation that:
        1. Defines the concept in plain language
        2. Explains why it matters for causal inference
        3. Gives a practical example
        4. Notes common misconceptions
        5. Suggests when to be concerned about it
        
        Make it accessible to someone with basic statistical knowledge.
        """
        
        try:
            if hasattr(self.llm_client, 'generate_response'):
                response = await self.llm_client.generate_response(prompt)
            else:
                response = await asyncio.to_thread(self.llm_client.generate, prompt)
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Statistical concept explanation failed: {e}")
            return f"Could not generate explanation for {concept}. Please consult statistical references."


# Convenience functions
def create_statistical_interpreter(llm_client) -> LLMStatisticalInterpreter:
    """Create an LLM statistical interpreter."""
    return LLMStatisticalInterpreter(llm_client)


async def interpret_effect_size(effect_size: float,
                              confidence_interval: Tuple[float, float],
                              llm_client,
                              p_value: Optional[float] = None,
                              sample_size: Optional[int] = None,
                              context: str = "",
                              domain: str = "general") -> StatisticalInterpretation:
    """Quick function to interpret statistical results with LLM."""
    
    interpreter = create_statistical_interpreter(llm_client)
    
    results = StatisticalResult(
        effect_size=effect_size,
        confidence_interval=confidence_interval,
        p_value=p_value,
        sample_size=sample_size,
        effect_size_type="cohen_d"  # Default assumption
    )
    
    return await interpreter.interpret_statistical_results(results, context, domain)


async def explain_statistical_significance(p_value: float,
                                         effect_size: float,
                                         llm_client,
                                         context: str = "") -> str:
    """Quick function to explain statistical vs practical significance."""
    
    interpreter = create_statistical_interpreter(llm_client)
    
    prompt = f"""
    Explain the difference between statistical and practical significance for these results:
    - P-value: {p_value}
    - Effect size: {effect_size}
    Context: {context}
    
    Help the reader understand what these numbers actually mean for their situation.
    """
    
    try:
        if hasattr(llm_client, 'generate_response'):
            response = await llm_client.generate_response(prompt)
        else:
            response = await asyncio.to_thread(llm_client.generate, prompt)
        
        return response.strip()
        
    except Exception as e:
        return f"P-value of {p_value} with effect size {effect_size} - consult statistical references for interpretation."