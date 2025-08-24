"""
LLM Effect Size Interpreter

This module provides natural language interpretation of effect sizes in causal analysis,
translating statistical measures into meaningful, domain-specific explanations that
stakeholders can understand and act upon.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import math
import numpy as np
from scipy import stats

from causalllm.logging import get_logger


class EffectSizeMetric(Enum):
    """Types of effect size metrics."""
    COHENS_D = "cohens_d"
    GLASS_DELTA = "glass_delta"
    HEDGES_G = "hedges_g"
    CORRELATION = "correlation"
    R_SQUARED = "r_squared"
    ODDS_RATIO = "odds_ratio"
    RISK_RATIO = "risk_ratio"
    HAZARD_RATIO = "hazard_ratio"
    MEAN_DIFFERENCE = "mean_difference"
    STANDARDIZED_MEAN_DIFFERENCE = "standardized_mean_difference"
    ETA_SQUARED = "eta_squared"
    PARTIAL_ETA_SQUARED = "partial_eta_squared"
    CRAMERS_V = "cramers_v"
    PHI_COEFFICIENT = "phi_coefficient"


class EffectMagnitude(Enum):
    """Effect magnitude categories."""
    NEGLIGIBLE = "negligible"
    VERY_SMALL = "very_small"
    SMALL = "small"
    SMALL_TO_MEDIUM = "small_to_medium"
    MEDIUM = "medium"
    MEDIUM_TO_LARGE = "medium_to_large"
    LARGE = "large"
    VERY_LARGE = "very_large"
    EXTREMELY_LARGE = "extremely_large"


class ClinicalSignificance(Enum):
    """Clinical/practical significance levels."""
    NOT_CLINICALLY_SIGNIFICANT = "not_clinically_significant"
    POSSIBLY_SIGNIFICANT = "possibly_significant"
    LIKELY_SIGNIFICANT = "likely_significant"
    CLINICALLY_SIGNIFICANT = "clinically_significant"
    HIGHLY_SIGNIFICANT = "highly_significant"


@dataclass
class EffectSizeInterpretation:
    """Comprehensive interpretation of an effect size."""
    
    metric_type: EffectSizeMetric
    raw_value: float
    magnitude_category: EffectMagnitude
    clinical_significance: ClinicalSignificance
    
    # Natural language explanations
    plain_language_summary: str
    technical_interpretation: str
    domain_specific_meaning: str
    practical_implications: str
    
    # Contextualization
    benchmark_comparisons: List[str] = field(default_factory=list)
    confidence_bounds_interpretation: Optional[str] = None
    statistical_significance_note: Optional[str] = None
    
    # Actionability
    decision_recommendations: List[str] = field(default_factory=list)
    follow_up_considerations: List[str] = field(default_factory=list)
    limitations_and_caveats: List[str] = field(default_factory=list)
    
    # Meta information
    interpretation_confidence: float = 0.0
    domain_expertise_level: str = ""
    references_and_standards: List[str] = field(default_factory=list)


@dataclass
class EffectSizeContext:
    """Context for effect size interpretation."""
    
    domain: str
    study_type: str
    intervention_type: str
    outcome_type: str
    sample_size: Optional[int] = None
    population_description: str = ""
    measurement_scale: str = ""
    comparison_baseline: str = ""
    time_horizon: str = ""
    cost_considerations: bool = False


class LLMEffectSizeInterpreter:
    """LLM-enhanced effect size interpretation system."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.logger = get_logger("causalllm.llm_effect_size_interpreter")
        
        # Effect size thresholds by metric type
        self.effect_size_thresholds = {
            EffectSizeMetric.COHENS_D: {
                EffectMagnitude.NEGLIGIBLE: (0.0, 0.15),
                EffectMagnitude.VERY_SMALL: (0.15, 0.2),
                EffectMagnitude.SMALL: (0.2, 0.5),
                EffectMagnitude.MEDIUM: (0.5, 0.8),
                EffectMagnitude.LARGE: (0.8, 1.2),
                EffectMagnitude.VERY_LARGE: (1.2, 2.0),
                EffectMagnitude.EXTREMELY_LARGE: (2.0, float('inf'))
            },
            EffectSizeMetric.CORRELATION: {
                EffectMagnitude.NEGLIGIBLE: (0.0, 0.1),
                EffectMagnitude.VERY_SMALL: (0.1, 0.2),
                EffectMagnitude.SMALL: (0.2, 0.3),
                EffectMagnitude.MEDIUM: (0.3, 0.5),
                EffectMagnitude.LARGE: (0.5, 0.7),
                EffectMagnitude.VERY_LARGE: (0.7, 0.9),
                EffectMagnitude.EXTREMELY_LARGE: (0.9, 1.0)
            },
            EffectSizeMetric.R_SQUARED: {
                EffectMagnitude.NEGLIGIBLE: (0.0, 0.01),
                EffectMagnitude.VERY_SMALL: (0.01, 0.04),
                EffectMagnitude.SMALL: (0.04, 0.09),
                EffectMagnitude.MEDIUM: (0.09, 0.25),
                EffectMagnitude.LARGE: (0.25, 0.64),
                EffectMagnitude.VERY_LARGE: (0.64, 0.81),
                EffectMagnitude.EXTREMELY_LARGE: (0.81, 1.0)
            },
            EffectSizeMetric.ODDS_RATIO: {
                EffectMagnitude.NEGLIGIBLE: (0.9, 1.1),
                EffectMagnitude.VERY_SMALL: (0.8, 1.2),
                EffectMagnitude.SMALL: (0.7, 1.5),
                EffectMagnitude.MEDIUM: (0.5, 2.0),
                EffectMagnitude.LARGE: (0.3, 3.0),
                EffectMagnitude.VERY_LARGE: (0.1, 5.0),
                EffectMagnitude.EXTREMELY_LARGE: (0.0, float('inf'))
            }
        }
        
        # Domain-specific considerations
        self.domain_standards = {
            "healthcare": {
                "minimal_important_difference": {
                    "pain_scale": 1.0,
                    "quality_of_life": 0.3,
                    "blood_pressure": 5.0,
                    "cholesterol": 10.0
                },
                "clinical_significance_thresholds": {
                    "drug_efficacy": {"small": 0.2, "medium": 0.5, "large": 0.8},
                    "behavioral_intervention": {"small": 0.3, "medium": 0.6, "large": 1.0}
                }
            },
            "business": {
                "roi_thresholds": {
                    "marketing_campaign": {"small": 0.05, "medium": 0.15, "large": 0.25},
                    "product_launch": {"small": 0.10, "medium": 0.20, "large": 0.35}
                },
                "conversion_improvements": {
                    "small": 0.01, "medium": 0.05, "large": 0.10
                }
            },
            "education": {
                "learning_gain_thresholds": {
                    "standardized_tests": {"small": 0.2, "medium": 0.5, "large": 0.8},
                    "skill_assessment": {"small": 0.3, "medium": 0.6, "large": 0.9}
                },
                "educational_significance": {
                    "months_of_learning": {"small": 1, "medium": 3, "large": 6}
                }
            }
        }
    
    async def interpret_effect_size(self, 
                                  effect_size_value: float,
                                  metric_type: EffectSizeMetric,
                                  context: EffectSizeContext,
                                  confidence_interval: Optional[Tuple[float, float]] = None,
                                  p_value: Optional[float] = None) -> EffectSizeInterpretation:
        """
        Provide comprehensive natural language interpretation of an effect size.
        
        Args:
            effect_size_value: The numerical effect size
            metric_type: Type of effect size metric
            context: Context for interpretation
            confidence_interval: Optional confidence interval
            p_value: Optional p-value for statistical significance
            
        Returns:
            Comprehensive effect size interpretation
        """
        self.logger.info(f"Interpreting effect size: {effect_size_value} ({metric_type.value})")
        
        # Step 1: Classify magnitude
        magnitude = self._classify_magnitude(effect_size_value, metric_type)
        
        # Step 2: Assess clinical/practical significance
        clinical_significance = self._assess_clinical_significance(
            effect_size_value, metric_type, context
        )
        
        # Step 3: Generate LLM interpretations
        interpretations = await self._generate_llm_interpretations(
            effect_size_value, metric_type, magnitude, clinical_significance, context
        )
        
        # Step 4: Create benchmark comparisons
        benchmarks = self._generate_benchmark_comparisons(
            effect_size_value, metric_type, context
        )
        
        # Step 5: Interpret confidence interval if provided
        ci_interpretation = None
        if confidence_interval:
            ci_interpretation = await self._interpret_confidence_interval(
                confidence_interval, effect_size_value, metric_type, context
            )
        
        # Step 6: Generate recommendations
        recommendations = await self._generate_decision_recommendations(
            effect_size_value, magnitude, clinical_significance, context
        )
        
        # Step 7: Identify limitations
        limitations = self._identify_interpretation_limitations(
            effect_size_value, metric_type, context
        )
        
        interpretation = EffectSizeInterpretation(
            metric_type=metric_type,
            raw_value=effect_size_value,
            magnitude_category=magnitude,
            clinical_significance=clinical_significance,
            plain_language_summary=interpretations["plain_language"],
            technical_interpretation=interpretations["technical"],
            domain_specific_meaning=interpretations["domain_specific"],
            practical_implications=interpretations["practical"],
            benchmark_comparisons=benchmarks,
            confidence_bounds_interpretation=ci_interpretation,
            statistical_significance_note=self._format_statistical_significance_note(p_value),
            decision_recommendations=recommendations["decisions"],
            follow_up_considerations=recommendations["follow_up"],
            limitations_and_caveats=limitations,
            interpretation_confidence=interpretations["confidence"],
            domain_expertise_level=context.domain,
            references_and_standards=self._get_relevant_standards(metric_type, context.domain)
        )
        
        self.logger.info("Effect size interpretation completed")
        return interpretation
    
    def _classify_magnitude(self, effect_size: float, metric_type: EffectSizeMetric) -> EffectMagnitude:
        """Classify effect size magnitude based on established thresholds."""
        
        abs_effect = abs(effect_size)
        thresholds = self.effect_size_thresholds.get(metric_type)
        
        if not thresholds:
            # Use Cohen's d thresholds as default
            thresholds = self.effect_size_thresholds[EffectSizeMetric.COHENS_D]
        
        for magnitude, (lower, upper) in thresholds.items():
            if lower <= abs_effect < upper:
                return magnitude
        
        return EffectMagnitude.EXTREMELY_LARGE
    
    def _assess_clinical_significance(self, 
                                    effect_size: float,
                                    metric_type: EffectSizeMetric,
                                    context: EffectSizeContext) -> ClinicalSignificance:
        """Assess clinical/practical significance based on domain standards."""
        
        domain_standards = self.domain_standards.get(context.domain, {})
        abs_effect = abs(effect_size)
        
        if context.domain == "healthcare":
            # Use clinical significance thresholds
            if context.intervention_type in domain_standards.get("clinical_significance_thresholds", {}):
                thresholds = domain_standards["clinical_significance_thresholds"][context.intervention_type]
                
                if abs_effect >= thresholds.get("large", 0.8):
                    return ClinicalSignificance.HIGHLY_SIGNIFICANT
                elif abs_effect >= thresholds.get("medium", 0.5):
                    return ClinicalSignificance.CLINICALLY_SIGNIFICANT
                elif abs_effect >= thresholds.get("small", 0.2):
                    return ClinicalSignificance.LIKELY_SIGNIFICANT
                elif abs_effect > 0.1:
                    return ClinicalSignificance.POSSIBLY_SIGNIFICANT
        
        elif context.domain == "business":
            # Use business impact thresholds
            if abs_effect >= 0.25:
                return ClinicalSignificance.HIGHLY_SIGNIFICANT
            elif abs_effect >= 0.15:
                return ClinicalSignificance.CLINICALLY_SIGNIFICANT
            elif abs_effect >= 0.05:
                return ClinicalSignificance.LIKELY_SIGNIFICANT
            elif abs_effect > 0.01:
                return ClinicalSignificance.POSSIBLY_SIGNIFICANT
        
        elif context.domain == "education":
            # Use educational significance thresholds
            if abs_effect >= 0.8:
                return ClinicalSignificance.HIGHLY_SIGNIFICANT
            elif abs_effect >= 0.5:
                return ClinicalSignificance.CLINICALLY_SIGNIFICANT
            elif abs_effect >= 0.3:
                return ClinicalSignificance.LIKELY_SIGNIFICANT
            elif abs_effect > 0.2:
                return ClinicalSignificance.POSSIBLY_SIGNIFICANT
        
        # Default assessment
        if abs_effect >= 0.8:
            return ClinicalSignificance.CLINICALLY_SIGNIFICANT
        elif abs_effect >= 0.5:
            return ClinicalSignificance.LIKELY_SIGNIFICANT
        elif abs_effect >= 0.2:
            return ClinicalSignificance.POSSIBLY_SIGNIFICANT
        
        return ClinicalSignificance.NOT_CLINICALLY_SIGNIFICANT
    
    async def _generate_llm_interpretations(self,
                                          effect_size: float,
                                          metric_type: EffectSizeMetric,
                                          magnitude: EffectMagnitude,
                                          clinical_significance: ClinicalSignificance,
                                          context: EffectSizeContext) -> Dict[str, Any]:
        """Generate comprehensive LLM interpretations."""
        
        prompt = f"""
        You are an expert statistician and domain specialist interpreting an effect size for stakeholders.
        
        EFFECT SIZE DETAILS:
        - Value: {effect_size:.4f}
        - Metric: {metric_type.value}
        - Statistical Magnitude: {magnitude.value}
        - Clinical/Practical Significance: {clinical_significance.value}
        
        CONTEXT:
        - Domain: {context.domain}
        - Study Type: {context.study_type}
        - Intervention: {context.intervention_type}
        - Outcome: {context.outcome_type}
        - Sample Size: {context.sample_size or 'Not specified'}
        - Population: {context.population_description}
        - Measurement: {context.measurement_scale}
        - Baseline: {context.comparison_baseline}
        - Time Horizon: {context.time_horizon}
        
        Provide comprehensive interpretations addressing different audiences:
        
        1. PLAIN LANGUAGE SUMMARY (for general audience):
           - What does this effect size mean in everyday terms?
           - How big is the impact in practical terms?
           - Should people care about this result?
        
        2. TECHNICAL INTERPRETATION (for researchers):
           - Statistical significance and magnitude assessment
           - Methodological considerations
           - Comparison to established benchmarks
        
        3. DOMAIN-SPECIFIC MEANING (for domain experts):
           - What does this mean specifically in {context.domain}?
           - How does this compare to typical effects in this field?
           - What are the implications for practice/policy?
        
        4. PRACTICAL IMPLICATIONS (for decision-makers):
           - What actions should be taken based on this result?
           - What are the real-world consequences?
           - Cost-benefit considerations if relevant
        
        Respond in JSON format:
        {{
            "plain_language": "clear explanation for general audience",
            "technical": "detailed statistical interpretation",
            "domain_specific": "field-specific implications and meaning",
            "practical": "actionable insights and implications",
            "confidence": 0.0-1.0,
            "key_insights": ["insight 1", "insight 2", "insight 3"]
        }}
        """
        
        try:
            response = await self.llm_client.generate_response(prompt)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
        except Exception as e:
            self.logger.error(f"LLM interpretation generation failed: {e}")
        
        # Fallback interpretation
        return self._generate_fallback_interpretation(effect_size, metric_type, magnitude, context)
    
    def _generate_fallback_interpretation(self,
                                        effect_size: float,
                                        metric_type: EffectSizeMetric,
                                        magnitude: EffectMagnitude,
                                        context: EffectSizeContext) -> Dict[str, Any]:
        """Generate fallback interpretation when LLM fails."""
        
        return {
            "plain_language": f"The effect size of {effect_size:.3f} represents a {magnitude.value} effect.",
            "technical": f"Effect size ({metric_type.value}) = {effect_size:.4f}, classified as {magnitude.value}",
            "domain_specific": f"In {context.domain}, this effect size is considered {magnitude.value}.",
            "practical": f"This {magnitude.value} effect suggests measurable impact on {context.outcome_type}.",
            "confidence": 0.6,
            "key_insights": [f"{magnitude.value.title()} effect detected", "Requires interpretation in context"]
        }
    
    def _generate_benchmark_comparisons(self,
                                      effect_size: float,
                                      metric_type: EffectSizeMetric,
                                      context: EffectSizeContext) -> List[str]:
        """Generate benchmark comparisons for context."""
        
        comparisons = []
        abs_effect = abs(effect_size)
        
        # Standard benchmarks
        if metric_type == EffectSizeMetric.COHENS_D:
            if abs_effect < 0.2:
                comparisons.append("Smaller than the difference between heights of 15 and 16-year-old girls")
            elif abs_effect < 0.5:
                comparisons.append("Similar to the IQ difference between PhD holders and college graduates")
            elif abs_effect < 0.8:
                comparisons.append("Comparable to the height difference between 13 and 18-year-old girls")
            else:
                comparisons.append("Larger than the IQ difference between college graduates and high school dropouts")
        
        # Domain-specific benchmarks
        if context.domain == "healthcare":
            if context.outcome_type.lower() in ["pain", "pain_scale"]:
                if abs_effect >= 0.5:
                    comparisons.append("Exceeds minimal clinically important difference for pain scales")
                else:
                    comparisons.append("Below typical threshold for clinically meaningful pain reduction")
        
        elif context.domain == "education":
            if abs_effect >= 0.4:
                comparisons.append("Equivalent to approximately 3-4 months of additional learning")
            elif abs_effect >= 0.25:
                comparisons.append("Similar to effects of good tutoring programs")
            else:
                comparisons.append("Smaller than typical classroom intervention effects")
        
        elif context.domain == "business":
            if context.outcome_type.lower() in ["conversion", "sales", "revenue"]:
                if abs_effect >= 0.1:
                    comparisons.append("Represents substantial business impact worth pursuing")
                else:
                    comparisons.append("May not justify implementation costs")
        
        return comparisons
    
    async def _interpret_confidence_interval(self,
                                           ci: Tuple[float, float],
                                           effect_size: float,
                                           metric_type: EffectSizeMetric,
                                           context: EffectSizeContext) -> str:
        """Interpret confidence interval in natural language."""
        
        lower, upper = ci
        ci_width = upper - lower
        
        prompt = f"""
        Interpret this confidence interval for an effect size in simple terms:
        
        Effect Size: {effect_size:.4f} ({metric_type.value})
        95% Confidence Interval: [{lower:.4f}, {upper:.4f}]
        Interval Width: {ci_width:.4f}
        Context: {context.domain} study of {context.intervention_type} on {context.outcome_type}
        
        Explain:
        1. What this confidence interval tells us about precision
        2. Whether the interval includes practically significant values
        3. Implications for decision-making
        4. Any concerns about the width of the interval
        
        Provide a clear, non-technical explanation.
        """
        
        try:
            response = await self.llm_client.generate_response(prompt)
            return response.strip()
        
        except Exception as e:
            self.logger.error(f"Confidence interval interpretation failed: {e}")
            
            # Fallback interpretation
            if lower > 0 and upper > 0:
                return f"We can be 95% confident the true effect is positive, ranging from {lower:.3f} to {upper:.3f}"
            elif lower < 0 and upper < 0:
                return f"We can be 95% confident the true effect is negative, ranging from {lower:.3f} to {upper:.3f}"
            else:
                return f"The confidence interval ({lower:.3f} to {upper:.3f}) includes zero, indicating uncertainty about effect direction"
    
    async def _generate_decision_recommendations(self,
                                               effect_size: float,
                                               magnitude: EffectMagnitude,
                                               clinical_significance: ClinicalSignificance,
                                               context: EffectSizeContext) -> Dict[str, List[str]]:
        """Generate decision and follow-up recommendations."""
        
        prompt = f"""
        Based on this effect size analysis, provide actionable recommendations:
        
        Effect Size: {effect_size:.4f}
        Statistical Magnitude: {magnitude.value}
        Practical Significance: {clinical_significance.value}
        Domain: {context.domain}
        Intervention: {context.intervention_type}
        Outcome: {context.outcome_type}
        
        Generate recommendations in two categories:
        
        1. IMMEDIATE DECISIONS:
           - Should this intervention be implemented?
           - What are the next steps?
           - How confident can decision-makers be?
        
        2. FOLLOW-UP CONSIDERATIONS:
           - What additional evidence is needed?
           - What should be monitored?
           - How can the effect be improved?
        
        Respond as JSON:
        {{
            "decisions": ["decision 1", "decision 2", "decision 3"],
            "follow_up": ["follow-up 1", "follow-up 2", "follow-up 3"]
        }}
        """
        
        try:
            response = await self.llm_client.generate_response(prompt)
            
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
        
        # Fallback recommendations
        decisions = []
        follow_up = []
        
        if clinical_significance in [ClinicalSignificance.CLINICALLY_SIGNIFICANT, ClinicalSignificance.HIGHLY_SIGNIFICANT]:
            decisions.append("Strong evidence supports implementation of this intervention")
            decisions.append("Proceed with broader rollout planning")
            follow_up.append("Monitor real-world effectiveness")
        elif clinical_significance == ClinicalSignificance.LIKELY_SIGNIFICANT:
            decisions.append("Consider implementation with careful monitoring")
            follow_up.append("Collect additional data to confirm effectiveness")
        else:
            decisions.append("Effect size may not justify implementation costs")
            follow_up.append("Consider alternative interventions or modifications")
        
        return {"decisions": decisions, "follow_up": follow_up}
    
    def _identify_interpretation_limitations(self,
                                           effect_size: float,
                                           metric_type: EffectSizeMetric,
                                           context: EffectSizeContext) -> List[str]:
        """Identify limitations and caveats in interpretation."""
        
        limitations = []
        
        # Sample size limitations
        if context.sample_size and context.sample_size < 30:
            limitations.append("Small sample size limits reliability of effect size estimate")
        elif context.sample_size and context.sample_size < 100:
            limitations.append("Moderate sample size - effect size estimate has some uncertainty")
        
        # Metric-specific limitations
        if metric_type == EffectSizeMetric.COHENS_D:
            limitations.append("Assumes approximately normal distributions and equal variances")
        elif metric_type == EffectSizeMetric.ODDS_RATIO:
            limitations.append("May be misleading when baseline risk is very low or very high")
        elif metric_type == EffectSizeMetric.CORRELATION:
            limitations.append("Correlation does not imply causation - consider confounding factors")
        
        # Domain-specific limitations
        if context.domain == "healthcare":
            limitations.append("Individual patient responses may vary significantly from average effect")
            if context.time_horizon:
                limitations.append(f"Effect measured at {context.time_horizon} - long-term effects unknown")
        
        # General limitations
        limitations.append("Effect size depends on measurement precision and study design quality")
        limitations.append("Results may not generalize to different populations or settings")
        
        return limitations
    
    def _format_statistical_significance_note(self, p_value: Optional[float]) -> Optional[str]:
        """Format statistical significance note."""
        
        if p_value is None:
            return None
        
        if p_value < 0.001:
            return "Highly statistically significant (p < 0.001)"
        elif p_value < 0.01:
            return f"Statistically significant (p = {p_value:.3f})"
        elif p_value < 0.05:
            return f"Statistically significant at α = 0.05 level (p = {p_value:.3f})"
        elif p_value < 0.10:
            return f"Marginally significant (p = {p_value:.3f})"
        else:
            return f"Not statistically significant (p = {p_value:.3f})"
    
    def _get_relevant_standards(self, metric_type: EffectSizeMetric, domain: str) -> List[str]:
        """Get relevant standards and references for the interpretation."""
        
        references = []
        
        # Metric-specific references
        if metric_type == EffectSizeMetric.COHENS_D:
            references.append("Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences")
        elif metric_type == EffectSizeMetric.CORRELATION:
            references.append("Cohen, J. (1992). A power primer. Psychological Bulletin, 112(1), 155-159")
        
        # Domain-specific standards
        if domain == "healthcare":
            references.append("Minimal Clinically Important Difference (MCID) guidelines")
        elif domain == "education":
            references.append("What Works Clearinghouse standards")
        elif domain == "business":
            references.append("Industry ROI benchmarks and conversion standards")
        
        return references
    
    async def compare_effect_sizes(self,
                                 effect_sizes: List[Tuple[float, str, EffectSizeContext]],
                                 metric_type: EffectSizeMetric) -> Dict[str, Any]:
        """Compare multiple effect sizes and provide relative interpretation."""
        
        self.logger.info(f"Comparing {len(effect_sizes)} effect sizes")
        
        # Interpret each effect size
        interpretations = []
        for effect_size, label, context in effect_sizes:
            interpretation = await self.interpret_effect_size(
                effect_size, metric_type, context
            )
            interpretations.append((interpretation, label))
        
        # Generate comparison
        comparison_prompt = f"""
        Compare these effect sizes and provide insights about their relative magnitudes and implications:
        
        Effect Sizes:
        """
        
        for interpretation, label in interpretations:
            comparison_prompt += f"""
        {label}: {interpretation.raw_value:.4f} ({interpretation.magnitude_category.value})
        - Domain: {interpretation.domain_expertise_level}
        - Clinical Significance: {interpretation.clinical_significance.value}
        """
        
        comparison_prompt += """
        
        Provide:
        1. Ranking from largest to smallest practical impact
        2. Key insights about the differences
        3. Recommendations for decision-making
        4. Which effects are most actionable
        
        Respond in JSON format with ranking, insights, and recommendations.
        """
        
        try:
            response = await self.llm_client.generate_response(comparison_prompt)
            
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                comparison_result = json.loads(json_match.group())
                
                return {
                    "individual_interpretations": interpretations,
                    "comparison_analysis": comparison_result
                }
        
        except Exception as e:
            self.logger.error(f"Effect size comparison failed: {e}")
        
        return {
            "individual_interpretations": interpretations,
            "comparison_analysis": {"error": "Comparison analysis failed"}
        }


# Convenience functions
def create_effect_size_interpreter(llm_client) -> LLMEffectSizeInterpreter:
    """Create an LLM effect size interpreter."""
    return LLMEffectSizeInterpreter(llm_client)


async def interpret_effect_size_simple(effect_size: float,
                                     metric_type: str,
                                     llm_client,
                                     domain: str = "general",
                                     intervention: str = "treatment",
                                     outcome: str = "outcome") -> EffectSizeInterpretation:
    """Simple function to interpret a single effect size."""
    
    interpreter = create_effect_size_interpreter(llm_client)
    
    context = EffectSizeContext(
        domain=domain,
        study_type="observational",
        intervention_type=intervention,
        outcome_type=outcome
    )
    
    metric_enum = EffectSizeMetric(metric_type)
    
    return await interpreter.interpret_effect_size(effect_size, metric_enum, context)


async def explain_effect_size_to_audience(effect_size: float,
                                        metric_type: str,
                                        audience: str,
                                        llm_client,
                                        domain: str = "general") -> str:
    """Explain effect size tailored to specific audience."""
    
    interpreter = create_effect_size_interpreter(llm_client)
    
    context = EffectSizeContext(
        domain=domain,
        study_type="research",
        intervention_type="intervention",
        outcome_type="outcome"
    )
    
    interpretation = await interpreter.interpret_effect_size(
        effect_size, EffectSizeMetric(metric_type), context
    )
    
    if audience.lower() in ["general", "public", "lay"]:
        return interpretation.plain_language_summary
    elif audience.lower() in ["technical", "researcher", "scientist"]:
        return interpretation.technical_interpretation
    elif audience.lower() in ["decision", "manager", "executive"]:
        return interpretation.practical_implications
    else:
        return interpretation.domain_specific_meaning