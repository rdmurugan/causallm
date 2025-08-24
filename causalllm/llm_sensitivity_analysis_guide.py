"""
LLM Sensitivity Analysis Guide - Automated sensitivity analysis recommendations and interpretation.

This module provides intelligent guidance for conducting sensitivity analyses in causal inference,
helping researchers understand the robustness of their causal conclusions to potential violations
of key assumptions.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class SensitivityTestType(Enum):
    """Types of sensitivity tests available."""
    UNOBSERVED_CONFOUNDING = "unobserved_confounding"
    MEASUREMENT_ERROR = "measurement_error"
    SELECTION_BIAS = "selection_bias"
    MODEL_SPECIFICATION = "model_specification"
    MISSING_DATA = "missing_data"
    EXTERNAL_VALIDITY = "external_validity"
    TEMPORAL_CONFOUNDING = "temporal_confounding"
    INSTRUMENTAL_VARIABLE = "instrumental_variable"


class SensitivityPriority(Enum):
    """Priority levels for sensitivity tests."""
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


class AnalysisContext(Enum):
    """Context for the causal analysis."""
    RANDOMIZED_TRIAL = "randomized_trial"
    OBSERVATIONAL = "observational"
    QUASI_EXPERIMENTAL = "quasi_experimental"
    NATURAL_EXPERIMENT = "natural_experiment"
    INSTRUMENTAL_VARIABLE = "instrumental_variable"


@dataclass
class SensitivityTest:
    """Represents a specific sensitivity test recommendation."""
    test_type: SensitivityTestType
    priority: SensitivityPriority
    method_name: str
    description: str
    rationale: str
    implementation_steps: List[str]
    interpretation_guide: str
    threshold_values: Optional[Dict[str, float]] = None
    required_parameters: Optional[Dict[str, Any]] = None
    software_recommendations: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)


@dataclass
class SensitivityAnalysisResult:
    """Results from conducting a sensitivity test."""
    test_type: SensitivityTestType
    test_name: str
    parameter_values: Dict[str, float]
    robustness_metrics: Dict[str, float]
    critical_threshold: Optional[float]
    conclusion: str
    robustness_level: str  # "robust", "moderately_robust", "sensitive"
    detailed_interpretation: str


@dataclass
class SensitivityAnalysisReport:
    """Comprehensive sensitivity analysis report."""
    study_context: str
    analysis_summary: str
    recommended_tests: List[SensitivityTest]
    conducted_tests: List[SensitivityAnalysisResult] = field(default_factory=list)
    overall_robustness: str = ""
    key_vulnerabilities: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence_assessment: str = ""


class LLMSensitivityAnalysisGuide:
    """
    LLM-powered guide for sensitivity analysis in causal inference.
    
    Provides intelligent recommendations for which sensitivity tests to conduct,
    how to interpret results, and what thresholds to use based on the specific
    research context and domain.
    """
    
    def __init__(self, llm_client: Any, domain: str = "general"):
        """
        Initialize the sensitivity analysis guide.
        
        Args:
            llm_client: LLM client for generating recommendations
            domain: Research domain (healthcare, economics, education, etc.)
        """
        self.llm_client = llm_client
        self.domain = domain
        self.logger = logging.getLogger(__name__)
        
        # Domain-specific sensitivity test priorities
        self.domain_priorities = {
            "healthcare": {
                SensitivityTestType.UNOBSERVED_CONFOUNDING: SensitivityPriority.CRITICAL,
                SensitivityTestType.SELECTION_BIAS: SensitivityPriority.CRITICAL,
                SensitivityTestType.MEASUREMENT_ERROR: SensitivityPriority.HIGH,
                SensitivityTestType.MISSING_DATA: SensitivityPriority.HIGH,
                SensitivityTestType.EXTERNAL_VALIDITY: SensitivityPriority.MODERATE,
            },
            "economics": {
                SensitivityTestType.UNOBSERVED_CONFOUNDING: SensitivityPriority.CRITICAL,
                SensitivityTestType.INSTRUMENTAL_VARIABLE: SensitivityPriority.CRITICAL,
                SensitivityTestType.MODEL_SPECIFICATION: SensitivityPriority.HIGH,
                SensitivityTestType.SELECTION_BIAS: SensitivityPriority.HIGH,
                SensitivityTestType.EXTERNAL_VALIDITY: SensitivityPriority.HIGH,
            },
            "education": {
                SensitivityTestType.SELECTION_BIAS: SensitivityPriority.CRITICAL,
                SensitivityTestType.UNOBSERVED_CONFOUNDING: SensitivityPriority.HIGH,
                SensitivityTestType.TEMPORAL_CONFOUNDING: SensitivityPriority.HIGH,
                SensitivityTestType.MISSING_DATA: SensitivityPriority.MODERATE,
                SensitivityTestType.EXTERNAL_VALIDITY: SensitivityPriority.MODERATE,
            },
            "general": {
                SensitivityTestType.UNOBSERVED_CONFOUNDING: SensitivityPriority.HIGH,
                SensitivityTestType.SELECTION_BIAS: SensitivityPriority.HIGH,
                SensitivityTestType.MODEL_SPECIFICATION: SensitivityPriority.MODERATE,
                SensitivityTestType.MEASUREMENT_ERROR: SensitivityPriority.MODERATE,
                SensitivityTestType.MISSING_DATA: SensitivityPriority.MODERATE,
            }
        }
    
    async def generate_sensitivity_analysis_plan(
        self,
        treatment_variable: str,
        outcome_variable: str,
        observed_confounders: List[str],
        data: Optional[pd.DataFrame] = None,
        analysis_context: AnalysisContext = AnalysisContext.OBSERVATIONAL,
        research_question: Optional[str] = None,
        study_design_description: Optional[str] = None
    ) -> SensitivityAnalysisReport:
        """
        Generate a comprehensive sensitivity analysis plan with LLM guidance.
        
        Args:
            treatment_variable: Name of the treatment/exposure variable
            outcome_variable: Name of the outcome variable
            observed_confounders: List of observed confounding variables
            data: Optional dataset for context-specific recommendations
            analysis_context: Type of study design
            research_question: The main research question
            study_design_description: Description of the study design
            
        Returns:
            SensitivityAnalysisReport with recommended tests and guidance
        """
        try:
            # Analyze study characteristics
            study_characteristics = self._analyze_study_characteristics(
                treatment_variable, outcome_variable, observed_confounders,
                data, analysis_context, study_design_description
            )
            
            # Generate LLM-powered recommendations
            llm_recommendations = await self._get_llm_sensitivity_recommendations(
                study_characteristics, research_question
            )
            
            # Create recommended sensitivity tests
            recommended_tests = self._create_recommended_tests(
                study_characteristics, llm_recommendations
            )
            
            # Generate overall analysis summary
            analysis_summary = await self._generate_analysis_summary(
                study_characteristics, recommended_tests
            )
            
            report = SensitivityAnalysisReport(
                study_context=study_characteristics,
                analysis_summary=analysis_summary,
                recommended_tests=recommended_tests
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating sensitivity analysis plan: {str(e)}")
            raise
    
    async def interpret_sensitivity_results(
        self,
        sensitivity_results: List[SensitivityAnalysisResult],
        original_effect_estimate: float,
        confidence_interval: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Interpret sensitivity analysis results with LLM guidance.
        
        Args:
            sensitivity_results: Results from conducted sensitivity tests
            original_effect_estimate: Original causal effect estimate
            confidence_interval: Confidence interval for original estimate
            
        Returns:
            Dictionary with interpretation, robustness assessment, and recommendations
        """
        try:
            # Analyze robustness patterns
            robustness_analysis = self._analyze_robustness_patterns(
                sensitivity_results, original_effect_estimate
            )
            
            # Generate LLM interpretation
            llm_interpretation = await self._get_llm_interpretation(
                sensitivity_results, original_effect_estimate, 
                confidence_interval, robustness_analysis
            )
            
            # Create overall assessment
            overall_assessment = self._create_overall_assessment(
                sensitivity_results, robustness_analysis, llm_interpretation
            )
            
            return {
                "robustness_analysis": robustness_analysis,
                "llm_interpretation": llm_interpretation,
                "overall_assessment": overall_assessment,
                "confidence_level": self._determine_confidence_level(sensitivity_results),
                "key_vulnerabilities": self._identify_key_vulnerabilities(sensitivity_results),
                "recommendations": self._generate_robustness_recommendations(
                    sensitivity_results, robustness_analysis
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error interpreting sensitivity results: {str(e)}")
            raise
    
    async def suggest_additional_tests(
        self,
        conducted_tests: List[SensitivityAnalysisResult],
        study_characteristics: str,
        current_robustness: str
    ) -> List[SensitivityTest]:
        """
        Suggest additional sensitivity tests based on current results.
        
        Args:
            conducted_tests: Already conducted sensitivity tests
            study_characteristics: Description of the study
            current_robustness: Current robustness assessment
            
        Returns:
            List of additional recommended tests
        """
        try:
            # Identify gaps in current testing
            conducted_types = {test.test_type for test in conducted_tests}
            
            # Get LLM recommendations for additional tests
            llm_prompt = f"""
            Based on the following study characteristics and already conducted sensitivity tests,
            recommend additional sensitivity analyses that should be performed:
            
            Study Context: {study_characteristics}
            Domain: {self.domain}
            Current Robustness: {current_robustness}
            
            Already Conducted Tests:
            {[test.test_name for test in conducted_tests]}
            
            Focus on tests that would address the most critical remaining threats to validity.
            Prioritize tests that are feasible and would provide the most valuable information.
            """
            
            llm_response = await self.llm_client.generate_response(llm_prompt)
            
            # Parse LLM recommendations and create test objects
            additional_tests = self._parse_additional_test_recommendations(
                llm_response, conducted_types
            )
            
            return additional_tests
            
        except Exception as e:
            self.logger.error(f"Error suggesting additional tests: {str(e)}")
            raise
    
    def _analyze_study_characteristics(
        self,
        treatment_variable: str,
        outcome_variable: str,
        observed_confounders: List[str],
        data: Optional[pd.DataFrame],
        analysis_context: AnalysisContext,
        study_design_description: Optional[str]
    ) -> str:
        """Analyze study characteristics to inform sensitivity test selection."""
        characteristics = []
        
        characteristics.append(f"Treatment Variable: {treatment_variable}")
        characteristics.append(f"Outcome Variable: {outcome_variable}")
        characteristics.append(f"Analysis Context: {analysis_context.value}")
        characteristics.append(f"Domain: {self.domain}")
        characteristics.append(f"Observed Confounders: {len(observed_confounders)} variables")
        
        if data is not None:
            characteristics.append(f"Sample Size: {len(data)}")
            characteristics.append(f"Missing Data: {data.isnull().sum().sum()} missing values")
        
        if study_design_description:
            characteristics.append(f"Study Design: {study_design_description}")
        
        return "; ".join(characteristics)
    
    async def _get_llm_sensitivity_recommendations(
        self,
        study_characteristics: str,
        research_question: Optional[str]
    ) -> str:
        """Get LLM recommendations for sensitivity analysis approach."""
        prompt = f"""
        As a causal inference expert, recommend a comprehensive sensitivity analysis strategy 
        for the following study:
        
        Study Characteristics: {study_characteristics}
        Research Question: {research_question or "Not specified"}
        Domain: {self.domain}
        
        Please provide:
        1. The most critical threats to causal validity
        2. Priority ranking of sensitivity tests
        3. Specific methods that would be most appropriate
        4. Key parameters and thresholds to consider
        5. Interpretation guidelines for this context
        
        Focus on practical, implementable recommendations that address the most serious
        threats to the validity of causal conclusions.
        """
        
        return await self.llm_client.generate_response(prompt)
    
    def _create_recommended_tests(
        self,
        study_characteristics: str,
        llm_recommendations: str
    ) -> List[SensitivityTest]:
        """Create specific sensitivity test recommendations."""
        domain_priorities = self.domain_priorities.get(self.domain, self.domain_priorities["general"])
        
        tests = []
        
        # Unobserved confounding test (usually highest priority)
        if SensitivityTestType.UNOBSERVED_CONFOUNDING in domain_priorities:
            tests.append(SensitivityTest(
                test_type=SensitivityTestType.UNOBSERVED_CONFOUNDING,
                priority=domain_priorities[SensitivityTestType.UNOBSERVED_CONFOUNDING],
                method_name="Rosenbaum Bounds / E-value Analysis",
                description="Test robustness to unmeasured confounding",
                rationale="Addresses the most critical threat in observational studies",
                implementation_steps=[
                    "Calculate point estimate and confidence interval",
                    "Compute E-value for point estimate and CI limit",
                    "Compare E-value to plausible confounder strength",
                    "Assess whether unmeasured confounding could explain results"
                ],
                interpretation_guide="Higher E-values indicate greater robustness to confounding",
                threshold_values={"minimum_robust_e_value": 1.25},
                software_recommendations=["R: EValue package", "Stata: evalue command", "Python: causality package"]
            ))
        
        # Selection bias test
        if SensitivityTestType.SELECTION_BIAS in domain_priorities:
            tests.append(SensitivityTest(
                test_type=SensitivityTestType.SELECTION_BIAS,
                priority=domain_priorities[SensitivityTestType.SELECTION_BIAS],
                method_name="Heckman Selection Model / Propensity Score Diagnostics",
                description="Test for selection bias in the study sample",
                rationale="Non-random selection into treatment/study can bias results",
                implementation_steps=[
                    "Model selection into treatment",
                    "Check overlap in propensity score distributions",
                    "Test sensitivity to selection mechanism assumptions",
                    "Compare results across different matching/weighting approaches"
                ],
                interpretation_guide="Results should be consistent across different selection models",
                software_recommendations=["R: sampleSelection", "Stata: heckman", "Python: econml"]
            ))
        
        # Add more tests based on domain and context
        if SensitivityTestType.MODEL_SPECIFICATION in domain_priorities:
            tests.append(SensitivityTest(
                test_type=SensitivityTestType.MODEL_SPECIFICATION,
                priority=domain_priorities[SensitivityTestType.MODEL_SPECIFICATION],
                method_name="Specification Curve Analysis",
                description="Test robustness across different model specifications",
                rationale="Results should be consistent across reasonable model choices",
                implementation_steps=[
                    "Define reasonable set of model specifications",
                    "Estimate effect across all specifications",
                    "Plot distribution of effect estimates",
                    "Check for specification-dependent results"
                ],
                interpretation_guide="Results are robust if consistent across specifications",
                software_recommendations=["R: specr package", "Stata: multiverse", "Python: custom implementation"]
            ))
        
        return tests
    
    async def _generate_analysis_summary(
        self,
        study_characteristics: str,
        recommended_tests: List[SensitivityTest]
    ) -> str:
        """Generate an analysis summary with LLM assistance."""
        test_summary = "\n".join([
            f"- {test.method_name} ({test.priority.value} priority): {test.description}"
            for test in recommended_tests
        ])
        
        prompt = f"""
        Create a concise summary of the recommended sensitivity analysis strategy:
        
        Study: {study_characteristics}
        
        Recommended Tests:
        {test_summary}
        
        Provide a 2-3 sentence summary explaining the overall strategy and why these
        particular tests were prioritized for this study.
        """
        
        return await self.llm_client.generate_response(prompt)
    
    def _analyze_robustness_patterns(
        self,
        sensitivity_results: List[SensitivityAnalysisResult],
        original_effect_estimate: float
    ) -> Dict[str, Any]:
        """Analyze patterns in sensitivity test results."""
        if not sensitivity_results:
            return {"status": "no_results", "message": "No sensitivity results to analyze"}
        
        robustness_levels = [result.robustness_level for result in sensitivity_results]
        robust_count = robustness_levels.count("robust")
        moderate_count = robustness_levels.count("moderately_robust")
        sensitive_count = robustness_levels.count("sensitive")
        
        # Calculate effect estimate stability
        effect_estimates = []
        for result in sensitivity_results:
            if "effect_estimate" in result.robustness_metrics:
                effect_estimates.append(result.robustness_metrics["effect_estimate"])
        
        stability_metrics = {}
        if effect_estimates:
            stability_metrics = {
                "mean_effect": np.mean(effect_estimates),
                "std_effect": np.std(effect_estimates),
                "min_effect": np.min(effect_estimates),
                "max_effect": np.max(effect_estimates),
                "coefficient_of_variation": np.std(effect_estimates) / np.abs(np.mean(effect_estimates))
            }
        
        return {
            "robustness_distribution": {
                "robust": robust_count,
                "moderately_robust": moderate_count,
                "sensitive": sensitive_count,
                "total_tests": len(sensitivity_results)
            },
            "stability_metrics": stability_metrics,
            "original_effect": original_effect_estimate,
            "critical_tests": [
                result for result in sensitivity_results 
                if result.robustness_level == "sensitive"
            ]
        }
    
    async def _get_llm_interpretation(
        self,
        sensitivity_results: List[SensitivityAnalysisResult],
        original_effect_estimate: float,
        confidence_interval: Optional[Tuple[float, float]],
        robustness_analysis: Dict[str, Any]
    ) -> str:
        """Get LLM interpretation of sensitivity results."""
        results_summary = "\n".join([
            f"- {result.test_name}: {result.robustness_level} "
            f"(conclusion: {result.conclusion})"
            for result in sensitivity_results
        ])
        
        prompt = f"""
        Interpret the following sensitivity analysis results for a causal inference study:
        
        Original Effect Estimate: {original_effect_estimate}
        Confidence Interval: {confidence_interval}
        Domain: {self.domain}
        
        Sensitivity Test Results:
        {results_summary}
        
        Robustness Summary:
        - {robustness_analysis['robustness_distribution']['robust']} tests showed robust results
        - {robustness_analysis['robustness_distribution']['sensitive']} tests showed sensitivity
        
        Provide a clear interpretation of what these results mean for the validity
        of the causal conclusion. Include:
        1. Overall confidence assessment
        2. Key threats that were ruled out vs. those that remain
        3. Practical implications for decision-making
        4. Limitations and caveats
        """
        
        return await self.llm_client.generate_response(prompt)
    
    def _create_overall_assessment(
        self,
        sensitivity_results: List[SensitivityAnalysisResult],
        robustness_analysis: Dict[str, Any],
        llm_interpretation: str
    ) -> str:
        """Create overall robustness assessment."""
        total_tests = len(sensitivity_results)
        robust_tests = robustness_analysis['robustness_distribution']['robust']
        sensitive_tests = robustness_analysis['robustness_distribution']['sensitive']
        
        if robust_tests >= 0.8 * total_tests:
            return "HIGH_CONFIDENCE"
        elif robust_tests >= 0.6 * total_tests:
            return "MODERATE_CONFIDENCE"
        elif sensitive_tests <= 0.2 * total_tests:
            return "MODERATE_CONFIDENCE"
        else:
            return "LOW_CONFIDENCE"
    
    def _determine_confidence_level(
        self,
        sensitivity_results: List[SensitivityAnalysisResult]
    ) -> str:
        """Determine overall confidence level based on sensitivity results."""
        if not sensitivity_results:
            return "INSUFFICIENT_TESTING"
        
        robustness_scores = {
            "robust": 3,
            "moderately_robust": 2,
            "sensitive": 1
        }
        
        total_score = sum(robustness_scores.get(result.robustness_level, 1) 
                         for result in sensitivity_results)
        max_possible_score = len(sensitivity_results) * 3
        
        confidence_ratio = total_score / max_possible_score
        
        if confidence_ratio >= 0.8:
            return "HIGH"
        elif confidence_ratio >= 0.6:
            return "MODERATE"
        else:
            return "LOW"
    
    def _identify_key_vulnerabilities(
        self,
        sensitivity_results: List[SensitivityAnalysisResult]
    ) -> List[str]:
        """Identify key vulnerabilities from sensitivity results."""
        vulnerabilities = []
        
        for result in sensitivity_results:
            if result.robustness_level == "sensitive":
                vulnerabilities.append(
                    f"{result.test_name}: {result.conclusion}"
                )
        
        return vulnerabilities
    
    def _generate_robustness_recommendations(
        self,
        sensitivity_results: List[SensitivityAnalysisResult],
        robustness_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on robustness analysis."""
        recommendations = []
        
        sensitive_tests = [r for r in sensitivity_results if r.robustness_level == "sensitive"]
        
        if sensitive_tests:
            recommendations.append(
                f"Address {len(sensitive_tests)} critical sensitivity concerns before "
                "making strong causal claims"
            )
        
        if robustness_analysis['robustness_distribution']['robust'] < 3:
            recommendations.append(
                "Conduct additional sensitivity tests to strengthen robustness assessment"
            )
        
        if "coefficient_of_variation" in robustness_analysis.get('stability_metrics', {}):
            cv = robustness_analysis['stability_metrics']['coefficient_of_variation']
            if cv > 0.2:
                recommendations.append(
                    f"Effect estimates show high variability (CV={cv:.2f}) - "
                    "investigate sources of instability"
                )
        
        return recommendations
    
    def _parse_additional_test_recommendations(
        self,
        llm_response: str,
        conducted_types: set
    ) -> List[SensitivityTest]:
        """Parse LLM response to create additional test recommendations."""
        # This would parse the LLM response and create SensitivityTest objects
        # For now, return a basic recommendation based on what hasn't been done
        
        additional_tests = []
        all_test_types = set(SensitivityTestType)
        missing_types = all_test_types - conducted_types
        
        for test_type in missing_types:
            if test_type == SensitivityTestType.MEASUREMENT_ERROR:
                additional_tests.append(SensitivityTest(
                    test_type=test_type,
                    priority=SensitivityPriority.MODERATE,
                    method_name="Measurement Error Sensitivity Analysis",
                    description="Test robustness to measurement error in key variables",
                    rationale="Measurement error can bias effect estimates",
                    implementation_steps=[
                        "Identify variables prone to measurement error",
                        "Model different levels of measurement error",
                        "Assess impact on effect estimates",
                        "Compare with validation data if available"
                    ],
                    interpretation_guide="Results should remain stable under plausible measurement error scenarios"
                ))
        
        return additional_tests[:3]  # Limit to top 3 recommendations