"""
Assumption Checker - Validates causal inference assumptions using LLM reasoning.

This module provides comprehensive validation of key causal inference assumptions,
combining statistical tests with LLM-powered reasoning to assess the plausibility
of causal identification assumptions.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import scipy.stats as stats


class CausalAssumption(Enum):
    """Types of causal inference assumptions."""
    EXCHANGEABILITY = "exchangeability"  # No unmeasured confounding
    POSITIVITY = "positivity"  # Overlap/common support
    CONSISTENCY = "consistency"  # Well-defined interventions
    NO_INTERFERENCE = "no_interference"  # SUTVA - no spillover effects
    MONOTONICITY = "monotonicity"  # For IV: monotonic treatment assignment
    EXCLUSION_RESTRICTION = "exclusion_restriction"  # IV affects outcome only through treatment
    IGNORABILITY = "ignorability"  # Treatment assignment ignorable given covariates
    PARALLEL_TRENDS = "parallel_trends"  # For diff-in-diff
    NO_ANTICIPATION = "no_anticipation"  # No pre-treatment effects
    STABLE_UNIT_TREATMENT = "stable_unit_treatment"  # SUTVA components
    LINEARITY = "linearity"  # Linear relationship assumptions
    HOMOSCEDASTICITY = "homoscedasticity"  # Constant variance
    INDEPENDENCE = "independence"  # Independent observations


class AssumptionStatus(Enum):
    """Status of assumption validation."""
    LIKELY_SATISFIED = "likely_satisfied"
    PLAUSIBLE = "plausible"
    QUESTIONABLE = "questionable"
    LIKELY_VIOLATED = "likely_violated"
    CANNOT_TEST = "cannot_test"


class ValidationMethod(Enum):
    """Methods for validating assumptions."""
    STATISTICAL_TEST = "statistical_test"
    GRAPHICAL_ANALYSIS = "graphical_analysis"
    LLM_REASONING = "llm_reasoning"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"
    PLACEBO_TEST = "placebo_test"
    FALSIFICATION_TEST = "falsification_test"


@dataclass
class AssumptionTest:
    """Represents a test for a specific assumption."""
    assumption: CausalAssumption
    test_name: str
    method: ValidationMethod
    description: str
    implementation_steps: List[str]
    interpretation_guide: str
    statistical_test: Optional[str] = None
    required_data: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)


@dataclass
class AssumptionValidationResult:
    """Results from validating a specific assumption."""
    assumption: CausalAssumption
    test_name: str
    status: AssumptionStatus
    evidence_strength: str  # "strong", "moderate", "weak"
    statistical_results: Dict[str, Any] = field(default_factory=dict)
    llm_assessment: str = ""
    supporting_evidence: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    confidence_level: float = 0.0  # 0-1 scale
    recommendations: List[str] = field(default_factory=list)


@dataclass
class AssumptionValidationReport:
    """Comprehensive assumption validation report."""
    study_context: str
    analysis_method: str
    validated_assumptions: List[AssumptionValidationResult]
    overall_assessment: str
    critical_violations: List[str] = field(default_factory=list)
    plausibility_score: float = 0.0  # 0-1 overall plausibility
    recommendations: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)


class AssumptionChecker:
    """
    LLM-powered assumption checker for causal inference.
    
    Validates key causal inference assumptions using a combination of
    statistical tests, graphical analysis, and LLM reasoning.
    """
    
    def __init__(self, llm_client: Any, domain: str = "general"):
        """
        Initialize the assumption checker.
        
        Args:
            llm_client: LLM client for reasoning about assumptions
            domain: Research domain for context-specific validation
        """
        self.llm_client = llm_client
        self.domain = domain
        self.logger = logging.getLogger(__name__)
        
        # Domain-specific assumption priorities
        self.domain_assumptions = {
            "healthcare": [
                CausalAssumption.EXCHANGEABILITY,
                CausalAssumption.POSITIVITY,
                CausalAssumption.CONSISTENCY,
                CausalAssumption.NO_INTERFERENCE,
                CausalAssumption.NO_ANTICIPATION
            ],
            "economics": [
                CausalAssumption.EXCHANGEABILITY,
                CausalAssumption.POSITIVITY,
                CausalAssumption.MONOTONICITY,
                CausalAssumption.EXCLUSION_RESTRICTION,
                CausalAssumption.PARALLEL_TRENDS
            ],
            "education": [
                CausalAssumption.EXCHANGEABILITY,
                CausalAssumption.NO_INTERFERENCE,
                CausalAssumption.CONSISTENCY,
                CausalAssumption.PARALLEL_TRENDS,
                CausalAssumption.NO_ANTICIPATION
            ],
            "general": [
                CausalAssumption.EXCHANGEABILITY,
                CausalAssumption.POSITIVITY,
                CausalAssumption.CONSISTENCY,
                CausalAssumption.NO_INTERFERENCE
            ]
        }
    
    async def validate_assumptions(
        self,
        data: pd.DataFrame,
        treatment_variable: str,
        outcome_variable: str,
        covariates: List[str],
        analysis_method: str = "regression",
        instrumental_variable: Optional[str] = None,
        time_variable: Optional[str] = None,
        cluster_variable: Optional[str] = None,
        study_description: Optional[str] = None
    ) -> AssumptionValidationReport:
        """
        Validate causal inference assumptions for a given analysis.
        
        Args:
            data: Dataset for analysis
            treatment_variable: Name of treatment variable
            outcome_variable: Name of outcome variable
            covariates: List of covariate names
            analysis_method: Causal inference method (regression, matching, IV, etc.)
            instrumental_variable: Name of IV if applicable
            time_variable: Name of time variable for panel data
            cluster_variable: Name of cluster variable if applicable
            study_description: Description of the study design
            
        Returns:
            AssumptionValidationReport with validation results
        """
        try:
            # Determine relevant assumptions based on method and domain
            relevant_assumptions = self._get_relevant_assumptions(
                analysis_method, instrumental_variable, time_variable
            )
            
            # Validate each assumption
            validation_results = []
            for assumption in relevant_assumptions:
                result = await self._validate_single_assumption(
                    assumption, data, treatment_variable, outcome_variable,
                    covariates, analysis_method, instrumental_variable,
                    time_variable, cluster_variable, study_description
                )
                validation_results.append(result)
            
            # Generate overall assessment
            overall_assessment = await self._generate_overall_assessment(
                validation_results, analysis_method, study_description
            )
            
            # Calculate plausibility score
            plausibility_score = self._calculate_plausibility_score(validation_results)
            
            # Identify critical violations
            critical_violations = self._identify_critical_violations(validation_results)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                validation_results, critical_violations, analysis_method
            )
            
            report = AssumptionValidationReport(
                study_context=study_description or f"{analysis_method} analysis in {self.domain}",
                analysis_method=analysis_method,
                validated_assumptions=validation_results,
                overall_assessment=overall_assessment,
                critical_violations=critical_violations,
                plausibility_score=plausibility_score,
                recommendations=recommendations
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error validating assumptions: {str(e)}")
            raise
    
    async def validate_exchangeability(
        self,
        data: pd.DataFrame,
        treatment_variable: str,
        outcome_variable: str,
        covariates: List[str],
        study_description: Optional[str] = None
    ) -> AssumptionValidationResult:
        """
        Validate the exchangeability assumption (no unmeasured confounding).
        
        Args:
            data: Dataset
            treatment_variable: Treatment variable name
            outcome_variable: Outcome variable name
            covariates: List of measured covariates
            study_description: Description of study design
            
        Returns:
            AssumptionValidationResult for exchangeability
        """
        try:
            # Statistical tests for balance
            balance_tests = self._perform_balance_tests(data, treatment_variable, covariates)
            
            # LLM reasoning about unmeasured confounding
            llm_assessment = await self._get_llm_exchangeability_assessment(
                treatment_variable, outcome_variable, covariates, 
                study_description, balance_tests
            )
            
            # Determine status based on evidence
            status, confidence = self._assess_exchangeability_status(
                balance_tests, llm_assessment
            )
            
            return AssumptionValidationResult(
                assumption=CausalAssumption.EXCHANGEABILITY,
                test_name="Exchangeability Assessment",
                status=status,
                evidence_strength=self._categorize_evidence_strength(confidence),
                statistical_results=balance_tests,
                llm_assessment=llm_assessment,
                confidence_level=confidence,
                supporting_evidence=self._extract_supporting_evidence(balance_tests, llm_assessment),
                concerns=self._extract_concerns(balance_tests, llm_assessment),
                recommendations=self._generate_exchangeability_recommendations(status, balance_tests)
            )
            
        except Exception as e:
            self.logger.error(f"Error validating exchangeability: {str(e)}")
            raise
    
    async def validate_positivity(
        self,
        data: pd.DataFrame,
        treatment_variable: str,
        covariates: List[str]
    ) -> AssumptionValidationResult:
        """
        Validate the positivity assumption (overlap/common support).
        
        Args:
            data: Dataset
            treatment_variable: Treatment variable name
            covariates: List of covariates for propensity score
            
        Returns:
            AssumptionValidationResult for positivity
        """
        try:
            # Calculate propensity scores if possible
            positivity_tests = self._perform_positivity_tests(
                data, treatment_variable, covariates
            )
            
            # LLM assessment of overlap
            llm_assessment = await self._get_llm_positivity_assessment(
                treatment_variable, covariates, positivity_tests
            )
            
            # Determine status
            status, confidence = self._assess_positivity_status(
                positivity_tests, llm_assessment
            )
            
            return AssumptionValidationResult(
                assumption=CausalAssumption.POSITIVITY,
                test_name="Positivity/Overlap Assessment",
                status=status,
                evidence_strength=self._categorize_evidence_strength(confidence),
                statistical_results=positivity_tests,
                llm_assessment=llm_assessment,
                confidence_level=confidence,
                supporting_evidence=self._extract_supporting_evidence(positivity_tests, llm_assessment),
                concerns=self._extract_concerns(positivity_tests, llm_assessment),
                recommendations=self._generate_positivity_recommendations(status, positivity_tests)
            )
            
        except Exception as e:
            self.logger.error(f"Error validating positivity: {str(e)}")
            raise
    
    def _get_relevant_assumptions(
        self,
        analysis_method: str,
        instrumental_variable: Optional[str],
        time_variable: Optional[str]
    ) -> List[CausalAssumption]:
        """Determine relevant assumptions based on analysis method."""
        base_assumptions = self.domain_assumptions.get(
            self.domain, self.domain_assumptions["general"]
        ).copy()
        
        # Add method-specific assumptions
        if "iv" in analysis_method.lower() or instrumental_variable:
            base_assumptions.extend([
                CausalAssumption.MONOTONICITY,
                CausalAssumption.EXCLUSION_RESTRICTION
            ])
        
        if "diff" in analysis_method.lower() or time_variable:
            base_assumptions.extend([
                CausalAssumption.PARALLEL_TRENDS,
                CausalAssumption.NO_ANTICIPATION
            ])
        
        if "regression" in analysis_method.lower():
            base_assumptions.extend([
                CausalAssumption.LINEARITY,
                CausalAssumption.HOMOSCEDASTICITY
            ])
        
        return list(set(base_assumptions))  # Remove duplicates
    
    async def _validate_single_assumption(
        self,
        assumption: CausalAssumption,
        data: pd.DataFrame,
        treatment_variable: str,
        outcome_variable: str,
        covariates: List[str],
        analysis_method: str,
        instrumental_variable: Optional[str],
        time_variable: Optional[str],
        cluster_variable: Optional[str],
        study_description: Optional[str]
    ) -> AssumptionValidationResult:
        """Validate a single causal assumption."""
        
        if assumption == CausalAssumption.EXCHANGEABILITY:
            return await self.validate_exchangeability(
                data, treatment_variable, outcome_variable, covariates, study_description
            )
        elif assumption == CausalAssumption.POSITIVITY:
            return await self.validate_positivity(
                data, treatment_variable, covariates
            )
        elif assumption == CausalAssumption.CONSISTENCY:
            return await self._validate_consistency(
                treatment_variable, outcome_variable, study_description
            )
        elif assumption == CausalAssumption.NO_INTERFERENCE:
            return await self._validate_no_interference(
                data, cluster_variable, study_description
            )
        elif assumption == CausalAssumption.PARALLEL_TRENDS:
            return await self._validate_parallel_trends(
                data, treatment_variable, outcome_variable, time_variable
            )
        else:
            # Generic validation for other assumptions
            return await self._generic_assumption_validation(
                assumption, data, study_description
            )
    
    def _perform_balance_tests(
        self,
        data: pd.DataFrame,
        treatment_variable: str,
        covariates: List[str]
    ) -> Dict[str, Any]:
        """Perform statistical balance tests for exchangeability."""
        results = {}
        
        try:
            treatment = data[treatment_variable]
            
            # Test balance for each covariate
            balance_tests = {}
            standardized_diffs = {}
            
            for covar in covariates:
                if covar in data.columns:
                    covar_data = data[covar].dropna()
                    treat_data = covar_data[treatment == 1]
                    control_data = covar_data[treatment == 0]
                    
                    if len(treat_data) > 0 and len(control_data) > 0:
                        # T-test for balance
                        t_stat, p_val = stats.ttest_ind(treat_data, control_data)
                        balance_tests[covar] = {"t_stat": t_stat, "p_value": p_val}
                        
                        # Standardized difference
                        pooled_std = np.sqrt((treat_data.var() + control_data.var()) / 2)
                        if pooled_std > 0:
                            std_diff = (treat_data.mean() - control_data.mean()) / pooled_std
                            standardized_diffs[covar] = abs(std_diff)
            
            results["balance_tests"] = balance_tests
            results["standardized_differences"] = standardized_diffs
            results["max_std_diff"] = max(standardized_diffs.values()) if standardized_diffs else 0
            results["imbalanced_vars"] = [
                var for var, diff in standardized_diffs.items() if diff > 0.1
            ]
            
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    def _perform_positivity_tests(
        self,
        data: pd.DataFrame,
        treatment_variable: str,
        covariates: List[str]
    ) -> Dict[str, Any]:
        """Perform tests for positivity/overlap assumption."""
        results = {}
        
        try:
            # Check for extreme propensity scores or perfect separation
            treatment = data[treatment_variable]
            
            # Simple overlap check
            n_treated = sum(treatment == 1)
            n_control = sum(treatment == 0)
            total_n = len(treatment)
            
            results["treatment_prevalence"] = n_treated / total_n
            results["n_treated"] = n_treated
            results["n_control"] = n_control
            
            # Check for covariate patterns with no overlap
            overlap_issues = []
            
            for covar in covariates[:5]:  # Limit to first 5 covariates
                if covar in data.columns:
                    # For categorical variables, check if any category has only treated or only controls
                    if data[covar].dtype == 'object' or data[covar].nunique() < 10:
                        crosstab = pd.crosstab(data[covar], treatment)
                        for category in crosstab.index:
                            if crosstab.loc[category, 0] == 0 or crosstab.loc[category, 1] == 0:
                                overlap_issues.append(f"{covar}={category}")
            
            results["overlap_issues"] = overlap_issues
            results["has_overlap_issues"] = len(overlap_issues) > 0
            
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    async def _get_llm_exchangeability_assessment(
        self,
        treatment_variable: str,
        outcome_variable: str,
        covariates: List[str],
        study_description: Optional[str],
        balance_tests: Dict[str, Any]
    ) -> str:
        """Get LLM assessment of exchangeability assumption."""
        
        balance_summary = ""
        if "standardized_differences" in balance_tests:
            imbalanced = balance_tests.get("imbalanced_vars", [])
            max_diff = balance_tests.get("max_std_diff", 0)
            balance_summary = f"Max standardized difference: {max_diff:.3f}. Imbalanced variables: {imbalanced}"
        
        prompt = f"""
        Assess the plausibility of the exchangeability assumption (no unmeasured confounding) 
        for this causal analysis:
        
        Treatment: {treatment_variable}
        Outcome: {outcome_variable}
        Domain: {self.domain}
        Observed Covariates: {covariates}
        Study Description: {study_description or "Not provided"}
        
        Statistical Balance: {balance_summary}
        
        Consider:
        1. What unmeasured confounders might exist given the domain and variables?
        2. How plausible is it that all confounders were measured?
        3. Do the observed covariates represent the key confounding pathways?
        4. What does the balance on observed variables suggest about unobserved balance?
        
        Provide an assessment of how likely this assumption is to hold and what the 
        main threats might be.
        """
        
        return await self.llm_client.generate_response(prompt)
    
    async def _get_llm_positivity_assessment(
        self,
        treatment_variable: str,
        covariates: List[str],
        positivity_tests: Dict[str, Any]
    ) -> str:
        """Get LLM assessment of positivity assumption."""
        
        overlap_summary = f"""
        Treatment prevalence: {positivity_tests.get('treatment_prevalence', 'unknown')}
        Overlap issues found: {positivity_tests.get('has_overlap_issues', 'unknown')}
        Specific issues: {positivity_tests.get('overlap_issues', [])}
        """
        
        prompt = f"""
        Assess the positivity/overlap assumption for this analysis:
        
        Treatment: {treatment_variable}
        Covariates: {covariates}
        Domain: {self.domain}
        
        Statistical Analysis: {overlap_summary}
        
        Consider:
        1. Is there sufficient overlap in covariate distributions between treated and control groups?
        2. Are there subgroups where treatment assignment is deterministic?
        3. What does the treatment prevalence suggest about positivity?
        4. Are there practical limitations to treatment assignment?
        
        Assess how likely this assumption is to hold and identify key concerns.
        """
        
        return await self.llm_client.generate_response(prompt)
    
    async def _validate_consistency(
        self,
        treatment_variable: str,
        outcome_variable: str,
        study_description: Optional[str]
    ) -> AssumptionValidationResult:
        """Validate consistency assumption (well-defined interventions)."""
        
        llm_assessment = await self.llm_client.generate_response(f"""
        Assess the consistency assumption for this analysis:
        
        Treatment: {treatment_variable}
        Outcome: {outcome_variable}
        Study: {study_description or "Not described"}
        Domain: {self.domain}
        
        Consider:
        1. Is the treatment well-defined and implemented consistently?
        2. Could there be different versions or intensities of the treatment?
        3. Are there potential implementation variations that matter?
        4. Is the outcome measured consistently across units?
        
        Assess the plausibility of the consistency assumption.
        """)
        
        # Simple heuristic for status
        if "well-defined" in llm_assessment.lower() and "consistent" in llm_assessment.lower():
            status = AssumptionStatus.PLAUSIBLE
            confidence = 0.7
        elif "concern" in llm_assessment.lower() or "variation" in llm_assessment.lower():
            status = AssumptionStatus.QUESTIONABLE
            confidence = 0.4
        else:
            status = AssumptionStatus.PLAUSIBLE
            confidence = 0.6
        
        return AssumptionValidationResult(
            assumption=CausalAssumption.CONSISTENCY,
            test_name="Consistency Assessment",
            status=status,
            evidence_strength=self._categorize_evidence_strength(confidence),
            llm_assessment=llm_assessment,
            confidence_level=confidence
        )
    
    async def _validate_no_interference(
        self,
        data: pd.DataFrame,
        cluster_variable: Optional[str],
        study_description: Optional[str]
    ) -> AssumptionValidationResult:
        """Validate no interference assumption (SUTVA)."""
        
        cluster_info = ""
        if cluster_variable and cluster_variable in data.columns:
            n_clusters = data[cluster_variable].nunique()
            cluster_info = f"Data has {n_clusters} clusters."
        
        llm_assessment = await self.llm_client.generate_response(f"""
        Assess the no interference assumption (SUTVA) for this analysis:
        
        Study: {study_description or "Not described"}
        Domain: {self.domain}
        Clustering: {cluster_info}
        
        Consider:
        1. Could treatment of one unit affect outcomes of other units?
        2. Are there network effects, spillovers, or contagion?
        3. Is there geographic, social, or temporal proximity that might matter?
        4. Could there be general equilibrium effects?
        
        Assess how plausible the no interference assumption is.
        """)
        
        # Determine status based on domain and clustering
        if cluster_variable:
            status = AssumptionStatus.QUESTIONABLE
            confidence = 0.4
        elif self.domain in ["healthcare", "education"]:
            status = AssumptionStatus.PLAUSIBLE
            confidence = 0.6
        else:
            status = AssumptionStatus.PLAUSIBLE
            confidence = 0.7
        
        return AssumptionValidationResult(
            assumption=CausalAssumption.NO_INTERFERENCE,
            test_name="No Interference Assessment",
            status=status,
            evidence_strength=self._categorize_evidence_strength(confidence),
            llm_assessment=llm_assessment,
            confidence_level=confidence
        )
    
    async def _validate_parallel_trends(
        self,
        data: pd.DataFrame,
        treatment_variable: str,
        outcome_variable: str,
        time_variable: Optional[str]
    ) -> AssumptionValidationResult:
        """Validate parallel trends assumption for difference-in-differences."""
        
        if not time_variable or time_variable not in data.columns:
            return AssumptionValidationResult(
                assumption=CausalAssumption.PARALLEL_TRENDS,
                test_name="Parallel Trends Assessment",
                status=AssumptionStatus.CANNOT_TEST,
                evidence_strength="none",
                llm_assessment="Cannot test parallel trends without time variable data.",
                confidence_level=0.0
            )
        
        # Simple pre-trend test
        pre_treatment_data = data[data[time_variable] < data[time_variable].median()]
        trend_test = self._perform_pre_trend_test(
            pre_treatment_data, treatment_variable, outcome_variable, time_variable
        )
        
        llm_assessment = await self.llm_client.generate_response(f"""
        Assess the parallel trends assumption for difference-in-differences analysis:
        
        Treatment: {treatment_variable}
        Outcome: {outcome_variable}
        Time Variable: {time_variable}
        Domain: {self.domain}
        
        Pre-trend Test: {trend_test}
        
        Consider:
        1. Do treated and control groups have similar trends before treatment?
        2. Are there time-varying confounders that affect groups differently?
        3. Is the treatment timing truly exogenous?
        4. Could there be anticipation effects?
        
        Assess the plausibility of parallel trends.
        """)
        
        # Status based on pre-trend test
        if trend_test.get("significant_pre_trend", False):
            status = AssumptionStatus.LIKELY_VIOLATED
            confidence = 0.2
        else:
            status = AssumptionStatus.PLAUSIBLE
            confidence = 0.6
        
        return AssumptionValidationResult(
            assumption=CausalAssumption.PARALLEL_TRENDS,
            test_name="Parallel Trends Assessment",
            status=status,
            evidence_strength=self._categorize_evidence_strength(confidence),
            statistical_results=trend_test,
            llm_assessment=llm_assessment,
            confidence_level=confidence
        )
    
    def _perform_pre_trend_test(
        self,
        data: pd.DataFrame,
        treatment_variable: str,
        outcome_variable: str,
        time_variable: str
    ) -> Dict[str, Any]:
        """Simple pre-trend test for parallel trends."""
        try:
            # Group by treatment and time, calculate means
            grouped = data.groupby([treatment_variable, time_variable])[outcome_variable].mean().reset_index()
            
            # Check if we have multiple time periods for both groups
            treat_times = grouped[grouped[treatment_variable] == 1][time_variable].nunique()
            control_times = grouped[grouped[treatment_variable] == 0][time_variable].nunique()
            
            if treat_times < 2 or control_times < 2:
                return {"error": "Insufficient time periods for trend test"}
            
            # Simple test: correlation of time with outcome for each group
            treat_corr = data[data[treatment_variable] == 1][[time_variable, outcome_variable]].corr().iloc[0,1]
            control_corr = data[data[treatment_variable] == 0][[time_variable, outcome_variable]].corr().iloc[0,1]
            
            trend_diff = abs(treat_corr - control_corr)
            
            return {
                "treated_trend_corr": treat_corr,
                "control_trend_corr": control_corr,
                "trend_difference": trend_diff,
                "significant_pre_trend": trend_diff > 0.2  # Simple threshold
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _generic_assumption_validation(
        self,
        assumption: CausalAssumption,
        data: pd.DataFrame,
        study_description: Optional[str]
    ) -> AssumptionValidationResult:
        """Generic validation for assumptions without specific tests."""
        
        llm_assessment = await self.llm_client.generate_response(f"""
        Assess the {assumption.value} assumption for this causal analysis:
        
        Study: {study_description or "Not described"}
        Domain: {self.domain}
        Data shape: {data.shape if data is not None else "Not provided"}
        
        Provide an assessment of how plausible this assumption is in this context
        and what the main concerns might be.
        """)
        
        return AssumptionValidationResult(
            assumption=assumption,
            test_name=f"{assumption.value.replace('_', ' ').title()} Assessment",
            status=AssumptionStatus.PLAUSIBLE,
            evidence_strength="moderate",
            llm_assessment=llm_assessment,
            confidence_level=0.5
        )
    
    def _assess_exchangeability_status(
        self,
        balance_tests: Dict[str, Any],
        llm_assessment: str
    ) -> Tuple[AssumptionStatus, float]:
        """Assess exchangeability status based on evidence."""
        
        max_std_diff = balance_tests.get("max_std_diff", 0)
        n_imbalanced = len(balance_tests.get("imbalanced_vars", []))
        
        # Statistical evidence
        if max_std_diff > 0.25 or n_imbalanced > len(balance_tests.get("standardized_differences", {})) * 0.5:
            statistical_concern = True
        else:
            statistical_concern = False
        
        # LLM assessment
        llm_concern = any(word in llm_assessment.lower() 
                         for word in ["concern", "violation", "unlikely", "implausible"])
        
        # Combine evidence
        if statistical_concern and llm_concern:
            return AssumptionStatus.LIKELY_VIOLATED, 0.2
        elif statistical_concern or llm_concern:
            return AssumptionStatus.QUESTIONABLE, 0.4
        elif "likely" in llm_assessment.lower() and "satisfied" in llm_assessment.lower():
            return AssumptionStatus.LIKELY_SATISFIED, 0.8
        else:
            return AssumptionStatus.PLAUSIBLE, 0.6
    
    def _assess_positivity_status(
        self,
        positivity_tests: Dict[str, Any],
        llm_assessment: str
    ) -> Tuple[AssumptionStatus, float]:
        """Assess positivity status based on evidence."""
        
        has_overlap_issues = positivity_tests.get("has_overlap_issues", False)
        treatment_prevalence = positivity_tests.get("treatment_prevalence", 0.5)
        
        # Check for extreme prevalences
        extreme_prevalence = treatment_prevalence < 0.05 or treatment_prevalence > 0.95
        
        if has_overlap_issues and extreme_prevalence:
            return AssumptionStatus.LIKELY_VIOLATED, 0.2
        elif has_overlap_issues or extreme_prevalence:
            return AssumptionStatus.QUESTIONABLE, 0.4
        else:
            return AssumptionStatus.PLAUSIBLE, 0.7
    
    def _categorize_evidence_strength(self, confidence: float) -> str:
        """Categorize evidence strength based on confidence level."""
        if confidence >= 0.7:
            return "strong"
        elif confidence >= 0.5:
            return "moderate"
        else:
            return "weak"
    
    def _extract_supporting_evidence(
        self,
        statistical_results: Dict[str, Any],
        llm_assessment: str
    ) -> List[str]:
        """Extract supporting evidence from results."""
        evidence = []
        
        if statistical_results:
            max_diff = statistical_results.get("max_std_diff", 0)
            if max_diff < 0.1:
                evidence.append(f"Good covariate balance (max std diff: {max_diff:.3f})")
        
        # Simple extraction from LLM assessment
        if "balance" in llm_assessment.lower():
            evidence.append("LLM assessment notes good balance")
        if "plausible" in llm_assessment.lower():
            evidence.append("LLM assessment finds assumption plausible")
        
        return evidence
    
    def _extract_concerns(
        self,
        statistical_results: Dict[str, Any],
        llm_assessment: str
    ) -> List[str]:
        """Extract concerns from results."""
        concerns = []
        
        if statistical_results:
            imbalanced_vars = statistical_results.get("imbalanced_vars", [])
            if imbalanced_vars:
                concerns.append(f"Imbalanced covariates: {imbalanced_vars}")
        
        # Simple extraction from LLM assessment
        if "concern" in llm_assessment.lower():
            concerns.append("LLM identifies potential concerns")
        if "unmeasured" in llm_assessment.lower():
            concerns.append("Potential unmeasured confounding identified")
        
        return concerns
    
    def _generate_exchangeability_recommendations(
        self,
        status: AssumptionStatus,
        balance_tests: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for exchangeability issues."""
        recommendations = []
        
        if status == AssumptionStatus.LIKELY_VIOLATED:
            recommendations.append("Conduct sensitivity analysis for unmeasured confounding")
            recommendations.append("Consider instrumental variable analysis if appropriate")
        
        if status == AssumptionStatus.QUESTIONABLE:
            recommendations.append("Improve covariate measurement and inclusion")
            recommendations.append("Consider matching or weighting methods")
        
        imbalanced_vars = balance_tests.get("imbalanced_vars", [])
        if imbalanced_vars:
            recommendations.append(f"Address imbalance in: {', '.join(imbalanced_vars)}")
        
        return recommendations
    
    def _generate_positivity_recommendations(
        self,
        status: AssumptionStatus,
        positivity_tests: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for positivity issues."""
        recommendations = []
        
        if status == AssumptionStatus.LIKELY_VIOLATED:
            recommendations.append("Trim extreme propensity scores")
            recommendations.append("Focus analysis on regions of common support")
        
        overlap_issues = positivity_tests.get("overlap_issues", [])
        if overlap_issues:
            recommendations.append(f"Address overlap issues in: {', '.join(overlap_issues)}")
        
        return recommendations
    
    async def _generate_overall_assessment(
        self,
        validation_results: List[AssumptionValidationResult],
        analysis_method: str,
        study_description: Optional[str]
    ) -> str:
        """Generate overall assessment of assumption validity."""
        
        status_counts = {}
        for result in validation_results:
            status = result.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        critical_violations = [r for r in validation_results 
                              if r.status == AssumptionStatus.LIKELY_VIOLATED]
        
        prompt = f"""
        Provide an overall assessment of assumption validity for this {analysis_method} analysis:
        
        Study: {study_description or "Not described"}
        Domain: {self.domain}
        
        Assumption Status Summary:
        {dict(status_counts)}
        
        Critical Violations: {len(critical_violations)} assumptions likely violated
        
        Provide a concise overall assessment of the plausibility of causal identification
        and the strength of causal conclusions that can be drawn.
        """
        
        return await self.llm_client.generate_response(prompt)
    
    def _calculate_plausibility_score(
        self,
        validation_results: List[AssumptionValidationResult]
    ) -> float:
        """Calculate overall plausibility score (0-1)."""
        
        if not validation_results:
            return 0.0
        
        status_scores = {
            AssumptionStatus.LIKELY_SATISFIED: 1.0,
            AssumptionStatus.PLAUSIBLE: 0.7,
            AssumptionStatus.QUESTIONABLE: 0.4,
            AssumptionStatus.LIKELY_VIOLATED: 0.1,
            AssumptionStatus.CANNOT_TEST: 0.5
        }
        
        total_score = sum(status_scores.get(result.status, 0.5) 
                         for result in validation_results)
        
        return total_score / len(validation_results)
    
    def _identify_critical_violations(
        self,
        validation_results: List[AssumptionValidationResult]
    ) -> List[str]:
        """Identify critical assumption violations."""
        
        critical_violations = []
        
        for result in validation_results:
            if result.status == AssumptionStatus.LIKELY_VIOLATED:
                critical_violations.append(
                    f"{result.assumption.value}: {result.test_name}"
                )
        
        return critical_violations
    
    async def _generate_recommendations(
        self,
        validation_results: List[AssumptionValidationResult],
        critical_violations: List[str],
        analysis_method: str
    ) -> List[str]:
        """Generate overall recommendations."""
        
        recommendations = []
        
        # Collect specific recommendations
        for result in validation_results:
            recommendations.extend(result.recommendations)
        
        # Add general recommendations
        if critical_violations:
            recommendations.append(
                f"Address {len(critical_violations)} critical assumption violations "
                "before drawing causal conclusions"
            )
        
        questionable_count = sum(
            1 for result in validation_results 
            if result.status == AssumptionStatus.QUESTIONABLE
        )
        
        if questionable_count > 0:
            recommendations.append(
                f"Strengthen evidence for {questionable_count} questionable assumptions"
            )
        
        # Method-specific recommendations
        if "regression" in analysis_method.lower():
            recommendations.append("Consider robustness checks with different model specifications")
        
        if "matching" in analysis_method.lower():
            recommendations.append("Validate matching quality and balance diagnostics")
        
        return list(set(recommendations))  # Remove duplicates