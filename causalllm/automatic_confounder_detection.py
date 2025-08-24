"""
Automatic Confounder Detection System

This module provides automated detection and identification of confounding variables
using a combination of statistical analysis, LLM reasoning, domain knowledge, and
causal discovery algorithms. It integrates with the existing confounder reasoning
module for enhanced accuracy.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import networkx as nx
import itertools
from collections import defaultdict

from causalllm.logging import get_logger
from causalllm.llm_confounder_reasoning import (
    LLMConfounderReasoning, ConfounderCandidate, ConfounderType, 
    ConfounderStrength, AdjustmentStrategy
)


class DetectionMethod(Enum):
    """Methods for confounder detection."""
    STATISTICAL_ASSOCIATION = "statistical_association"
    MUTUAL_INFORMATION = "mutual_information"
    RANDOM_FOREST_IMPORTANCE = "random_forest_importance"
    LLM_REASONING = "llm_reasoning"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    CAUSAL_DISCOVERY = "causal_discovery"
    HYBRID_ENSEMBLE = "hybrid_ensemble"


class ConfounderEvidence(Enum):
    """Types of evidence for confounding."""
    STATISTICAL_CORRELATION = "statistical_correlation"
    DOMAIN_EXPERT_KNOWLEDGE = "domain_knowledge"
    TEMPORAL_PRECEDENCE = "temporal_precedence"
    BIOLOGICAL_PLAUSIBILITY = "biological_plausibility"
    PRIOR_LITERATURE = "prior_literature"
    DATA_DRIVEN_DISCOVERY = "data_driven_discovery"


@dataclass
class ConfounderDetectionResult:
    """Result from automatic confounder detection."""
    
    variable_name: str
    affects_treatment_score: float  # 0-1 score
    affects_outcome_score: float   # 0-1 score
    confounding_score: float       # Combined score
    detection_method: DetectionMethod
    evidence_type: List[ConfounderEvidence]
    statistical_evidence: Dict[str, Any]
    reasoning: str
    confidence: float
    recommended_adjustment: List[AdjustmentStrategy]


@dataclass
class ConfounderDetectionSummary:
    """Summary of all confounder detection results."""
    
    treatment_variable: str
    outcome_variable: str
    detected_confounders: List[ConfounderDetectionResult]
    colliders_detected: List[ConfounderDetectionResult]
    mediators_detected: List[ConfounderDetectionResult]
    instrumental_variables: List[ConfounderDetectionResult]
    overall_confounding_risk: str
    recommended_adjustment_set: List[str]
    alternative_adjustment_sets: List[List[str]]
    validation_suggestions: List[str]


class AutomaticConfounderDetector:
    """Comprehensive automatic confounder detection system."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.llm_confounder_reasoner = LLMConfounderReasoning(llm_client)
        self.logger = get_logger("causalllm.automatic_confounder_detection")
        
        # Detection thresholds
        self.correlation_threshold = 0.1
        self.mutual_info_threshold = 0.05
        self.importance_threshold = 0.01
        self.significance_threshold = 0.05
        
        # Domain-specific confounder patterns
        self.domain_confounder_patterns = {
            "healthcare": {
                "always_check": ["age", "gender", "socioeconomic_status", "comorbidities"],
                "interaction_patterns": [
                    ("age", "treatment_response"),
                    ("gender", "drug_metabolism"),
                    ("comorbidities", "treatment_selection")
                ],
                "typical_confounders": {
                    "observational_study": ["indication_severity", "healthcare_access", "physician_preference"],
                    "intervention_study": ["baseline_characteristics", "adherence", "motivation"]
                }
            },
            "business": {
                "always_check": ["market_segment", "company_size", "industry", "geographic_region"],
                "interaction_patterns": [
                    ("market_conditions", "strategy_effectiveness"),
                    ("company_size", "resource_availability"),
                    ("competition", "pricing_strategy")
                ],
                "typical_confounders": {
                    "marketing_analysis": ["brand_strength", "market_share", "seasonal_effects"],
                    "hr_analysis": ["job_level", "tenure", "department", "performance_history"]
                }
            },
            "education": {
                "always_check": ["prior_achievement", "socioeconomic_background", "school_characteristics"],
                "interaction_patterns": [
                    ("teacher_quality", "student_motivation"),
                    ("class_size", "teaching_method"),
                    ("resources", "student_needs")
                ],
                "typical_confounders": {
                    "intervention_study": ["selection_bias", "implementation_fidelity", "hawthorne_effect"],
                    "policy_analysis": ["school_resources", "community_characteristics", "selection_effects"]
                }
            }
        }
    
    async def detect_confounders(self,
                               data: pd.DataFrame,
                               treatment_variable: str,
                               outcome_variable: str,
                               variable_descriptions: Optional[Dict[str, str]] = None,
                               domain: str = "general",
                               context: str = "",
                               methods: List[DetectionMethod] = None,
                               exclude_variables: List[str] = None) -> ConfounderDetectionSummary:
        """
        Comprehensive automatic confounder detection.
        
        Args:
            data: Dataset for analysis
            treatment_variable: Treatment/exposure variable
            outcome_variable: Outcome variable
            variable_descriptions: Optional descriptions of variables
            domain: Domain context
            context: Additional context
            methods: Detection methods to use (default: all)
            exclude_variables: Variables to exclude from detection
            
        Returns:
            Comprehensive confounder detection summary
        """
        self.logger.info(f"Starting automatic confounder detection for {treatment_variable} -> {outcome_variable}")
        
        if methods is None:
            methods = [DetectionMethod.HYBRID_ENSEMBLE]
        
        if exclude_variables is None:
            exclude_variables = []
        
        # Get candidate variables
        candidate_variables = [col for col in data.columns 
                             if col not in [treatment_variable, outcome_variable] + exclude_variables]
        
        # Run detection methods
        detection_results = []
        
        for method in methods:
            if method == DetectionMethod.HYBRID_ENSEMBLE:
                results = await self._hybrid_ensemble_detection(
                    data, treatment_variable, outcome_variable, 
                    candidate_variables, variable_descriptions, domain, context
                )
            else:
                results = await self._single_method_detection(
                    data, treatment_variable, outcome_variable,
                    candidate_variables, method, variable_descriptions, domain, context
                )
            
            detection_results.extend(results)
        
        # Aggregate and rank results
        aggregated_results = self._aggregate_detection_results(detection_results)
        
        # Classify detected variables
        confounders, colliders, mediators, instrumental = self._classify_detected_variables(
            aggregated_results, data, treatment_variable, outcome_variable
        )
        
        # Generate recommendations
        adjustment_set, alternatives = await self._recommend_adjustment_sets(
            confounders, colliders, data, treatment_variable, outcome_variable, domain
        )
        
        # Assess overall confounding risk
        confounding_risk = self._assess_overall_confounding_risk(confounders, data)
        
        # Generate validation suggestions
        validation_suggestions = self._generate_validation_suggestions(
            confounders, data, treatment_variable, outcome_variable
        )
        
        summary = ConfounderDetectionSummary(
            treatment_variable=treatment_variable,
            outcome_variable=outcome_variable,
            detected_confounders=confounders,
            colliders_detected=colliders,
            mediators_detected=mediators,
            instrumental_variables=instrumental,
            overall_confounding_risk=confounding_risk,
            recommended_adjustment_set=adjustment_set,
            alternative_adjustment_sets=alternatives,
            validation_suggestions=validation_suggestions
        )
        
        self.logger.info(f"Confounder detection completed. Found {len(confounders)} potential confounders.")
        return summary
    
    async def _hybrid_ensemble_detection(self,
                                       data: pd.DataFrame,
                                       treatment: str,
                                       outcome: str,
                                       candidates: List[str],
                                       descriptions: Optional[Dict[str, str]],
                                       domain: str,
                                       context: str) -> List[ConfounderDetectionResult]:
        """Use ensemble of methods for robust detection."""
        
        self.logger.info("Running hybrid ensemble confounder detection")
        
        all_results = []
        
        # 1. Statistical association detection
        stat_results = await self._statistical_association_detection(
            data, treatment, outcome, candidates
        )
        all_results.extend(stat_results)
        
        # 2. Mutual information detection
        mi_results = await self._mutual_information_detection(
            data, treatment, outcome, candidates
        )
        all_results.extend(mi_results)
        
        # 3. Random Forest importance detection
        rf_results = await self._random_forest_detection(
            data, treatment, outcome, candidates
        )
        all_results.extend(rf_results)
        
        # 4. LLM reasoning detection
        if descriptions:
            llm_results = await self._llm_reasoning_detection(
                data, treatment, outcome, candidates, descriptions, domain, context
            )
            all_results.extend(llm_results)
        
        # 5. Domain knowledge detection
        domain_results = await self._domain_knowledge_detection(
            data, treatment, outcome, candidates, domain
        )
        all_results.extend(domain_results)
        
        return all_results
    
    async def _statistical_association_detection(self,
                                               data: pd.DataFrame,
                                               treatment: str,
                                               outcome: str,
                                               candidates: List[str]) -> List[ConfounderDetectionResult]:
        """Detect confounders using statistical associations."""
        
        results = []
        
        for candidate in candidates:
            if candidate not in data.columns:
                continue
            
            # Check association with treatment
            treatment_association = self._calculate_association(
                data[candidate], data[treatment]
            )
            
            # Check association with outcome
            outcome_association = self._calculate_association(
                data[candidate], data[outcome]
            )
            
            # Both associations must be significant for confounding
            if (treatment_association['significant'] and 
                outcome_association['significant']):
                
                confounding_score = (
                    treatment_association['strength'] + 
                    outcome_association['strength']
                ) / 2
                
                result = ConfounderDetectionResult(
                    variable_name=candidate,
                    affects_treatment_score=treatment_association['strength'],
                    affects_outcome_score=outcome_association['strength'],
                    confounding_score=confounding_score,
                    detection_method=DetectionMethod.STATISTICAL_ASSOCIATION,
                    evidence_type=[ConfounderEvidence.STATISTICAL_CORRELATION],
                    statistical_evidence={
                        'treatment_association': treatment_association,
                        'outcome_association': outcome_association
                    },
                    reasoning=f"Statistically significant association with both treatment (p={treatment_association['p_value']:.3f}) and outcome (p={outcome_association['p_value']:.3f})",
                    confidence=min(confounding_score + 0.1, 1.0),
                    recommended_adjustment=[AdjustmentStrategy.REGRESSION_ADJUSTMENT]
                )
                
                results.append(result)
        
        return results
    
    def _calculate_association(self, var1: pd.Series, var2: pd.Series) -> Dict[str, Any]:
        """Calculate association between two variables."""
        
        # Remove missing values
        clean_data = pd.DataFrame({'var1': var1, 'var2': var2}).dropna()
        
        if len(clean_data) < 10:
            return {'strength': 0.0, 'p_value': 1.0, 'significant': False}
        
        var1_clean = clean_data['var1']
        var2_clean = clean_data['var2']
        
        # Determine variable types
        var1_numeric = pd.api.types.is_numeric_dtype(var1_clean)
        var2_numeric = pd.api.types.is_numeric_dtype(var2_clean)
        
        try:
            if var1_numeric and var2_numeric:
                # Both continuous: Pearson correlation
                corr, p_val = stats.pearsonr(var1_clean, var2_clean)
                return {
                    'strength': abs(corr),
                    'p_value': p_val,
                    'significant': p_val < self.significance_threshold,
                    'test': 'pearson_correlation'
                }
            elif not var1_numeric and not var2_numeric:
                # Both categorical: Chi-square test
                contingency = pd.crosstab(var1_clean, var2_clean)
                chi2, p_val, _, _ = stats.chi2_contingency(contingency)
                
                # Cramer's V for effect size
                n = contingency.sum().sum()
                cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
                
                return {
                    'strength': cramers_v,
                    'p_value': p_val,
                    'significant': p_val < self.significance_threshold,
                    'test': 'chi_square'
                }
            else:
                # One continuous, one categorical: ANOVA/t-test
                if var1_numeric:
                    continuous, categorical = var1_clean, var2_clean
                else:
                    continuous, categorical = var2_clean, var1_clean
                
                groups = [continuous[categorical == cat] for cat in categorical.unique()]
                groups = [g for g in groups if len(g) > 0]
                
                if len(groups) >= 2:
                    f_stat, p_val = stats.f_oneway(*groups)
                    
                    # Eta-squared for effect size
                    ss_between = sum(len(g) * (g.mean() - continuous.mean())**2 for g in groups)
                    ss_total = ((continuous - continuous.mean())**2).sum()
                    eta_squared = ss_between / ss_total if ss_total > 0 else 0
                    
                    return {
                        'strength': eta_squared,
                        'p_value': p_val,
                        'significant': p_val < self.significance_threshold,
                        'test': 'anova'
                    }
        
        except Exception as e:
            self.logger.warning(f"Association calculation failed: {e}")
        
        return {'strength': 0.0, 'p_value': 1.0, 'significant': False}
    
    async def _mutual_information_detection(self,
                                          data: pd.DataFrame,
                                          treatment: str,
                                          outcome: str,
                                          candidates: List[str]) -> List[ConfounderDetectionResult]:
        """Detect confounders using mutual information."""
        
        results = []
        
        # Prepare data
        clean_data = data[[treatment, outcome] + candidates].dropna()
        
        if len(clean_data) < 20:
            return results
        
        # Determine if variables are continuous or discrete
        treatment_continuous = pd.api.types.is_numeric_dtype(clean_data[treatment])
        outcome_continuous = pd.api.types.is_numeric_dtype(clean_data[outcome])
        
        for candidate in candidates:
            if candidate not in clean_data.columns:
                continue
                
            try:
                candidate_continuous = pd.api.types.is_numeric_dtype(clean_data[candidate])
                
                # Calculate mutual information with treatment
                if treatment_continuous:
                    mi_treatment = mutual_info_regression(
                        clean_data[[candidate]], clean_data[treatment]
                    )[0]
                else:
                    mi_treatment = mutual_info_classif(
                        clean_data[[candidate]], clean_data[treatment]
                    )[0]
                
                # Calculate mutual information with outcome
                if outcome_continuous:
                    mi_outcome = mutual_info_regression(
                        clean_data[[candidate]], clean_data[outcome]
                    )[0]
                else:
                    mi_outcome = mutual_info_classif(
                        clean_data[[candidate]], clean_data[outcome]
                    )[0]
                
                # Check if both mutual informations are significant
                if mi_treatment > self.mutual_info_threshold and mi_outcome > self.mutual_info_threshold:
                    confounding_score = (mi_treatment + mi_outcome) / 2
                    
                    result = ConfounderDetectionResult(
                        variable_name=candidate,
                        affects_treatment_score=mi_treatment,
                        affects_outcome_score=mi_outcome,
                        confounding_score=confounding_score,
                        detection_method=DetectionMethod.MUTUAL_INFORMATION,
                        evidence_type=[ConfounderEvidence.DATA_DRIVEN_DISCOVERY],
                        statistical_evidence={
                            'mi_treatment': mi_treatment,
                            'mi_outcome': mi_outcome
                        },
                        reasoning=f"High mutual information with both treatment ({mi_treatment:.3f}) and outcome ({mi_outcome:.3f})",
                        confidence=min(confounding_score + 0.2, 1.0),
                        recommended_adjustment=[AdjustmentStrategy.REGRESSION_ADJUSTMENT]
                    )
                    
                    results.append(result)
            
            except Exception as e:
                self.logger.warning(f"Mutual information calculation failed for {candidate}: {e}")
                continue
        
        return results
    
    async def _random_forest_detection(self,
                                     data: pd.DataFrame,
                                     treatment: str,
                                     outcome: str,
                                     candidates: List[str]) -> List[ConfounderDetectionResult]:
        """Detect confounders using Random Forest feature importance."""
        
        results = []
        
        # Prepare data
        clean_data = data[[treatment, outcome] + candidates].dropna()
        
        if len(clean_data) < 20:
            return results
        
        # Encode categorical variables
        encoded_data = clean_data.copy()
        categorical_vars = []
        
        for col in encoded_data.columns:
            if not pd.api.types.is_numeric_dtype(encoded_data[col]):
                encoded_data[col] = pd.Categorical(encoded_data[col]).codes
                categorical_vars.append(col)
        
        try:
            # Random Forest for treatment prediction
            treatment_continuous = pd.api.types.is_numeric_dtype(data[treatment])
            
            if treatment_continuous:
                rf_treatment = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                rf_treatment = RandomForestClassifier(n_estimators=100, random_state=42)
            
            X_treatment = encoded_data[candidates]
            y_treatment = encoded_data[treatment]
            
            rf_treatment.fit(X_treatment, y_treatment)
            treatment_importances = rf_treatment.feature_importances_
            
            # Random Forest for outcome prediction
            outcome_continuous = pd.api.types.is_numeric_dtype(data[outcome])
            
            if outcome_continuous:
                rf_outcome = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                rf_outcome = RandomForestClassifier(n_estimators=100, random_state=42)
            
            X_outcome = encoded_data[candidates]
            y_outcome = encoded_data[outcome]
            
            rf_outcome.fit(X_outcome, y_outcome)
            outcome_importances = rf_outcome.feature_importances_
            
            # Find variables important for both
            for i, candidate in enumerate(candidates):
                treatment_importance = treatment_importances[i]
                outcome_importance = outcome_importances[i]
                
                if (treatment_importance > self.importance_threshold and 
                    outcome_importance > self.importance_threshold):
                    
                    confounding_score = (treatment_importance + outcome_importance) / 2
                    
                    result = ConfounderDetectionResult(
                        variable_name=candidate,
                        affects_treatment_score=treatment_importance,
                        affects_outcome_score=outcome_importance,
                        confounding_score=confounding_score,
                        detection_method=DetectionMethod.RANDOM_FOREST_IMPORTANCE,
                        evidence_type=[ConfounderEvidence.DATA_DRIVEN_DISCOVERY],
                        statistical_evidence={
                            'rf_treatment_importance': treatment_importance,
                            'rf_outcome_importance': outcome_importance
                        },
                        reasoning=f"High Random Forest importance for both treatment ({treatment_importance:.3f}) and outcome ({outcome_importance:.3f})",
                        confidence=min(confounding_score + 0.15, 1.0),
                        recommended_adjustment=[AdjustmentStrategy.REGRESSION_ADJUSTMENT]
                    )
                    
                    results.append(result)
        
        except Exception as e:
            self.logger.error(f"Random Forest detection failed: {e}")
        
        return results
    
    async def _llm_reasoning_detection(self,
                                     data: pd.DataFrame,
                                     treatment: str,
                                     outcome: str,
                                     candidates: List[str],
                                     descriptions: Dict[str, str],
                                     domain: str,
                                     context: str) -> List[ConfounderDetectionResult]:
        """Use LLM reasoning for confounder detection."""
        
        # Use existing LLM confounder reasoning module
        llm_confounders = await self.llm_confounder_reasoner.identify_potential_confounders(
            treatment, outcome, descriptions, domain, context
        )
        
        results = []
        
        for llm_confounder in llm_confounders:
            if llm_confounder.variable_name in candidates:
                
                # Convert to detection result format
                result = ConfounderDetectionResult(
                    variable_name=llm_confounder.variable_name,
                    affects_treatment_score=0.8 if llm_confounder.affects_treatment else 0.2,
                    affects_outcome_score=0.8 if llm_confounder.affects_outcome else 0.2,
                    confounding_score=self._strength_to_score(llm_confounder.strength),
                    detection_method=DetectionMethod.LLM_REASONING,
                    evidence_type=[ConfounderEvidence.DOMAIN_EXPERT_KNOWLEDGE],
                    statistical_evidence={},
                    reasoning=llm_confounder.reasoning,
                    confidence=llm_confounder.strength.value in ["strong", "critical"] and 0.9 or 0.7,
                    recommended_adjustment=llm_confounder.adjustment_strategies
                )
                
                results.append(result)
        
        return results
    
    def _strength_to_score(self, strength: ConfounderStrength) -> float:
        """Convert confounder strength to numeric score."""
        mapping = {
            ConfounderStrength.WEAK: 0.3,
            ConfounderStrength.MODERATE: 0.6,
            ConfounderStrength.STRONG: 0.8,
            ConfounderStrength.CRITICAL: 0.95
        }
        return mapping.get(strength, 0.5)
    
    async def _domain_knowledge_detection(self,
                                        data: pd.DataFrame,
                                        treatment: str,
                                        outcome: str,
                                        candidates: List[str],
                                        domain: str) -> List[ConfounderDetectionResult]:
        """Detect confounders using domain knowledge patterns."""
        
        results = []
        
        if domain not in self.domain_confounder_patterns:
            return results
        
        domain_patterns = self.domain_confounder_patterns[domain]
        
        # Check always-check variables
        for always_check_var in domain_patterns.get("always_check", []):
            # Find matching variables (exact match or contains)
            matching_vars = [
                var for var in candidates 
                if (always_check_var.lower() in var.lower() or 
                    var.lower() in always_check_var.lower())
            ]
            
            for var in matching_vars:
                result = ConfounderDetectionResult(
                    variable_name=var,
                    affects_treatment_score=0.7,
                    affects_outcome_score=0.7,
                    confounding_score=0.8,
                    detection_method=DetectionMethod.DOMAIN_KNOWLEDGE,
                    evidence_type=[ConfounderEvidence.DOMAIN_EXPERT_KNOWLEDGE],
                    statistical_evidence={},
                    reasoning=f"Domain knowledge indicates {always_check_var} is typically a confounder in {domain}",
                    confidence=0.85,
                    recommended_adjustment=[
                        AdjustmentStrategy.CONTROL_DIRECTLY,
                        AdjustmentStrategy.STRATIFICATION
                    ]
                )
                
                results.append(result)
        
        return results
    
    async def _single_method_detection(self,
                                     data: pd.DataFrame,
                                     treatment: str,
                                     outcome: str,
                                     candidates: List[str],
                                     method: DetectionMethod,
                                     descriptions: Optional[Dict[str, str]],
                                     domain: str,
                                     context: str) -> List[ConfounderDetectionResult]:
        """Run a single detection method."""
        
        if method == DetectionMethod.STATISTICAL_ASSOCIATION:
            return await self._statistical_association_detection(data, treatment, outcome, candidates)
        elif method == DetectionMethod.MUTUAL_INFORMATION:
            return await self._mutual_information_detection(data, treatment, outcome, candidates)
        elif method == DetectionMethod.RANDOM_FOREST_IMPORTANCE:
            return await self._random_forest_detection(data, treatment, outcome, candidates)
        elif method == DetectionMethod.LLM_REASONING and descriptions:
            return await self._llm_reasoning_detection(data, treatment, outcome, candidates, descriptions, domain, context)
        elif method == DetectionMethod.DOMAIN_KNOWLEDGE:
            return await self._domain_knowledge_detection(data, treatment, outcome, candidates, domain)
        else:
            return []
    
    def _aggregate_detection_results(self, 
                                   results: List[ConfounderDetectionResult]) -> List[ConfounderDetectionResult]:
        """Aggregate results from multiple detection methods."""
        
        # Group by variable name
        variable_results = defaultdict(list)
        for result in results:
            variable_results[result.variable_name].append(result)
        
        aggregated = []
        
        for var_name, var_results in variable_results.items():
            # Combine scores and evidence
            treatment_scores = [r.affects_treatment_score for r in var_results]
            outcome_scores = [r.affects_outcome_score for r in var_results]
            confounding_scores = [r.confounding_score for r in var_results]
            confidences = [r.confidence for r in var_results]
            
            # Aggregate scores (weighted average by confidence)
            total_confidence = sum(confidences)
            if total_confidence > 0:
                agg_treatment_score = sum(s * c for s, c in zip(treatment_scores, confidences)) / total_confidence
                agg_outcome_score = sum(s * c for s, c in zip(outcome_scores, confidences)) / total_confidence
                agg_confounding_score = sum(s * c for s, c in zip(confounding_scores, confidences)) / total_confidence
            else:
                agg_treatment_score = np.mean(treatment_scores)
                agg_outcome_score = np.mean(outcome_scores)
                agg_confounding_score = np.mean(confounding_scores)
            
            # Combine evidence types
            all_evidence_types = []
            all_methods = []
            all_statistical_evidence = {}
            all_reasoning = []
            all_adjustments = []
            
            for result in var_results:
                all_evidence_types.extend(result.evidence_type)
                all_methods.append(result.detection_method.value)
                all_statistical_evidence.update(result.statistical_evidence)
                all_reasoning.append(result.reasoning)
                all_adjustments.extend(result.recommended_adjustment)
            
            # Create aggregated result
            aggregated_result = ConfounderDetectionResult(
                variable_name=var_name,
                affects_treatment_score=agg_treatment_score,
                affects_outcome_score=agg_outcome_score,
                confounding_score=agg_confounding_score,
                detection_method=DetectionMethod.HYBRID_ENSEMBLE,
                evidence_type=list(set(all_evidence_types)),
                statistical_evidence=all_statistical_evidence,
                reasoning=f"Multiple methods detected confounding: {'; '.join(set(all_reasoning))}",
                confidence=min(max(confidences) + 0.1, 1.0),  # Boost confidence for multiple methods
                recommended_adjustment=list(set(all_adjustments))
            )
            
            aggregated.append(aggregated_result)
        
        # Sort by confounding score
        return sorted(aggregated, key=lambda x: x.confounding_score, reverse=True)
    
    def _classify_detected_variables(self,
                                   results: List[ConfounderDetectionResult],
                                   data: pd.DataFrame,
                                   treatment: str,
                                   outcome: str) -> Tuple[List[ConfounderDetectionResult], 
                                                          List[ConfounderDetectionResult],
                                                          List[ConfounderDetectionResult],
                                                          List[ConfounderDetectionResult]]:
        """Classify detected variables into confounders, colliders, mediators, and instruments."""
        
        confounders = []
        colliders = []
        mediators = []
        instrumental = []
        
        # Simple classification based on scores
        # In practice, this would use more sophisticated causal discovery
        
        for result in results:
            # Confounder: affects both treatment and outcome
            if (result.affects_treatment_score > 0.3 and 
                result.affects_outcome_score > 0.3):
                confounders.append(result)
            
            # Note: Proper collider and mediator detection would require
            # more sophisticated analysis or temporal information
            # For now, we classify based on simple heuristics
            
            # Variables that might be colliders (affected by both treatment and outcome)
            # would need different statistical tests
            
            # Variables that might be mediators (on causal path)
            # would need mediation analysis
        
        return confounders, colliders, mediators, instrumental
    
    async def _recommend_adjustment_sets(self,
                                       confounders: List[ConfounderDetectionResult],
                                       colliders: List[ConfounderDetectionResult],
                                       data: pd.DataFrame,
                                       treatment: str,
                                       outcome: str,
                                       domain: str) -> Tuple[List[str], List[List[str]]]:
        """Recommend adjustment sets for causal estimation."""
        
        # Primary recommendation: strong confounders
        strong_confounders = [
            c.variable_name for c in confounders 
            if c.confounding_score > 0.6
        ]
        
        # Alternative sets
        alternatives = []
        
        # All detected confounders
        all_confounders = [c.variable_name for c in confounders]
        if len(all_confounders) != len(strong_confounders):
            alternatives.append(all_confounders)
        
        # Top N confounders
        top_n_confounders = [c.variable_name for c in confounders[:5]]
        if top_n_confounders != strong_confounders:
            alternatives.append(top_n_confounders)
        
        # Domain-specific recommendations
        if domain in self.domain_confounder_patterns:
            domain_essentials = []
            for essential in self.domain_confounder_patterns[domain].get("always_check", []):
                matching = [c.variable_name for c in confounders 
                           if essential.lower() in c.variable_name.lower()]
                domain_essentials.extend(matching)
            
            if domain_essentials and domain_essentials != strong_confounders:
                alternatives.append(domain_essentials)
        
        return strong_confounders, alternatives[:3]  # Top 3 alternatives
    
    def _assess_overall_confounding_risk(self,
                                       confounders: List[ConfounderDetectionResult],
                                       data: pd.DataFrame) -> str:
        """Assess overall risk of confounding bias."""
        
        if not confounders:
            return "Low - No major confounders detected"
        
        max_score = max(c.confounding_score for c in confounders)
        num_strong_confounders = sum(1 for c in confounders if c.confounding_score > 0.7)
        
        if max_score > 0.8 or num_strong_confounders >= 3:
            return "High - Multiple strong confounders detected"
        elif max_score > 0.6 or num_strong_confounders >= 1:
            return "Moderate - Some confounders detected"
        else:
            return "Low to Moderate - Weak confounders detected"
    
    def _generate_validation_suggestions(self,
                                       confounders: List[ConfounderDetectionResult],
                                       data: pd.DataFrame,
                                       treatment: str,
                                       outcome: str) -> List[str]:
        """Generate suggestions for validating confounder detection."""
        
        suggestions = []
        
        # Data collection suggestions
        if len(confounders) > 0:
            suggestions.append("Collect additional data on identified confounders to improve measurement")
        
        # Statistical validation
        suggestions.append("Test robustness of results across different adjustment sets")
        
        if len(confounders) >= 2:
            suggestions.append("Perform sensitivity analysis for unmeasured confounding")
        
        # Domain validation
        suggestions.append("Validate identified confounders with domain experts")
        
        # Instrumental variables
        suggestions.append("Consider instrumental variable analysis if valid instruments available")
        
        # Temporal validation
        if "time" in data.columns or "date" in data.columns:
            suggestions.append("Use temporal information to validate causal ordering")
        
        return suggestions


# Convenience functions
def create_confounder_detector(llm_client) -> AutomaticConfounderDetector:
    """Create an automatic confounder detector."""
    return AutomaticConfounderDetector(llm_client)


async def detect_confounders_automatically(data: pd.DataFrame,
                                         treatment_variable: str,
                                         outcome_variable: str,
                                         llm_client,
                                         variable_descriptions: Optional[Dict[str, str]] = None,
                                         domain: str = "general") -> ConfounderDetectionSummary:
    """Quick function to detect confounders automatically."""
    
    detector = create_confounder_detector(llm_client)
    return await detector.detect_confounders(
        data, treatment_variable, outcome_variable, 
        variable_descriptions, domain
    )