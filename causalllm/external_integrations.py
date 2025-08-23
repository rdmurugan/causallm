"""
Integration with external causal inference libraries for Tier 2 capabilities.

This module provides seamless integration with popular causal inference libraries
including DoWhy, EconML, CausalML, pgmpy, and others, while maintaining
CausalLLM's LLM-guided approach and unified interface.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import asyncio
import json
import warnings
import importlib
from pathlib import Path

from causalllm.logging import get_logger


class ExternalLibrary(Enum):
    """Supported external causal inference libraries."""
    DOWHY = "dowhy"
    ECONML = "econml"
    CAUSALML = "causalml"
    PGMPY = "pgmpy"
    NETWORKX = "networkx"
    SCIKIT_LEARN = "sklearn"
    SCIPY = "scipy"
    STATSMODELS = "statsmodels"


class IntegrationMethod(Enum):
    """Methods for integrating with external libraries."""
    WRAP_ESTIMATOR = "wrap_estimator"        # Wrap external estimator
    IMPORT_GRAPH = "import_graph"           # Import graph structure
    EXPORT_DATA = "export_data"             # Export data for external use
    HYBRID_ANALYSIS = "hybrid_analysis"     # Combine CausalLLM + external
    VALIDATION = "validation"               # Use external lib for validation


@dataclass
class LibraryCapabilities:
    """Capabilities of an external library."""
    
    library_name: str
    available: bool
    version: Optional[str] = None
    supported_methods: List[str] = field(default_factory=list)
    data_requirements: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)


@dataclass
class IntegrationResult:
    """Result from external library integration."""
    
    library_used: ExternalLibrary
    method_used: IntegrationMethod
    results: Dict[str, Any]
    causalllm_analysis: Optional[Dict[str, Any]] = None
    combined_insights: List[str] = field(default_factory=list)
    validation_scores: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    integration_metadata: Dict[str, Any] = field(default_factory=dict)


class ExternalLibraryIntegrator(ABC):
    """Abstract base class for external library integrations."""
    
    def __init__(self, library: ExternalLibrary):
        self.library = library
        self.logger = get_logger(f"causalllm.external_integrations.{library.value}")
        self.capabilities = None
        self._check_availability()
    
    @abstractmethod
    def _check_availability(self) -> LibraryCapabilities:
        """Check if external library is available and gather capabilities."""
        pass
    
    @abstractmethod
    async def integrate(self, data: pd.DataFrame,
                       variables: Dict[str, str],
                       method: IntegrationMethod,
                       **kwargs) -> IntegrationResult:
        """Perform integration with external library."""
        pass


class DoWhyIntegrator(ExternalLibraryIntegrator):
    """Integration with Microsoft DoWhy causal inference library."""
    
    def __init__(self):
        super().__init__(ExternalLibrary.DOWHY)
        self.dowhy = None
        self.causal_model = None
    
    def _check_availability(self) -> LibraryCapabilities:
        """Check DoWhy availability and capabilities."""
        try:
            import dowhy
            from dowhy import CausalModel
            self.dowhy = dowhy
            
            self.capabilities = LibraryCapabilities(
                library_name="DoWhy",
                available=True,
                version=dowhy.__version__,
                supported_methods=[
                    "backdoor_adjustment",
                    "instrumental_variables", 
                    "regression_discontinuity",
                    "difference_in_differences",
                    "propensity_score_matching"
                ],
                data_requirements=["treatment_variable", "outcome_variable", "confounders"],
                strengths=[
                    "Formal causal inference framework",
                    "Multiple identification methods",
                    "Assumption validation",
                    "Refutation tests"
                ],
                limitations=[
                    "Requires domain knowledge for graph specification",
                    "Limited automated discovery",
                    "Assumes correct causal graph"
                ]
            )
            
            self.logger.info(f"DoWhy {dowhy.__version__} available")
            
        except ImportError:
            self.capabilities = LibraryCapabilities(
                library_name="DoWhy",
                available=False,
                limitations=["DoWhy not installed"]
            )
            self.logger.warning("DoWhy not available - install with: pip install dowhy")
        
        return self.capabilities
    
    async def integrate(self, data: pd.DataFrame,
                       variables: Dict[str, str],
                       method: IntegrationMethod,
                       treatment_variable: str = None,
                       outcome_variable: str = None,
                       confounders: List[str] = None,
                       instruments: List[str] = None,
                       **kwargs) -> IntegrationResult:
        """Integrate with DoWhy for causal analysis."""
        
        if not self.capabilities.available:
            raise RuntimeError("DoWhy is not available")
        
        self.logger.info(f"DoWhy integration using method: {method.value}")
        
        try:
            if method == IntegrationMethod.WRAP_ESTIMATOR:
                return await self._wrap_dowhy_estimator(
                    data, variables, treatment_variable, outcome_variable, confounders, **kwargs
                )
            elif method == IntegrationMethod.VALIDATION:
                return await self._validate_with_dowhy(
                    data, variables, treatment_variable, outcome_variable, confounders, **kwargs
                )
            elif method == IntegrationMethod.HYBRID_ANALYSIS:
                return await self._hybrid_dowhy_analysis(
                    data, variables, treatment_variable, outcome_variable, confounders, **kwargs
                )
            else:
                raise ValueError(f"Integration method {method.value} not supported for DoWhy")
        
        except Exception as e:
            self.logger.error(f"DoWhy integration failed: {e}")
            raise
    
    async def _wrap_dowhy_estimator(self, data: pd.DataFrame,
                                  variables: Dict[str, str],
                                  treatment_variable: str,
                                  outcome_variable: str,
                                  confounders: List[str],
                                  **kwargs) -> IntegrationResult:
        """Wrap DoWhy causal estimation."""
        
        # Build causal graph
        graph = self._build_causal_graph(variables, treatment_variable, outcome_variable, confounders)
        
        # Create DoWhy causal model
        from dowhy import CausalModel
        
        model = CausalModel(
            data=data,
            treatment=treatment_variable,
            outcome=outcome_variable,
            graph=graph,
            **kwargs
        )
        
        # Identify causal effect
        identification = model.identify_effect(proceed_when_unidentifiable=True)
        
        # Estimate causal effect
        estimation_method = kwargs.get('estimation_method', 'backdoor.linear_regression')
        causal_estimate = model.estimate_effect(identification, method_name=estimation_method)
        
        # Run refutation tests
        refutation_results = []
        refutation_methods = ['random_common_cause', 'placebo_treatment_refuter', 'data_subset_refuter']
        
        for refute_method in refutation_methods:
            try:
                refutation = model.refute_estimate(identification, causal_estimate, method_name=refute_method)
                refutation_results.append({
                    "method": refute_method,
                    "estimate": refutation.new_effect,
                    "p_value": getattr(refutation, 'p_value', None)
                })
            except Exception as e:
                self.logger.warning(f"Refutation {refute_method} failed: {e}")
        
        # Compile results
        results = {
            "causal_estimate": float(causal_estimate.value),
            "confidence_intervals": causal_estimate.confidence_intervals if hasattr(causal_estimate, 'confidence_intervals') else None,
            "identification_method": str(identification),
            "estimation_method": estimation_method,
            "refutation_tests": refutation_results,
            "model_assumptions": self._extract_assumptions(model)
        }
        
        # Generate recommendations
        recommendations = []
        if abs(causal_estimate.value) > 0.1:
            recommendations.append(f"Strong causal effect detected: {causal_estimate.value:.3f}")
        
        # Check refutation robustness
        robust_refutations = sum(1 for r in refutation_results if abs(r["estimate"]) < 0.1)
        if robust_refutations >= len(refutation_results) * 0.7:
            recommendations.append("Causal estimate appears robust to refutation tests")
        else:
            recommendations.append("Causal estimate may be sensitive to unobserved confounders")
        
        return IntegrationResult(
            library_used=ExternalLibrary.DOWHY,
            method_used=method,
            results=results,
            recommendations=recommendations,
            integration_metadata={
                "dowhy_version": self.capabilities.version,
                "refutation_tests_passed": robust_refutations,
                "total_refutation_tests": len(refutation_results)
            }
        )
    
    async def _validate_with_dowhy(self, data: pd.DataFrame,
                                 variables: Dict[str, str],
                                 treatment_variable: str,
                                 outcome_variable: str,
                                 confounders: List[str],
                                 causalllm_results: Dict[str, Any] = None,
                                 **kwargs) -> IntegrationResult:
        """Use DoWhy to validate CausalLLM results."""
        
        # Get DoWhy estimate
        dowhy_result = await self._wrap_dowhy_estimator(
            data, variables, treatment_variable, outcome_variable, confounders, **kwargs
        )
        
        validation_scores = {}
        combined_insights = []
        
        if causalllm_results:
            # Compare estimates
            dowhy_estimate = dowhy_result.results["causal_estimate"]
            causalllm_estimate = causalllm_results.get("causal_estimate", 0)
            
            # Calculate agreement
            if abs(dowhy_estimate) > 0.01 and abs(causalllm_estimate) > 0.01:
                agreement = 1 - abs(dowhy_estimate - causalllm_estimate) / max(abs(dowhy_estimate), abs(causalllm_estimate))
                validation_scores["estimate_agreement"] = max(0, agreement)
                
                if agreement > 0.7:
                    combined_insights.append("DoWhy and CausalLLM estimates are in good agreement")
                else:
                    combined_insights.append("DoWhy and CausalLLM estimates show some disagreement")
            
            # Direction agreement
            direction_agreement = np.sign(dowhy_estimate) == np.sign(causalllm_estimate)
            validation_scores["direction_agreement"] = float(direction_agreement)
            
            if direction_agreement:
                combined_insights.append("Both methods agree on effect direction")
            else:
                combined_insights.append("Methods disagree on effect direction - requires investigation")
        
        return IntegrationResult(
            library_used=ExternalLibrary.DOWHY,
            method_used=IntegrationMethod.VALIDATION,
            results=dowhy_result.results,
            causalllm_analysis=causalllm_results,
            combined_insights=combined_insights,
            validation_scores=validation_scores,
            recommendations=dowhy_result.recommendations
        )
    
    async def _hybrid_dowhy_analysis(self, data: pd.DataFrame,
                                   variables: Dict[str, str], 
                                   treatment_variable: str,
                                   outcome_variable: str,
                                   confounders: List[str],
                                   llm_client=None,
                                   **kwargs) -> IntegrationResult:
        """Combine DoWhy analysis with LLM insights."""
        
        # Get DoWhy results
        dowhy_result = await self._wrap_dowhy_estimator(
            data, variables, treatment_variable, outcome_variable, confounders, **kwargs
        )
        
        # Generate LLM interpretation if client provided
        combined_insights = []
        if llm_client:
            llm_interpretation = await self._generate_llm_interpretation(
                dowhy_result.results, variables, llm_client
            )
            combined_insights.extend(llm_interpretation)
        
        return IntegrationResult(
            library_used=ExternalLibrary.DOWHY,
            method_used=IntegrationMethod.HYBRID_ANALYSIS,
            results=dowhy_result.results,
            combined_insights=combined_insights,
            recommendations=dowhy_result.recommendations,
            integration_metadata=dowhy_result.integration_metadata
        )
    
    def _build_causal_graph(self, variables: Dict[str, str],
                          treatment: str, outcome: str,
                          confounders: List[str]) -> str:
        """Build causal graph specification for DoWhy."""
        
        graph_edges = []
        
        # Add treatment -> outcome edge
        graph_edges.append(f"{treatment} -> {outcome}")
        
        # Add confounder edges
        if confounders:
            for confounder in confounders:
                graph_edges.append(f"{confounder} -> {treatment}")
                graph_edges.append(f"{confounder} -> {outcome}")
        
        graph_str = "digraph { " + "; ".join(graph_edges) + " }"
        return graph_str
    
    def _extract_assumptions(self, model) -> List[str]:
        """Extract key assumptions from DoWhy model."""
        assumptions = [
            "No unobserved confounders (given the specified graph)",
            "Stable Unit Treatment Value Assumption (SUTVA)",
            "Positivity: all units have non-zero probability of receiving treatment"
        ]
        
        return assumptions
    
    async def _generate_llm_interpretation(self, dowhy_results: Dict[str, Any],
                                         variables: Dict[str, str],
                                         llm_client) -> List[str]:
        """Generate LLM interpretation of DoWhy results."""
        
        prompt = f"""
        Interpret these DoWhy causal analysis results in plain language:
        
        RESULTS:
        - Causal estimate: {dowhy_results['causal_estimate']}
        - Estimation method: {dowhy_results['estimation_method']}
        - Refutation tests: {len(dowhy_results['refutation_tests'])} tests performed
        
        VARIABLES:
        {json.dumps(variables, indent=2)}
        
        Provide 2-3 key insights about:
        1. What this causal estimate means
        2. How robust the finding appears to be
        3. What practical implications this has
        
        Keep explanations accessible and actionable.
        """
        
        try:
            if hasattr(llm_client, 'generate_response'):
                response = await llm_client.generate_response(prompt)
            else:
                response = await asyncio.to_thread(llm_client.generate, prompt)
            
            # Split response into insights
            insights = [line.strip() for line in response.split('\n') if line.strip() and not line.startswith('#')]
            return insights[:3]  # Take first 3 insights
            
        except Exception as e:
            self.logger.warning(f"LLM interpretation failed: {e}")
            return ["DoWhy analysis completed - see numerical results for details"]


class EconMLIntegrator(ExternalLibraryIntegrator):
    """Integration with Microsoft EconML library."""
    
    def __init__(self):
        super().__init__(ExternalLibrary.ECONML)
    
    def _check_availability(self) -> LibraryCapabilities:
        """Check EconML availability."""
        try:
            import econml
            
            self.capabilities = LibraryCapabilities(
                library_name="EconML",
                available=True,
                version=econml.__version__,
                supported_methods=[
                    "double_ml",
                    "causal_forest",
                    "dr_learner",
                    "iv_regression",
                    "meta_learners"
                ],
                strengths=[
                    "Machine learning for causal inference",
                    "Heterogeneous treatment effects",
                    "High-dimensional data handling"
                ]
            )
            
        except ImportError:
            self.capabilities = LibraryCapabilities(
                library_name="EconML",
                available=False,
                limitations=["EconML not installed"]
            )
        
        return self.capabilities
    
    async def integrate(self, data: pd.DataFrame,
                       variables: Dict[str, str],
                       method: IntegrationMethod,
                       **kwargs) -> IntegrationResult:
        """Integrate with EconML."""
        
        if not self.capabilities.available:
            raise RuntimeError("EconML is not available")
        
        # Placeholder implementation
        return IntegrationResult(
            library_used=ExternalLibrary.ECONML,
            method_used=method,
            results={"status": "EconML integration not yet implemented"},
            recommendations=["Install EconML: pip install econml"]
        )


class UniversalExternalIntegrator:
    """Universal integrator that manages all external library integrations."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.logger = get_logger("causalllm.universal_external_integrator")
        
        # Initialize integrators
        self.integrators = {
            ExternalLibrary.DOWHY: DoWhyIntegrator(),
            ExternalLibrary.ECONML: EconMLIntegrator(),
            # Add more integrators as implemented
        }
        
        # Check what's available
        self.available_libraries = self._check_available_libraries()
    
    def _check_available_libraries(self) -> Dict[ExternalLibrary, LibraryCapabilities]:
        """Check which external libraries are available."""
        available = {}
        
        for library, integrator in self.integrators.items():
            capabilities = integrator.capabilities
            available[library] = capabilities
            
            if capabilities.available:
                self.logger.info(f"{capabilities.library_name} {capabilities.version} is available")
            else:
                self.logger.info(f"{capabilities.library_name} is not available")
        
        return available
    
    async def integrate_with_library(self, library: ExternalLibrary,
                                   method: IntegrationMethod,
                                   data: pd.DataFrame,
                                   variables: Dict[str, str],
                                   **kwargs) -> IntegrationResult:
        """Integrate with a specific external library."""
        
        if library not in self.integrators:
            raise ValueError(f"Library {library.value} not supported")
        
        if not self.available_libraries[library].available:
            raise RuntimeError(f"Library {library.value} is not available")
        
        integrator = self.integrators[library]
        return await integrator.integrate(data, variables, method, **kwargs)
    
    async def recommend_best_library(self, data: pd.DataFrame,
                                   variables: Dict[str, str],
                                   analysis_goal: str,
                                   **kwargs) -> Tuple[ExternalLibrary, str]:
        """Recommend best external library for given analysis."""
        
        if not self.available_libraries:
            return None, "No external libraries available"
        
        available_libs = [lib for lib, caps in self.available_libraries.items() if caps.available]
        
        if not available_libs:
            return None, "No external libraries available"
        
        # Simple recommendation logic based on analysis goal
        if "treatment" in analysis_goal.lower() and "effect" in analysis_goal.lower():
            if ExternalLibrary.DOWHY in available_libs:
                return ExternalLibrary.DOWHY, "DoWhy excels at treatment effect estimation with formal causal inference"
            elif ExternalLibrary.ECONML in available_libs:
                return ExternalLibrary.ECONML, "EconML provides ML-based treatment effect estimation"
        
        # Default to first available
        return available_libs[0], f"Using {available_libs[0].value} as default"
    
    async def validate_causalllm_with_external(self, causalllm_results: Dict[str, Any],
                                             data: pd.DataFrame,
                                             variables: Dict[str, str],
                                             **kwargs) -> List[IntegrationResult]:
        """Validate CausalLLM results using multiple external libraries."""
        
        validation_results = []
        
        for library, capabilities in self.available_libraries.items():
            if not capabilities.available:
                continue
            
            try:
                result = await self.integrate_with_library(
                    library=library,
                    method=IntegrationMethod.VALIDATION,
                    data=data,
                    variables=variables,
                    causalllm_results=causalllm_results,
                    **kwargs
                )
                validation_results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Validation with {library.value} failed: {e}")
        
        return validation_results
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """Get summary of integration capabilities."""
        
        summary = {
            "total_libraries": len(self.integrators),
            "available_libraries": len([cap for cap in self.available_libraries.values() if cap.available]),
            "library_status": {}
        }
        
        for library, capabilities in self.available_libraries.items():
            summary["library_status"][library.value] = {
                "available": capabilities.available,
                "version": capabilities.version,
                "methods": capabilities.supported_methods,
                "strengths": capabilities.strengths
            }
        
        return summary
    
    async def run_comprehensive_analysis(self, data: pd.DataFrame,
                                       variables: Dict[str, str],
                                       treatment_variable: str,
                                       outcome_variable: str,
                                       **kwargs) -> Dict[ExternalLibrary, IntegrationResult]:
        """Run comprehensive analysis using all available libraries."""
        
        results = {}
        
        for library, capabilities in self.available_libraries.items():
            if not capabilities.available:
                continue
            
            try:
                result = await self.integrate_with_library(
                    library=library,
                    method=IntegrationMethod.WRAP_ESTIMATOR,
                    data=data,
                    variables=variables,
                    treatment_variable=treatment_variable,
                    outcome_variable=outcome_variable,
                    **kwargs
                )
                results[library] = result
                
            except Exception as e:
                self.logger.error(f"Analysis with {library.value} failed: {e}")
        
        return results
    
    async def generate_integration_report(self, results: Dict[ExternalLibrary, IntegrationResult],
                                        variables: Dict[str, str]) -> str:
        """Generate comprehensive report from multiple library results."""
        
        if not results:
            return "No external library results available for reporting."
        
        if not self.llm_client:
            # Generate simple text report without LLM
            report_lines = ["# External Library Integration Report\n"]
            
            for library, result in results.items():
                report_lines.append(f"## {library.value.title()} Results")
                report_lines.append(f"- Method: {result.method_used.value}")
                
                if "causal_estimate" in result.results:
                    estimate = result.results["causal_estimate"]
                    report_lines.append(f"- Causal Estimate: {estimate:.4f}")
                
                for rec in result.recommendations:
                    report_lines.append(f"- {rec}")
                
                report_lines.append("")
            
            return "\n".join(report_lines)
        
        # Generate LLM-enhanced report
        prompt = f"""
        Create a comprehensive causal analysis report comparing results from multiple external libraries.
        
        VARIABLES ANALYZED:
        {json.dumps(variables, indent=2)}
        
        LIBRARY RESULTS:
        """
        
        for library, result in results.items():
            prompt += f"""
            
        {library.value.upper()}:
        - Method: {result.method_used.value}
        - Results: {json.dumps(result.results, indent=2, default=str)}
        - Recommendations: {result.recommendations}
        """
        
        prompt += """
        
        Generate a report with:
        1. Executive Summary
        2. Comparison of Results 
        3. Convergence Analysis
        4. Methodological Strengths/Limitations
        5. Final Recommendations
        
        Focus on practical insights and actionable conclusions.
        """
        
        try:
            if hasattr(self.llm_client, 'generate_response'):
                report = await self.llm_client.generate_response(prompt)
            else:
                report = await asyncio.to_thread(self.llm_client.generate, prompt)
            
            return report
            
        except Exception as e:
            self.logger.error(f"LLM report generation failed: {e}")
            return "Report generation failed - see individual library results"


# Convenience functions
def create_external_integrator(llm_client=None) -> UniversalExternalIntegrator:
    """Create universal external library integrator."""
    return UniversalExternalIntegrator(llm_client)


async def integrate_external_library(library: ExternalLibrary,
                                    data: pd.DataFrame,
                                    variables: Dict[str, str],
                                    method: IntegrationMethod = IntegrationMethod.WRAP_ESTIMATOR,
                                    llm_client=None,
                                    **kwargs) -> IntegrationResult:
    """Quick function to integrate with external library."""
    integrator = create_external_integrator(llm_client)
    return await integrator.integrate_with_library(library, method, data, variables, **kwargs)


async def validate_with_external_libraries(causalllm_results: Dict[str, Any],
                                         data: pd.DataFrame,
                                         variables: Dict[str, str],
                                         llm_client=None,
                                         **kwargs) -> List[IntegrationResult]:
    """Validate CausalLLM results with external libraries."""
    integrator = create_external_integrator(llm_client)
    return await integrator.validate_causalllm_with_external(
        causalllm_results, data, variables, **kwargs
    )