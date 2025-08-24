"""
LLM-Enhanced Causal Discovery Agent

This module provides automated discovery of causal structures from data using LLM reasoning
combined with traditional causal discovery algorithms. It includes structure learning,
variable relationship analysis, and causal graph construction.
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
import networkx as nx
from scipy import stats
import warnings

from causalllm.logging import get_logger


class DiscoveryMethod(Enum):
    """Methods for causal discovery."""
    LLM_REASONING = "llm_reasoning"
    STATISTICAL_TESTS = "statistical_tests"
    CONSTRAINT_BASED = "constraint_based"
    SCORE_BASED = "score_based"
    HYBRID_LLM_STATISTICAL = "hybrid_llm_statistical"


class EdgeType(Enum):
    """Types of edges in causal graphs."""
    CAUSAL = "causal"           # X -> Y
    CONFOUNDING = "confounding" # X <-> Y
    SELECTION = "selection"     # X -- Y
    UNKNOWN = "unknown"         # X ? Y


class ConfidenceLevel(Enum):
    """Confidence levels for discovered relationships."""
    VERY_LOW = "very_low"      # 0.0-0.2
    LOW = "low"                # 0.2-0.4
    MODERATE = "moderate"       # 0.4-0.6
    HIGH = "high"              # 0.6-0.8
    VERY_HIGH = "very_high"    # 0.8-1.0


@dataclass
class CausalEdge:
    """Represents a causal relationship between variables."""
    
    source: str
    target: str
    edge_type: EdgeType
    confidence: float
    reasoning: str
    statistical_support: Dict[str, float]
    mechanism_description: str
    strength: str = "unknown"  # weak, moderate, strong
    evidence_sources: List[str] = field(default_factory=list)


@dataclass
class VariableAnalysis:
    """Analysis of a single variable's causal role."""
    
    variable_name: str
    description: str
    variable_type: str  # continuous, categorical, binary
    causal_role: str   # treatment, outcome, confounder, mediator, collider
    parents: List[str]
    children: List[str]
    reasoning: str
    confidence: float


@dataclass
class CausalStructure:
    """Complete causal structure discovered from data."""
    
    variables: List[VariableAnalysis]
    edges: List[CausalEdge]
    graph: nx.DiGraph
    discovery_method: DiscoveryMethod
    overall_confidence: float
    assumptions: List[str]
    limitations: List[str]
    alternative_structures: List['CausalStructure'] = field(default_factory=list)


class LLMCausalDiscoveryAgent:
    """LLM-enhanced automated causal discovery system."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.logger = get_logger("causalllm.llm_causal_discovery")
        
        # Discovery parameters
        self.min_correlation_threshold = 0.1
        self.significance_threshold = 0.05
        self.max_variables_for_exhaustive = 10
        
        # Domain-specific causal patterns
        self.domain_patterns = {
            "healthcare": {
                "common_causes": ["age", "gender", "socioeconomic_status"],
                "temporal_patterns": ["treatment -> outcome", "exposure -> disease"],
                "typical_confounders": ["comorbidities", "healthcare_access"]
            },
            "business": {
                "common_causes": ["market_conditions", "company_size"],
                "temporal_patterns": ["investment -> performance", "strategy -> outcome"],
                "typical_confounders": ["industry_effects", "economic_cycle"]
            },
            "education": {
                "common_causes": ["socioeconomic_background", "prior_achievement"],
                "temporal_patterns": ["intervention -> learning", "resource -> outcome"],
                "typical_confounders": ["school_quality", "family_support"]
            }
        }
    
    async def discover_causal_structure(self, 
                                      data: pd.DataFrame,
                                      variable_descriptions: Dict[str, str],
                                      domain: str = "general",
                                      context: str = "",
                                      method: DiscoveryMethod = DiscoveryMethod.HYBRID_LLM_STATISTICAL,
                                      target_variable: Optional[str] = None) -> CausalStructure:
        """
        Discover causal structure from data using LLM reasoning and statistical analysis.
        
        Args:
            data: Dataset for causal discovery
            variable_descriptions: Descriptions of variables
            domain: Domain context (healthcare, business, etc.)
            context: Additional context about the data/study
            method: Discovery method to use
            target_variable: Optional target variable to focus discovery on
            
        Returns:
            Discovered causal structure
        """
        self.logger.info("Starting LLM-enhanced causal discovery")
        
        # Step 1: Analyze individual variables
        variable_analyses = await self._analyze_variables(
            data, variable_descriptions, domain, context
        )
        
        # Step 2: Discover potential causal relationships
        if method == DiscoveryMethod.LLM_REASONING:
            edges = await self._llm_based_discovery(
                data, variable_analyses, domain, context, target_variable
            )
        elif method == DiscoveryMethod.STATISTICAL_TESTS:
            edges = await self._statistical_discovery(data, variable_analyses)
        elif method == DiscoveryMethod.HYBRID_LLM_STATISTICAL:
            edges = await self._hybrid_discovery(
                data, variable_analyses, domain, context, target_variable
            )
        else:
            edges = await self._constraint_based_discovery(data, variable_analyses)
        
        # Step 3: Build causal graph
        graph = self._build_causal_graph(variable_analyses, edges)
        
        # Step 4: Validate and refine structure
        validated_structure = await self._validate_structure(
            graph, edges, variable_analyses, data, domain, context
        )
        
        # Step 5: Generate alternative structures
        alternatives = await self._generate_alternative_structures(
            validated_structure, data, variable_analyses, domain
        )
        
        structure = CausalStructure(
            variables=variable_analyses,
            edges=edges,
            graph=graph,
            discovery_method=method,
            overall_confidence=self._calculate_overall_confidence(edges),
            assumptions=self._identify_assumptions(method, domain),
            limitations=self._identify_limitations(data, method),
            alternative_structures=alternatives[:3]  # Top 3 alternatives
        )
        
        self.logger.info(f"Causal discovery completed. Found {len(edges)} edges.")
        return structure
    
    async def _analyze_variables(self, 
                               data: pd.DataFrame,
                               descriptions: Dict[str, str],
                               domain: str,
                               context: str) -> List[VariableAnalysis]:
        """Analyze individual variables for their causal roles."""
        
        self.logger.info("Analyzing individual variables for causal roles")
        
        # Get basic statistics
        stats_summary = self._get_variable_statistics(data)
        
        # LLM analysis of variable roles
        variable_roles = await self._llm_analyze_variable_roles(
            data.columns.tolist(), descriptions, stats_summary, domain, context
        )
        
        analyses = []
        for var in data.columns:
            var_type = self._determine_variable_type(data[var])
            role_info = variable_roles.get(var, {})
            
            analysis = VariableAnalysis(
                variable_name=var,
                description=descriptions.get(var, ""),
                variable_type=var_type,
                causal_role=role_info.get("role", "unknown"),
                parents=[],  # Will be filled during edge discovery
                children=[],
                reasoning=role_info.get("reasoning", ""),
                confidence=role_info.get("confidence", 0.5)
            )
            analyses.append(analysis)
        
        return analyses
    
    def _get_variable_statistics(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Get basic statistics for each variable."""
        
        stats = {}
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                stats[col] = {
                    "type": "numeric",
                    "mean": data[col].mean(),
                    "std": data[col].std(),
                    "min": data[col].min(),
                    "max": data[col].max(),
                    "missing": data[col].isnull().sum(),
                    "unique_values": data[col].nunique()
                }
            else:
                stats[col] = {
                    "type": "categorical",
                    "unique_values": data[col].nunique(),
                    "most_common": data[col].mode().iloc[0] if len(data[col].mode()) > 0 else None,
                    "missing": data[col].isnull().sum()
                }
        
        return stats
    
    async def _llm_analyze_variable_roles(self,
                                        variables: List[str],
                                        descriptions: Dict[str, str],
                                        stats: Dict[str, Dict],
                                        domain: str,
                                        context: str) -> Dict[str, Dict]:
        """Use LLM to analyze the causal role of each variable."""
        
        # Build comprehensive prompt
        var_info = []
        for var in variables:
            desc = descriptions.get(var, "")
            stat_info = stats.get(var, {})
            var_info.append(f"- {var}: {desc} (Type: {stat_info.get('type', 'unknown')})")
        
        domain_guidance = ""
        if domain in self.domain_patterns:
            patterns = self.domain_patterns[domain]
            domain_guidance = f"""
            Domain-specific guidance for {domain}:
            - Common causes: {patterns['common_causes']}
            - Typical patterns: {patterns['temporal_patterns']}
            - Frequent confounders: {patterns['typical_confounders']}
            """
        
        prompt = f"""
        You are a causal inference expert analyzing variables to determine their likely causal roles.
        
        CONTEXT: {context}
        DOMAIN: {domain}
        
        VARIABLES:
        {chr(10).join(var_info)}
        
        {domain_guidance}
        
        For each variable, determine its most likely causal role:
        - treatment/intervention: Variables that can be manipulated or represent exposures
        - outcome: Variables that are results or endpoints of interest
        - confounder: Variables that affect both treatments and outcomes
        - mediator: Variables on the causal path between treatment and outcome
        - collider: Variables affected by multiple other variables
        - instrumental: Variables that affect treatment but not outcome directly
        - covariate: Other relevant variables that may need adjustment
        
        Consider:
        1. Temporal relationships (what typically comes first)
        2. Domain knowledge about typical causal patterns
        3. Variable types and their typical roles
        4. The context provided
        
        Respond with JSON:
        {{
            "variable_name": {{
                "role": "treatment|outcome|confounder|mediator|collider|instrumental|covariate",
                "reasoning": "detailed reasoning for this classification",
                "confidence": 0.0-1.0,
                "alternative_roles": ["other possible roles"]
            }}
        }}
        """
        
        try:
            response = await self.llm_client.generate_response(prompt)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed_roles = json.loads(json_match.group())
                return parsed_roles
            else:
                self.logger.warning("Could not parse variable role analysis")
                return {}
                
        except Exception as e:
            self.logger.error(f"Variable role analysis failed: {e}")
            return {}
    
    def _determine_variable_type(self, series: pd.Series) -> str:
        """Determine the type of a variable."""
        
        if pd.api.types.is_numeric_dtype(series):
            unique_values = series.nunique()
            if unique_values == 2:
                return "binary"
            elif unique_values <= 10:
                return "ordinal"
            else:
                return "continuous"
        else:
            return "categorical"
    
    async def _llm_based_discovery(self,
                                 data: pd.DataFrame,
                                 variable_analyses: List[VariableAnalysis],
                                 domain: str,
                                 context: str,
                                 target_variable: Optional[str]) -> List[CausalEdge]:
        """Discover causal relationships using pure LLM reasoning."""
        
        self.logger.info("Performing LLM-based causal discovery")
        
        # Build variable summary for LLM
        var_summary = []
        for var_analysis in variable_analyses:
            var_summary.append(f"- {var_analysis.variable_name}: {var_analysis.description} "
                             f"(Role: {var_analysis.causal_role}, Type: {var_analysis.variable_type})")
        
        # Get basic correlations for LLM context
        numeric_data = data.select_dtypes(include=[np.number])
        correlations = []
        if len(numeric_data.columns) > 1:
            corr_matrix = numeric_data.corr()
            for i, var1 in enumerate(corr_matrix.columns):
                for j, var2 in enumerate(corr_matrix.columns):
                    if i < j and abs(corr_matrix.loc[var1, var2]) > self.min_correlation_threshold:
                        correlations.append(f"{var1} ↔ {var2}: {corr_matrix.loc[var1, var2]:.3f}")
        
        target_guidance = ""
        if target_variable:
            target_guidance = f"\nFOCUS: Pay special attention to relationships involving '{target_variable}'"
        
        prompt = f"""
        You are a causal inference expert discovering causal relationships from variable information.
        
        CONTEXT: {context}
        DOMAIN: {domain}
        {target_guidance}
        
        VARIABLES:
        {chr(10).join(var_summary)}
        
        OBSERVED CORRELATIONS:
        {chr(10).join(correlations[:20])}  
        
        Based on domain knowledge, variable roles, and context, identify likely CAUSAL relationships.
        
        Consider:
        1. Temporal precedence (cause precedes effect)
        2. Domain-specific causal patterns
        3. Biological/logical plausibility
        4. Variable roles (treatments cause outcomes, confounders affect both, etc.)
        5. Avoid reverse causation
        
        For each causal relationship, provide:
        - Direction (which variable causes which)
        - Mechanism (how the causation works)
        - Confidence level
        - Supporting reasoning
        
        Respond with JSON array:
        [
            {{
                "source": "cause_variable",
                "target": "effect_variable", 
                "edge_type": "causal",
                "confidence": 0.0-1.0,
                "mechanism_description": "how this causation works",
                "reasoning": "detailed reasoning for this relationship",
                "strength": "weak|moderate|strong"
            }}
        ]
        
        Only include relationships you are reasonably confident about (confidence > 0.3).
        """
        
        try:
            response = await self.llm_client.generate_response(prompt, max_tokens=2000)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                edges_data = json.loads(json_match.group())
                
                edges = []
                for edge_data in edges_data:
                    edge = CausalEdge(
                        source=edge_data.get("source", ""),
                        target=edge_data.get("target", ""),
                        edge_type=EdgeType(edge_data.get("edge_type", "causal")),
                        confidence=float(edge_data.get("confidence", 0.5)),
                        reasoning=edge_data.get("reasoning", ""),
                        statistical_support={},
                        mechanism_description=edge_data.get("mechanism_description", ""),
                        strength=edge_data.get("strength", "moderate"),
                        evidence_sources=["llm_reasoning"]
                    )
                    edges.append(edge)
                
                return edges
            else:
                self.logger.warning("Could not parse LLM causal discovery response")
                return []
                
        except Exception as e:
            self.logger.error(f"LLM causal discovery failed: {e}")
            return []
    
    async def _statistical_discovery(self,
                                   data: pd.DataFrame,
                                   variable_analyses: List[VariableAnalysis]) -> List[CausalEdge]:
        """Discover relationships using statistical tests."""
        
        self.logger.info("Performing statistical causal discovery")
        
        edges = []
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return edges
        
        # Correlation-based discovery
        corr_matrix = numeric_data.corr()
        
        for i, var1 in enumerate(corr_matrix.columns):
            for j, var2 in enumerate(corr_matrix.columns):
                if i < j:
                    correlation = corr_matrix.loc[var1, var2]
                    
                    if abs(correlation) > self.min_correlation_threshold:
                        # Simple heuristic: assume temporal/logical ordering
                        source, target = self._determine_direction_heuristic(
                            var1, var2, variable_analyses
                        )
                        
                        # Statistical significance test
                        stat, p_value = stats.pearsonr(data[var1].dropna(), data[var2].dropna())
                        
                        if p_value < self.significance_threshold:
                            confidence = min(abs(correlation) + (1 - p_value), 1.0)
                            
                            edge = CausalEdge(
                                source=source,
                                target=target,
                                edge_type=EdgeType.CAUSAL,
                                confidence=confidence,
                                reasoning=f"Statistical correlation: r={correlation:.3f}, p={p_value:.3f}",
                                statistical_support={"correlation": correlation, "p_value": p_value},
                                mechanism_description="Statistical association (mechanism unknown)",
                                strength="moderate" if abs(correlation) > 0.5 else "weak",
                                evidence_sources=["statistical_correlation"]
                            )
                            edges.append(edge)
        
        return edges
    
    def _determine_direction_heuristic(self, 
                                     var1: str, 
                                     var2: str,
                                     variable_analyses: List[VariableAnalysis]) -> Tuple[str, str]:
        """Heuristic to determine causal direction between two variables."""
        
        # Find variable analyses
        var1_analysis = next((v for v in variable_analyses if v.variable_name == var1), None)
        var2_analysis = next((v for v in variable_analyses if v.variable_name == var2), None)
        
        if var1_analysis and var2_analysis:
            # Use role-based heuristics
            role_precedence = {
                "treatment": 1,
                "instrumental": 2,
                "confounder": 3,
                "mediator": 4,
                "outcome": 5,
                "collider": 6
            }
            
            var1_precedence = role_precedence.get(var1_analysis.causal_role, 3)
            var2_precedence = role_precedence.get(var2_analysis.causal_role, 3)
            
            if var1_precedence < var2_precedence:
                return var1, var2
            elif var2_precedence < var1_precedence:
                return var2, var1
        
        # Default: alphabetical order
        return (var1, var2) if var1 < var2 else (var2, var1)
    
    async def _hybrid_discovery(self,
                              data: pd.DataFrame,
                              variable_analyses: List[VariableAnalysis],
                              domain: str,
                              context: str,
                              target_variable: Optional[str]) -> List[CausalEdge]:
        """Combine LLM reasoning with statistical evidence."""
        
        self.logger.info("Performing hybrid LLM-statistical causal discovery")
        
        # Get LLM-based edges
        llm_edges = await self._llm_based_discovery(
            data, variable_analyses, domain, context, target_variable
        )
        
        # Get statistical edges
        stat_edges = await self._statistical_discovery(data, variable_analyses)
        
        # Merge and validate
        merged_edges = self._merge_edge_evidence(llm_edges, stat_edges)
        
        # Enhance with additional statistical support
        enhanced_edges = await self._enhance_with_statistical_tests(merged_edges, data)
        
        return enhanced_edges
    
    def _merge_edge_evidence(self, 
                           llm_edges: List[CausalEdge],
                           stat_edges: List[CausalEdge]) -> List[CausalEdge]:
        """Merge evidence from LLM and statistical discovery."""
        
        # Create edge lookup
        edge_map = {}
        
        # Add LLM edges
        for edge in llm_edges:
            key = (edge.source, edge.target)
            edge_map[key] = edge
        
        # Enhance with statistical evidence
        for stat_edge in stat_edges:
            key = (stat_edge.source, stat_edge.target)
            reverse_key = (stat_edge.target, stat_edge.source)
            
            if key in edge_map:
                # Same direction - enhance existing edge
                existing = edge_map[key]
                existing.statistical_support.update(stat_edge.statistical_support)
                existing.confidence = min((existing.confidence + stat_edge.confidence) / 2 + 0.1, 1.0)
                existing.evidence_sources.extend(stat_edge.evidence_sources)
                existing.reasoning += f" | Statistical support: {stat_edge.reasoning}"
            elif reverse_key in edge_map:
                # Opposite direction - flag as uncertain
                existing = edge_map[reverse_key]
                existing.confidence *= 0.7  # Reduce confidence due to conflict
                existing.reasoning += f" | Conflicting statistical evidence suggests reverse direction"
            else:
                # New edge from statistical analysis
                edge_map[key] = stat_edge
        
        return list(edge_map.values())
    
    async def _enhance_with_statistical_tests(self,
                                            edges: List[CausalEdge],
                                            data: pd.DataFrame) -> List[CausalEdge]:
        """Enhance edges with additional statistical tests."""
        
        enhanced_edges = []
        
        for edge in edges:
            # Check if variables exist in data
            if edge.source not in data.columns or edge.target not in data.columns:
                enhanced_edges.append(edge)
                continue
            
            source_data = data[edge.source].dropna()
            target_data = data[edge.target].dropna()
            
            # Align data (common indices)
            common_idx = source_data.index.intersection(target_data.index)
            if len(common_idx) < 10:  # Need minimum data
                enhanced_edges.append(edge)
                continue
            
            source_aligned = source_data.loc[common_idx]
            target_aligned = target_data.loc[common_idx]
            
            # Additional statistical tests
            additional_tests = {}
            
            # Partial correlation (if more variables available)
            if len(data.columns) > 2:
                other_vars = [col for col in data.columns 
                            if col not in [edge.source, edge.target]]
                if other_vars:
                    control_data = data[other_vars].loc[common_idx].dropna()
                    if len(control_data) > 10:
                        # Simplified partial correlation
                        from sklearn.linear_model import LinearRegression
                        reg_source = LinearRegression().fit(control_data, source_aligned.loc[control_data.index])
                        reg_target = LinearRegression().fit(control_data, target_aligned.loc[control_data.index])
                        
                        residual_source = source_aligned.loc[control_data.index] - reg_source.predict(control_data)
                        residual_target = target_aligned.loc[control_data.index] - reg_target.predict(control_data)
                        
                        partial_corr, partial_p = stats.pearsonr(residual_source, residual_target)
                        additional_tests['partial_correlation'] = partial_corr
                        additional_tests['partial_p_value'] = partial_p
            
            # Update edge with additional evidence
            edge.statistical_support.update(additional_tests)
            
            # Adjust confidence based on additional evidence
            if 'partial_correlation' in additional_tests:
                partial_strength = abs(additional_tests['partial_correlation'])
                if partial_strength > 0.1 and additional_tests.get('partial_p_value', 1.0) < 0.05:
                    edge.confidence = min(edge.confidence + 0.1, 1.0)
                else:
                    edge.confidence *= 0.9
            
            enhanced_edges.append(edge)
        
        return enhanced_edges
    
    async def _constraint_based_discovery(self,
                                        data: pd.DataFrame,
                                        variable_analyses: List[VariableAnalysis]) -> List[CausalEdge]:
        """Perform constraint-based causal discovery (simplified PC algorithm)."""
        
        self.logger.info("Performing constraint-based causal discovery")
        
        # Simplified implementation - for full PC algorithm, use dedicated libraries
        edges = []
        variables = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(variables) < 2:
            return edges
        
        # Phase 1: Find correlations
        correlations = {}
        for var1, var2 in itertools.combinations(variables, 2):
            try:
                corr, p_val = stats.pearsonr(data[var1].dropna(), data[var2].dropna())
                if abs(corr) > self.min_correlation_threshold and p_val < self.significance_threshold:
                    correlations[(var1, var2)] = {'correlation': corr, 'p_value': p_val}
            except:
                continue
        
        # Phase 2: Test for independence given conditioning sets
        # (Simplified - only test with one conditioning variable)
        independent_pairs = set()
        
        for (var1, var2), corr_info in correlations.items():
            for conditioning_var in variables:
                if conditioning_var not in [var1, var2]:
                    try:
                        # Test conditional independence
                        subset = data[[var1, var2, conditioning_var]].dropna()
                        if len(subset) > 20:
                            partial_corr = self._calculate_partial_correlation(
                                subset[var1], subset[var2], subset[conditioning_var]
                            )
                            
                            if abs(partial_corr) < 0.1:  # Approximately independent
                                independent_pairs.add((var1, var2))
                                break
                    except:
                        continue
        
        # Phase 3: Create edges for non-independent pairs
        for (var1, var2), corr_info in correlations.items():
            if (var1, var2) not in independent_pairs:
                # Determine direction using variable roles
                source, target = self._determine_direction_heuristic(
                    var1, var2, variable_analyses
                )
                
                edge = CausalEdge(
                    source=source,
                    target=target,
                    edge_type=EdgeType.CAUSAL,
                    confidence=min(abs(corr_info['correlation']) + 0.2, 0.8),
                    reasoning=f"Constraint-based discovery: not d-separated by observed variables",
                    statistical_support=corr_info,
                    mechanism_description="Discovered through conditional independence testing",
                    strength="moderate",
                    evidence_sources=["constraint_based"]
                )
                edges.append(edge)
        
        return edges
    
    def _calculate_partial_correlation(self, x: pd.Series, y: pd.Series, z: pd.Series) -> float:
        """Calculate partial correlation between x and y given z."""
        
        from sklearn.linear_model import LinearRegression
        
        # Regress x on z
        reg_x = LinearRegression().fit(z.values.reshape(-1, 1), x)
        residual_x = x - reg_x.predict(z.values.reshape(-1, 1))
        
        # Regress y on z  
        reg_y = LinearRegression().fit(z.values.reshape(-1, 1), y)
        residual_y = y - reg_y.predict(z.values.reshape(-1, 1))
        
        # Correlation between residuals
        partial_corr, _ = stats.pearsonr(residual_x, residual_y)
        return partial_corr
    
    def _build_causal_graph(self,
                          variable_analyses: List[VariableAnalysis],
                          edges: List[CausalEdge]) -> nx.DiGraph:
        """Build NetworkX graph from discovered edges."""
        
        graph = nx.DiGraph()
        
        # Add nodes
        for var_analysis in variable_analyses:
            graph.add_node(
                var_analysis.variable_name,
                description=var_analysis.description,
                causal_role=var_analysis.causal_role,
                variable_type=var_analysis.variable_type
            )
        
        # Add edges
        for edge in edges:
            if edge.source and edge.target:
                graph.add_edge(
                    edge.source,
                    edge.target,
                    confidence=edge.confidence,
                    edge_type=edge.edge_type.value,
                    mechanism=edge.mechanism_description,
                    strength=edge.strength
                )
        
        # Update parent-child relationships in variable analyses
        for var_analysis in variable_analyses:
            var_analysis.parents = list(graph.predecessors(var_analysis.variable_name))
            var_analysis.children = list(graph.successors(var_analysis.variable_name))
        
        return graph
    
    async def _validate_structure(self,
                                graph: nx.DiGraph,
                                edges: List[CausalEdge],
                                variable_analyses: List[VariableAnalysis],
                                data: pd.DataFrame,
                                domain: str,
                                context: str) -> CausalStructure:
        """Validate discovered causal structure."""
        
        self.logger.info("Validating discovered causal structure")
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(graph):
            self.logger.warning("Discovered structure contains cycles - removing weakest edges")
            edges = self._remove_cycles(graph, edges)
            graph = self._build_causal_graph(variable_analyses, edges)
        
        # Domain-specific validation
        validation_issues = await self._domain_validate_structure(graph, domain, context)
        
        # Adjust confidences based on validation
        for issue in validation_issues:
            if 'edge' in issue:
                edge_info = issue['edge']
                matching_edges = [e for e in edges 
                                if e.source == edge_info.get('source') and e.target == edge_info.get('target')]
                for edge in matching_edges:
                    edge.confidence *= 0.8
                    edge.reasoning += f" | Validation concern: {issue['concern']}"
        
        return graph
    
    def _remove_cycles(self, graph: nx.DiGraph, edges: List[CausalEdge]) -> List[CausalEdge]:
        """Remove cycles by removing lowest confidence edges."""
        
        edges_sorted = sorted(edges, key=lambda e: e.confidence)
        filtered_edges = []
        temp_graph = nx.DiGraph()
        
        # Add nodes
        temp_graph.add_nodes_from(graph.nodes())
        
        # Add edges in order of confidence, skipping those that create cycles
        for edge in edges_sorted:
            temp_graph.add_edge(edge.source, edge.target)
            
            if nx.is_directed_acyclic_graph(temp_graph):
                filtered_edges.append(edge)
            else:
                temp_graph.remove_edge(edge.source, edge.target)
        
        return filtered_edges
    
    async def _domain_validate_structure(self,
                                       graph: nx.DiGraph,
                                       domain: str,
                                       context: str) -> List[Dict[str, Any]]:
        """Validate structure against domain knowledge."""
        
        validation_prompt = f"""
        You are validating a discovered causal structure for domain: {domain}
        
        CONTEXT: {context}
        
        DISCOVERED STRUCTURE:
        Nodes: {list(graph.nodes())}
        Edges: {[(u, v) for u, v in graph.edges()]}
        
        Please identify any issues with this causal structure based on domain knowledge:
        1. Implausible causal relationships
        2. Missing important relationships
        3. Incorrect causal directions
        4. Violations of temporal ordering
        
        Respond with JSON:
        {{
            "issues": [
                {{
                    "type": "implausible|missing|incorrect_direction|temporal_violation",
                    "concern": "description of the issue",
                    "edge": {{"source": "var1", "target": "var2"}},
                    "severity": "low|medium|high"
                }}
            ],
            "overall_assessment": "brief assessment of structure validity"
        }}
        """
        
        try:
            response = await self.llm_client.generate_response(validation_prompt)
            
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                validation_result = json.loads(json_match.group())
                return validation_result.get('issues', [])
            
        except Exception as e:
            self.logger.error(f"Structure validation failed: {e}")
        
        return []
    
    async def _generate_alternative_structures(self,
                                             main_structure: nx.DiGraph,
                                             data: pd.DataFrame,
                                             variable_analyses: List[VariableAnalysis],
                                             domain: str) -> List['CausalStructure']:
        """Generate alternative causal structures."""
        
        # For now, return empty list - could implement:
        # 1. Edge direction reversals
        # 2. Different statistical thresholds
        # 3. Alternative domain assumptions
        
        return []
    
    def _calculate_overall_confidence(self, edges: List[CausalEdge]) -> float:
        """Calculate overall confidence in the discovered structure."""
        
        if not edges:
            return 0.0
        
        confidences = [edge.confidence for edge in edges]
        return sum(confidences) / len(confidences)
    
    def _identify_assumptions(self, method: DiscoveryMethod, domain: str) -> List[str]:
        """Identify key assumptions made during discovery."""
        
        base_assumptions = [
            "Causal sufficiency (no unmeasured confounders)",
            "Faithfulness (conditional independences reflect causal structure)",
            "No selection bias in data collection"
        ]
        
        if method == DiscoveryMethod.LLM_REASONING:
            base_assumptions.append("LLM domain knowledge is accurate and unbiased")
        
        if method in [DiscoveryMethod.STATISTICAL_TESTS, DiscoveryMethod.CONSTRAINT_BASED]:
            base_assumptions.extend([
                "Linear relationships (for correlation-based methods)",
                "Gaussian distributions (for parametric tests)"
            ])
        
        return base_assumptions
    
    def _identify_limitations(self, data: pd.DataFrame, method: DiscoveryMethod) -> List[str]:
        """Identify limitations of the discovery process."""
        
        limitations = []
        
        # Data limitations
        if len(data) < 100:
            limitations.append("Small sample size may affect reliability")
        
        if data.isnull().sum().sum() > len(data) * 0.1:
            limitations.append("High rate of missing data")
        
        if len(data.columns) > 20:
            limitations.append("High-dimensional data may lead to spurious associations")
        
        # Method limitations
        if method == DiscoveryMethod.LLM_REASONING:
            limitations.append("Purely reasoning-based without statistical validation")
        
        if method == DiscoveryMethod.STATISTICAL_TESTS:
            limitations.append("Cannot establish causal direction from correlations alone")
        
        limitations.append("Cross-sectional data limits temporal precedence inference")
        
        return limitations


# Convenience functions
def create_causal_discovery_agent(llm_client) -> LLMCausalDiscoveryAgent:
    """Create an LLM causal discovery agent."""
    return LLMCausalDiscoveryAgent(llm_client)


async def discover_causal_structure_from_data(data: pd.DataFrame,
                                            variable_descriptions: Dict[str, str],
                                            llm_client,
                                            domain: str = "general",
                                            context: str = "",
                                            method: DiscoveryMethod = DiscoveryMethod.HYBRID_LLM_STATISTICAL) -> CausalStructure:
    """Quick function to discover causal structure from data."""
    
    agent = create_causal_discovery_agent(llm_client)
    return await agent.discover_causal_structure(
        data, variable_descriptions, domain, context, method
    )