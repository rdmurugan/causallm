"""
Temporal causal modeling capabilities for Tier 2 enhancements.

This module provides advanced time-series causal analysis, dynamic causal graphs,
intervention timing optimization, and longitudinal effect estimation using LLM-guided
temporal reasoning and statistical methods.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import asyncio
import json
import warnings
from collections import defaultdict, deque

from causalllm.logging import get_logger


class TemporalRelationType(Enum):
    """Types of temporal causal relationships."""
    INSTANTANEOUS = "instantaneous"  # X(t) → Y(t)
    LAGGED = "lagged"               # X(t) → Y(t+k)
    PERSISTENT = "persistent"       # X(t) → Y(t+k), Y(t+k+1), ...
    CUMULATIVE = "cumulative"       # ∑X(t-k:t) → Y(t)
    CYCLICAL = "cyclical"          # X(t) → Y(t+period*k)
    DECAYING = "decaying"          # X(t) → Y(t+k) with decay


class TimeUnit(Enum):
    """Time units for temporal modeling."""
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"
    YEARS = "years"


class CausalMechanism(Enum):
    """Causal mechanisms over time."""
    DIRECT_EFFECT = "direct"
    MEDIATED_EFFECT = "mediated"
    MODERATED_EFFECT = "moderated"
    THRESHOLD_EFFECT = "threshold"
    SATURATION_EFFECT = "saturation"
    INTERACTION_EFFECT = "interaction"


@dataclass
class TemporalCausalEdge:
    """Represents a temporal causal relationship."""
    
    cause: str
    effect: str
    relation_type: TemporalRelationType
    lag: int  # Time lag in specified units
    time_unit: TimeUnit
    strength: float  # Causal effect size
    confidence: float
    mechanism: CausalMechanism
    duration: Optional[int] = None  # How long the effect lasts
    decay_rate: Optional[float] = None  # For decaying effects
    threshold: Optional[float] = None  # For threshold effects
    seasonality: Optional[Dict[str, float]] = None  # Seasonal patterns
    evidence: Optional[Dict[str, Any]] = None


@dataclass
class TemporalState:
    """State of the system at a specific time point."""
    
    timestamp: datetime
    variable_values: Dict[str, float]
    interventions: Dict[str, Any]
    external_factors: Dict[str, float]
    confidence: float = 1.0


@dataclass
class TemporalTrajectory:
    """Sequence of states over time showing causal evolution."""
    
    states: List[TemporalState]
    start_time: datetime
    end_time: datetime
    sampling_frequency: str
    trajectory_type: str  # "observed", "counterfactual", "predicted"
    confidence_intervals: Optional[Dict[str, List[Tuple[float, float]]]] = None


@dataclass
class TemporalInterventionPlan:
    """Time-sensitive intervention plan."""
    
    interventions: List[Tuple[datetime, str, Any]]  # (time, variable, value)
    expected_trajectory: TemporalTrajectory
    optimal_timing: Dict[str, datetime]
    timing_sensitivity: Dict[str, float]
    coordination_requirements: List[str]
    monitoring_schedule: List[Tuple[datetime, List[str]]]  # (time, variables_to_monitor)


@dataclass
class TemporalAnalysisResult:
    """Result from temporal causal analysis."""
    
    temporal_edges: List[TemporalCausalEdge]
    dynamic_graph_evolution: List[Tuple[datetime, List[TemporalCausalEdge]]]
    causal_forecasts: Dict[str, TemporalTrajectory]
    intervention_windows: Dict[str, List[Tuple[datetime, datetime]]]  # Optimal intervention periods
    temporal_confounders: List[str]
    analysis_metadata: Dict[str, Any]
    reasoning_trace: List[str]


class TemporalCausalModel(ABC):
    """Abstract base class for temporal causal models."""
    
    def __init__(self, time_unit: TimeUnit = TimeUnit.DAYS):
        self.time_unit = time_unit
        self.logger = get_logger(f"causalllm.temporal_causal_model")
    
    @abstractmethod
    async def fit(self, temporal_data: pd.DataFrame, 
                 variables: Dict[str, str],
                 time_column: str = "timestamp") -> None:
        """Fit the temporal causal model to data."""
        pass
    
    @abstractmethod
    async def discover_temporal_structure(self, 
                                         max_lag: int = 10) -> List[TemporalCausalEdge]:
        """Discover temporal causal relationships."""
        pass
    
    @abstractmethod
    async def forecast_trajectory(self, 
                                 start_state: TemporalState,
                                 horizon: int,
                                 interventions: Optional[List[Tuple[int, str, Any]]] = None) -> TemporalTrajectory:
        """Forecast future trajectory of the system."""
        pass


class LLMGuidedTemporalModel(TemporalCausalModel):
    """LLM-guided temporal causal modeling."""
    
    def __init__(self, llm_client, time_unit: TimeUnit = TimeUnit.DAYS):
        super().__init__(time_unit)
        self.llm_client = llm_client
        self.temporal_data = None
        self.variables = None
        self.discovered_edges = []
        self.temporal_patterns = {}
    
    async def fit(self, temporal_data: pd.DataFrame,
                 variables: Dict[str, str], 
                 time_column: str = "timestamp") -> None:
        """Fit the temporal model to data."""
        self.logger.info("Fitting LLM-guided temporal causal model")
        
        self.temporal_data = temporal_data.copy()
        self.variables = variables
        
        # Ensure time column is datetime
        if time_column in temporal_data.columns:
            self.temporal_data[time_column] = pd.to_datetime(temporal_data[time_column])
            self.temporal_data = self.temporal_data.sort_values(time_column)
        else:
            raise ValueError(f"Time column '{time_column}' not found in data")
        
        # Analyze temporal patterns
        await self._analyze_temporal_patterns()
        
        self.logger.info("Temporal model fitting completed")
    
    async def _analyze_temporal_patterns(self):
        """Analyze temporal patterns in the data."""
        self.logger.debug("Analyzing temporal patterns")
        
        for variable in self.variables.keys():
            if variable in self.temporal_data.columns:
                series = self.temporal_data[variable].dropna()
                
                if len(series) < 10:
                    continue
                
                patterns = {}
                
                # Basic trend analysis
                if len(series) > 1:
                    trend = np.polyfit(range(len(series)), series, 1)[0]
                    patterns["trend"] = "increasing" if trend > 0.01 else "decreasing" if trend < -0.01 else "stable"
                
                # Volatility
                patterns["volatility"] = np.std(series) / np.mean(series) if np.mean(series) != 0 else 0
                
                # Seasonality detection (simplified)
                if len(series) >= 12:
                    seasonal_strength = self._detect_seasonality(series)
                    patterns["seasonality"] = seasonal_strength
                
                # Autocorrelation
                if len(series) >= 5:
                    autocorr = series.autocorr() if hasattr(series, 'autocorr') else 0
                    patterns["autocorrelation"] = autocorr
                
                self.temporal_patterns[variable] = patterns
    
    def _detect_seasonality(self, series: pd.Series) -> float:
        """Simple seasonality detection."""
        try:
            # Check for weekly seasonality (if data is daily)
            if len(series) >= 14:
                weekly_corr = 0
                for lag in [7, 14]:
                    if len(series) > lag:
                        corr = series.corr(series.shift(lag))
                        weekly_corr = max(weekly_corr, abs(corr) if not np.isnan(corr) else 0)
                return weekly_corr
            return 0.0
        except:
            return 0.0
    
    async def discover_temporal_structure(self, max_lag: int = 10) -> List[TemporalCausalEdge]:
        """Discover temporal causal relationships using LLM guidance."""
        self.logger.info("Discovering temporal causal structure")
        
        if self.temporal_data is None:
            raise ValueError("Model must be fitted before discovering structure")
        
        # Generate statistical evidence for temporal relationships
        statistical_evidence = await self._generate_temporal_statistical_evidence(max_lag)
        
        # Query LLM for temporal causal insights
        discovered_edges = await self._llm_discover_temporal_causation(statistical_evidence, max_lag)
        
        # Validate and refine discovered relationships
        refined_edges = await self._validate_temporal_relationships(discovered_edges)
        
        self.discovered_edges = refined_edges
        return refined_edges
    
    async def _generate_temporal_statistical_evidence(self, max_lag: int) -> Dict[str, Any]:
        """Generate statistical evidence for temporal relationships."""
        evidence = {
            "cross_correlations": {},
            "granger_causality": {},
            "lagged_correlations": {},
            "temporal_patterns": self.temporal_patterns
        }
        
        variable_names = list(self.variables.keys())
        
        for cause_var in variable_names:
            for effect_var in variable_names:
                if cause_var == effect_var:
                    continue
                
                if cause_var in self.temporal_data.columns and effect_var in self.temporal_data.columns:
                    # Cross-correlation analysis
                    cause_series = self.temporal_data[cause_var].dropna()
                    effect_series = self.temporal_data[effect_var].dropna()
                    
                    if len(cause_series) > max_lag and len(effect_series) > max_lag:
                        cross_corrs = {}
                        for lag in range(max_lag + 1):
                            if lag == 0:
                                corr = cause_series.corr(effect_series)
                            else:
                                # Lag correlation: cause at t, effect at t+lag
                                aligned_cause = cause_series[:-lag]
                                aligned_effect = effect_series[lag:]
                                if len(aligned_cause) > 0 and len(aligned_effect) > 0:
                                    corr = aligned_cause.corr(aligned_effect)
                                else:
                                    corr = 0
                            
                            cross_corrs[lag] = corr if not np.isnan(corr) else 0
                        
                        evidence["cross_correlations"][(cause_var, effect_var)] = cross_corrs
                        
                        # Find best lag
                        best_lag = max(cross_corrs.keys(), key=lambda k: abs(cross_corrs[k]))
                        evidence["lagged_correlations"][(cause_var, effect_var)] = {
                            "best_lag": best_lag,
                            "correlation": cross_corrs[best_lag]
                        }
        
        return evidence
    
    async def _llm_discover_temporal_causation(self, statistical_evidence: Dict[str, Any], 
                                              max_lag: int) -> List[TemporalCausalEdge]:
        """Use LLM to discover temporal causal relationships."""
        
        prompt = f"""
        You are an expert in temporal causal analysis. Analyze the time series data to identify
        causal relationships that unfold over time.
        
        VARIABLES:
        """
        
        for var, desc in self.variables.items():
            prompt += f"\n- {var}: {desc}"
        
        prompt += f"""
        
        TEMPORAL PATTERNS:
        """
        
        for var, patterns in self.temporal_patterns.items():
            prompt += f"\n- {var}: trend={patterns.get('trend', 'unknown')}, volatility={patterns.get('volatility', 0):.3f}"
        
        if statistical_evidence.get("lagged_correlations"):
            prompt += f"""
            
            STATISTICAL EVIDENCE (Lagged Correlations):
            """
            
            for (cause, effect), data in statistical_evidence["lagged_correlations"].items():
                if abs(data["correlation"]) > 0.3:  # Only show significant correlations
                    prompt += f"\n- {cause} → {effect} (lag {data['best_lag']}): correlation = {data['correlation']:.3f}"
        
        prompt += f"""
        
        TASK: Identify temporal causal relationships where one variable influences another over time.
        Consider:
        1. Temporal precedence (cause must precede effect)
        2. Statistical evidence from lag correlations
        3. Domain knowledge about realistic causal mechanisms
        4. Temporal patterns and seasonality
        
        Respond with JSON array of temporal causal edges:
        [
          {{
            "cause": "variable_name",
            "effect": "variable_name", 
            "relation_type": "lagged/persistent/cumulative/instantaneous",
            "lag": 2,
            "strength": 0.75,
            "confidence": 0.8,
            "mechanism": "direct/mediated/threshold/decaying",
            "reasoning": "explanation of the temporal causal relationship",
            "duration": 5
          }}
        ]
        
        Only include relationships you believe are genuinely causal over time.
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
                edges_data = json.loads(json_match.group())
                
                discovered_edges = []
                for edge_data in edges_data:
                    try:
                        # Map string values to enums
                        relation_type = TemporalRelationType(edge_data.get("relation_type", "lagged"))
                        mechanism = CausalMechanism(edge_data.get("mechanism", "direct"))
                        
                        edge = TemporalCausalEdge(
                            cause=edge_data["cause"],
                            effect=edge_data["effect"],
                            relation_type=relation_type,
                            lag=edge_data.get("lag", 1),
                            time_unit=self.time_unit,
                            strength=edge_data.get("strength", 0.5),
                            confidence=edge_data.get("confidence", 0.5),
                            mechanism=mechanism,
                            duration=edge_data.get("duration"),
                            evidence={"llm_reasoning": edge_data.get("reasoning", "")}
                        )
                        
                        discovered_edges.append(edge)
                        
                    except (KeyError, ValueError) as e:
                        self.logger.warning(f"Skipping malformed edge data: {e}")
                        continue
                
                self.logger.info(f"LLM discovered {len(discovered_edges)} temporal causal edges")
                return discovered_edges
            
            else:
                self.logger.warning("Could not parse JSON from LLM response")
                return []
                
        except Exception as e:
            self.logger.error(f"LLM temporal discovery failed: {e}")
            return []
    
    async def _validate_temporal_relationships(self, edges: List[TemporalCausalEdge]) -> List[TemporalCausalEdge]:
        """Validate and refine temporal causal relationships."""
        validated_edges = []
        
        for edge in edges:
            # Check if variables exist in data
            if edge.cause not in self.variables or edge.effect not in self.variables:
                continue
            
            # Validate against statistical evidence if available
            if hasattr(self, 'statistical_evidence'):
                correlation_key = (edge.cause, edge.effect)
                if correlation_key in self.statistical_evidence.get("lagged_correlations", {}):
                    corr_data = self.statistical_evidence["lagged_correlations"][correlation_key]
                    
                    # Adjust confidence based on correlation strength
                    corr_strength = abs(corr_data["correlation"])
                    if corr_strength > 0.5:
                        edge.confidence = min(edge.confidence + 0.2, 0.95)
                    elif corr_strength < 0.2:
                        edge.confidence = max(edge.confidence - 0.3, 0.1)
                    
                    # Update evidence
                    if edge.evidence is None:
                        edge.evidence = {}
                    edge.evidence["statistical_correlation"] = corr_data["correlation"]
                    edge.evidence["statistical_lag"] = corr_data["best_lag"]
            
            validated_edges.append(edge)
        
        return validated_edges
    
    async def forecast_trajectory(self, start_state: TemporalState,
                                 horizon: int,
                                 interventions: Optional[List[Tuple[int, str, Any]]] = None) -> TemporalTrajectory:
        """Forecast future trajectory using temporal model."""
        self.logger.info(f"Forecasting trajectory for {horizon} time steps")
        
        if not self.discovered_edges:
            raise ValueError("No temporal structure discovered. Call discover_temporal_structure() first.")
        
        states = [start_state]
        current_state = start_state
        interventions = interventions or []
        intervention_dict = {step: (var, val) for step, var, val in interventions}
        
        for step in range(1, horizon + 1):
            next_values = current_state.variable_values.copy()
            
            # Apply interventions at this step
            if step in intervention_dict:
                var, val = intervention_dict[step]
                next_values[var] = val
            
            # Apply temporal causal effects
            for edge in self.discovered_edges:
                if edge.lag <= step and edge.cause in current_state.variable_values:
                    # Get historical value at appropriate lag
                    if edge.lag == 0:
                        cause_value = current_state.variable_values[edge.cause]
                    elif step - edge.lag >= 0 and step - edge.lag < len(states):
                        cause_value = states[step - edge.lag].variable_values[edge.cause]
                    else:
                        continue  # Not enough history
                    
                    # Apply causal effect based on mechanism
                    effect_change = self._calculate_temporal_effect(edge, cause_value, step)
                    
                    if edge.effect in next_values:
                        next_values[edge.effect] += effect_change
                    else:
                        next_values[edge.effect] = effect_change
            
            # Create next state
            next_timestamp = start_state.timestamp + timedelta(
                **{self.time_unit.value: step}
            )
            
            next_state = TemporalState(
                timestamp=next_timestamp,
                variable_values=next_values,
                interventions=intervention_dict.get(step, {}),
                external_factors={},
                confidence=max(0.1, current_state.confidence - 0.05 * step)  # Decay confidence over time
            )
            
            states.append(next_state)
            current_state = next_state
        
        return TemporalTrajectory(
            states=states,
            start_time=start_state.timestamp,
            end_time=states[-1].timestamp,
            sampling_frequency=self.time_unit.value,
            trajectory_type="predicted"
        )
    
    def _calculate_temporal_effect(self, edge: TemporalCausalEdge, cause_value: float, step: int) -> float:
        """Calculate temporal causal effect based on mechanism."""
        base_effect = edge.strength * cause_value
        
        if edge.mechanism == CausalMechanism.DIRECT_EFFECT:
            return base_effect
        
        elif edge.mechanism == CausalMechanism.DECAYING:
            if edge.decay_rate:
                decay_factor = np.exp(-edge.decay_rate * step)
                return base_effect * decay_factor
            return base_effect
        
        elif edge.mechanism == CausalMechanism.THRESHOLD_EFFECT:
            if edge.threshold and abs(cause_value) > edge.threshold:
                return base_effect * 2  # Amplify effect above threshold
            return base_effect * 0.1  # Minimal effect below threshold
        
        elif edge.mechanism == CausalMechanism.SATURATION_EFFECT:
            # Sigmoid-like saturation
            return base_effect * (1 - np.exp(-abs(cause_value)))
        
        else:
            return base_effect


class AdvancedTemporalAnalyzer:
    """Advanced temporal causal analysis system."""
    
    def __init__(self, llm_client, time_unit: TimeUnit = TimeUnit.DAYS):
        self.llm_client = llm_client
        self.time_unit = time_unit
        self.logger = get_logger("causalllm.advanced_temporal_analyzer")
        
        # Initialize temporal models
        self.models = {
            "llm_guided": LLMGuidedTemporalModel(llm_client, time_unit)
        }
    
    async def analyze_temporal_causation(self, temporal_data: pd.DataFrame,
                                        variables: Dict[str, str],
                                        time_column: str = "timestamp",
                                        max_lag: int = 10,
                                        domain_context: str = "") -> TemporalAnalysisResult:
        """Comprehensive temporal causal analysis."""
        self.logger.info("Starting comprehensive temporal causal analysis")
        
        reasoning_trace = []
        
        # Step 1: Fit temporal model
        reasoning_trace.append("Fitting temporal causal model to data")
        model = self.models["llm_guided"]
        await model.fit(temporal_data, variables, time_column)
        
        # Step 2: Discover temporal structure
        reasoning_trace.append("Discovering temporal causal structure")
        temporal_edges = await model.discover_temporal_structure(max_lag)
        
        # Step 3: Analyze dynamic graph evolution
        reasoning_trace.append("Analyzing dynamic graph evolution")
        graph_evolution = await self._analyze_graph_evolution(temporal_data, temporal_edges, time_column)
        
        # Step 4: Generate causal forecasts
        reasoning_trace.append("Generating causal forecasts")
        forecasts = await self._generate_causal_forecasts(model, temporal_data, variables, time_column)
        
        # Step 5: Identify optimal intervention windows
        reasoning_trace.append("Identifying optimal intervention windows")
        intervention_windows = await self._identify_intervention_windows(temporal_edges, temporal_data, time_column)
        
        # Step 6: Detect temporal confounders
        reasoning_trace.append("Detecting temporal confounders")
        temporal_confounders = await self._detect_temporal_confounders(temporal_edges, temporal_data)
        
        # Compile metadata
        analysis_metadata = {
            "data_timespan": {
                "start": temporal_data[time_column].min(),
                "end": temporal_data[time_column].max(),
                "duration": len(temporal_data)
            },
            "temporal_edges_discovered": len(temporal_edges),
            "max_lag_analyzed": max_lag,
            "time_unit": time_unit.value,
            "model_used": "llm_guided"
        }
        
        return TemporalAnalysisResult(
            temporal_edges=temporal_edges,
            dynamic_graph_evolution=graph_evolution,
            causal_forecasts=forecasts,
            intervention_windows=intervention_windows,
            temporal_confounders=temporal_confounders,
            analysis_metadata=analysis_metadata,
            reasoning_trace=reasoning_trace
        )
    
    async def _analyze_graph_evolution(self, data: pd.DataFrame, 
                                     edges: List[TemporalCausalEdge],
                                     time_column: str) -> List[Tuple[datetime, List[TemporalCausalEdge]]]:
        """Analyze how causal graph structure changes over time."""
        evolution = []
        
        # Simple approach: analyze graph stability over time windows
        data_sorted = data.sort_values(time_column)
        window_size = max(30, len(data) // 10)  # Adaptive window size
        
        for i in range(0, len(data) - window_size, window_size // 2):
            window_data = data_sorted.iloc[i:i + window_size]
            window_time = window_data[time_column].iloc[0]
            
            # For now, assume edge strengths may vary but structure is stable
            # In advanced implementation, would re-analyze structure for each window
            window_edges = []
            for edge in edges:
                # Simple variation in edge strength over time (placeholder)
                time_factor = 1.0 + 0.1 * np.sin(i / 10)  # Simulated temporal variation
                varied_edge = TemporalCausalEdge(
                    cause=edge.cause,
                    effect=edge.effect,
                    relation_type=edge.relation_type,
                    lag=edge.lag,
                    time_unit=edge.time_unit,
                    strength=edge.strength * time_factor,
                    confidence=edge.confidence,
                    mechanism=edge.mechanism,
                    evidence=edge.evidence
                )
                window_edges.append(varied_edge)
            
            evolution.append((window_time, window_edges))
        
        return evolution
    
    async def _generate_causal_forecasts(self, model: TemporalCausalModel,
                                       data: pd.DataFrame,
                                       variables: Dict[str, str],
                                       time_column: str) -> Dict[str, TemporalTrajectory]:
        """Generate forecasts for each variable."""
        forecasts = {}
        
        # Use last observation as starting state
        last_obs = data.iloc[-1]
        start_state = TemporalState(
            timestamp=last_obs[time_column],
            variable_values={var: last_obs[var] for var in variables.keys() if var in data.columns},
            interventions={},
            external_factors={}
        )
        
        # Generate forecast for next 10 time steps
        forecast_horizon = 10
        
        try:
            trajectory = await model.forecast_trajectory(start_state, forecast_horizon)
            
            # Split trajectory by variable
            for var in variables.keys():
                if var in start_state.variable_values:
                    var_states = []
                    for state in trajectory.states:
                        if var in state.variable_values:
                            var_state = TemporalState(
                                timestamp=state.timestamp,
                                variable_values={var: state.variable_values[var]},
                                interventions=state.interventions,
                                external_factors=state.external_factors,
                                confidence=state.confidence
                            )
                            var_states.append(var_state)
                    
                    forecasts[var] = TemporalTrajectory(
                        states=var_states,
                        start_time=trajectory.start_time,
                        end_time=trajectory.end_time,
                        sampling_frequency=trajectory.sampling_frequency,
                        trajectory_type="forecast"
                    )
            
        except Exception as e:
            self.logger.warning(f"Failed to generate forecasts: {e}")
            # Return empty forecasts
            for var in variables.keys():
                forecasts[var] = TemporalTrajectory(
                    states=[],
                    start_time=start_state.timestamp,
                    end_time=start_state.timestamp,
                    sampling_frequency=self.time_unit.value,
                    trajectory_type="failed_forecast"
                )
        
        return forecasts
    
    async def _identify_intervention_windows(self, edges: List[TemporalCausalEdge],
                                           data: pd.DataFrame,
                                           time_column: str) -> Dict[str, List[Tuple[datetime, datetime]]]:
        """Identify optimal time windows for interventions."""
        intervention_windows = {}
        
        # Group edges by cause variable
        edges_by_cause = defaultdict(list)
        for edge in edges:
            edges_by_cause[edge.cause].append(edge)
        
        data_sorted = data.sort_values(time_column)
        time_span = data_sorted[time_column].max() - data_sorted[time_column].min()
        window_duration = time_span / 10  # Divide into 10 potential windows
        
        for cause_var, cause_edges in edges_by_cause.items():
            windows = []
            
            # Find periods when intervention on cause_var would be most effective
            # Based on temporal patterns and effect strengths
            
            start_time = data_sorted[time_column].min()
            for i in range(10):  # Create 10 potential windows
                window_start = start_time + i * window_duration
                window_end = window_start + window_duration
                
                # Calculate effectiveness score for this window
                effectiveness = 0.0
                for edge in cause_edges:
                    # Higher effectiveness for stronger, more confident edges
                    effectiveness += edge.strength * edge.confidence
                
                # Simple heuristic: include window if effectiveness is above threshold
                if effectiveness > 0.5:
                    windows.append((window_start, window_end))
            
            intervention_windows[cause_var] = windows
        
        return intervention_windows
    
    async def _detect_temporal_confounders(self, edges: List[TemporalCausalEdge],
                                         data: pd.DataFrame) -> List[str]:
        """Detect variables that act as temporal confounders."""
        confounders = set()
        
        # Look for variables that affect multiple other variables
        cause_counts = defaultdict(int)
        for edge in edges:
            cause_counts[edge.cause] += 1
        
        # Variables affecting 2+ others are potential confounders
        for cause, count in cause_counts.items():
            if count >= 2:
                confounders.add(cause)
        
        # Also check for variables with high autocorrelation (persistent effects)
        for var in data.columns:
            if var != "timestamp" and var in data.columns:
                series = data[var].dropna()
                if len(series) > 5:
                    try:
                        autocorr = series.autocorr(lag=1) if hasattr(series, 'autocorr') else 0
                        if autocorr > 0.7:  # High persistence
                            confounders.add(var)
                    except:
                        pass
        
        return list(confounders)
    
    async def optimize_temporal_interventions(self, 
                                            temporal_edges: List[TemporalCausalEdge],
                                            target_variable: str,
                                            intervention_budget: float,
                                            planning_horizon: int) -> TemporalInterventionPlan:
        """Optimize timing and coordination of interventions."""
        self.logger.info("Optimizing temporal intervention strategy")
        
        # Find edges that affect the target variable
        relevant_edges = [edge for edge in temporal_edges if edge.effect == target_variable]
        
        if not relevant_edges:
            raise ValueError(f"No causal edges found affecting target variable: {target_variable}")
        
        # Use LLM to optimize intervention timing
        prompt = f"""
        You are an expert in temporal intervention optimization. Design an optimal intervention
        strategy to maximize impact on the target variable '{target_variable}'.
        
        AVAILABLE CAUSAL LEVERS:
        """
        
        for edge in relevant_edges:
            prompt += f"""
        - {edge.cause} → {target_variable} (lag: {edge.lag} {edge.time_unit.value}, strength: {edge.strength:.2f})
          Mechanism: {edge.mechanism.value}, Confidence: {edge.confidence:.2f}
        """
        
        prompt += f"""
        
        CONSTRAINTS:
        - Intervention budget: {intervention_budget}
        - Planning horizon: {planning_horizon} time steps
        
        TASK: Design an optimal temporal intervention plan that:
        1. Maximizes impact on {target_variable}
        2. Considers causal lags and timing
        3. Stays within budget
        4. Coordinates multiple interventions if beneficial
        
        Respond with JSON:
        {{
          "interventions": [
            {{
              "time_step": 3,
              "variable": "variable_name",
              "intervention_value": "description",
              "cost": 150,
              "expected_impact": 0.8
            }}
          ],
          "timing_rationale": "explanation of why this timing is optimal",
          "coordination_strategy": "how interventions work together",
          "monitoring_plan": [
            {{
              "time_step": 5,
              "variables_to_monitor": ["var1", "var2"],
              "purpose": "track intermediate effects"
            }}
          ]
        }}
        """
        
        try:
            if hasattr(self.llm_client, 'generate_response'):
                response = await self.llm_client.generate_response(prompt)
            else:
                response = await asyncio.to_thread(self.llm_client.generate, prompt)
            
            # Parse response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
                
                # Convert to TemporalInterventionPlan
                interventions = []
                base_time = datetime.now()
                
                for intervention in plan_data.get("interventions", []):
                    time_step = intervention.get("time_step", 0)
                    intervention_time = base_time + timedelta(**{self.time_unit.value: time_step})
                    interventions.append((
                        intervention_time,
                        intervention.get("variable", ""),
                        intervention.get("intervention_value", "")
                    ))
                
                # Create monitoring schedule
                monitoring_schedule = []
                for monitor in plan_data.get("monitoring_plan", []):
                    time_step = monitor.get("time_step", 0)
                    monitor_time = base_time + timedelta(**{self.time_unit.value: time_step})
                    monitoring_schedule.append((
                        monitor_time,
                        monitor.get("variables_to_monitor", [])
                    ))
                
                # Calculate optimal timing for each intervention variable
                optimal_timing = {}
                for intervention_time, variable, _ in interventions:
                    optimal_timing[variable] = intervention_time
                
                plan = TemporalInterventionPlan(
                    interventions=interventions,
                    expected_trajectory=None,  # Would need to simulate
                    optimal_timing=optimal_timing,
                    timing_sensitivity={},  # Would need sensitivity analysis
                    coordination_requirements=[plan_data.get("coordination_strategy", "")],
                    monitoring_schedule=monitoring_schedule
                )
                
                return plan
                
            else:
                raise ValueError("Could not parse intervention plan from LLM response")
                
        except Exception as e:
            self.logger.error(f"Failed to optimize temporal interventions: {e}")
            raise


# Convenience functions
def create_temporal_analyzer(llm_client, time_unit: TimeUnit = TimeUnit.DAYS) -> AdvancedTemporalAnalyzer:
    """Create a temporal causal analyzer."""
    return AdvancedTemporalAnalyzer(llm_client, time_unit)


async def analyze_temporal_causation(temporal_data: pd.DataFrame,
                                    variables: Dict[str, str],
                                    llm_client,
                                    time_column: str = "timestamp",
                                    time_unit: TimeUnit = TimeUnit.DAYS,
                                    max_lag: int = 10) -> TemporalAnalysisResult:
    """Quick function for temporal causal analysis."""
    analyzer = create_temporal_analyzer(llm_client, time_unit)
    return await analyzer.analyze_temporal_causation(
        temporal_data, variables, time_column, max_lag
    )