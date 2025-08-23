"""
Adaptive intervention optimization system for Tier 2 capabilities.

This module provides sophisticated intervention planning, optimization,
and adaptive decision-making for causal systems using LLM guidance
and reinforcement learning approaches.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import asyncio
import json
import time
from collections import deque
import math

from causalllm.logging import get_logger


class OptimizationObjective(Enum):
    """Available optimization objectives."""
    MAXIMIZE_OUTCOME = "maximize_outcome"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_UTILITY = "maximize_utility"
    MINIMIZE_RISK = "minimize_risk"
    MULTI_OBJECTIVE = "multi_objective"


class InterventionType(Enum):
    """Types of interventions."""
    SINGLE_VARIABLE = "single_variable"
    MULTI_VARIABLE = "multi_variable"
    SEQUENTIAL = "sequential"
    CONDITIONAL = "conditional"
    ADAPTIVE = "adaptive"


class ConstraintType(Enum):
    """Types of optimization constraints."""
    BUDGET = "budget"
    FEASIBILITY = "feasibility"
    ETHICAL = "ethical"
    TEMPORAL = "temporal"
    RESOURCE = "resource"


@dataclass
class InterventionAction:
    """Represents a single intervention action."""
    
    variable: str
    value: Any
    cost: float = 0.0
    feasibility_score: float = 1.0
    ethical_score: float = 1.0
    temporal_constraint: Optional[str] = None
    expected_effect: Optional[float] = None
    uncertainty: Optional[float] = None


@dataclass
class InterventionPlan:
    """A complete intervention plan with multiple actions."""
    
    actions: List[InterventionAction]
    total_cost: float
    expected_outcome: float
    confidence_score: float
    risk_score: float
    plan_type: InterventionType
    reasoning: str
    constraints_satisfied: Dict[ConstraintType, bool] = field(default_factory=dict)
    execution_timeline: Optional[List[Tuple[str, str]]] = None  # (time, action_description)


@dataclass
class OptimizationConstraint:
    """Represents an optimization constraint."""
    
    constraint_type: ConstraintType
    description: str
    value: Any
    soft_constraint: bool = False
    penalty_weight: float = 1.0


@dataclass
class OptimizationResult:
    """Result from intervention optimization."""
    
    optimal_plan: InterventionPlan
    alternative_plans: List[InterventionPlan]
    optimization_metrics: Dict[str, float]
    convergence_info: Dict[str, Any]
    reasoning_trace: List[str]
    time_taken: float
    iterations: int


@dataclass
class AdaptiveState:
    """State for adaptive intervention optimization."""
    
    current_outcomes: Dict[str, float]
    intervention_history: List[InterventionPlan]
    learned_parameters: Dict[str, float]
    environment_changes: List[Dict[str, Any]]
    adaptation_triggers: List[str]
    confidence_evolution: List[float]


class InterventionOptimizer(ABC):
    """Abstract base class for intervention optimization algorithms."""
    
    def __init__(self, objective: OptimizationObjective):
        self.objective = objective
        self.logger = get_logger(f"causalllm.intervention_optimizer.{objective.value}")
    
    @abstractmethod
    async def optimize(self, 
                      variables: Dict[str, str],
                      causal_graph: Any,
                      target_outcome: str,
                      constraints: List[OptimizationConstraint],
                      data: Optional[pd.DataFrame] = None,
                      **kwargs) -> OptimizationResult:
        """Optimize interventions for given objective."""
        pass


class LLMGuidedOptimizer(InterventionOptimizer):
    """LLM-guided intervention optimization using causal reasoning."""
    
    def __init__(self, llm_client, objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_UTILITY):
        super().__init__(objective)
        self.llm_client = llm_client
        self.intervention_database = self._initialize_intervention_database()
    
    def _initialize_intervention_database(self) -> List[Dict[str, Any]]:
        """Initialize database of intervention strategies and examples."""
        return [
            {
                "domain": "healthcare",
                "intervention_type": "medication_dosage",
                "strategy": "gradual_increase",
                "success_factors": ["patient_compliance", "side_effects_monitoring", "regular_checkups"],
                "typical_outcomes": {"efficacy": 0.75, "side_effects": 0.15}
            },
            {
                "domain": "marketing",
                "intervention_type": "campaign_optimization",
                "strategy": "multi_channel_approach",
                "success_factors": ["audience_segmentation", "message_personalization", "timing"],
                "typical_outcomes": {"conversion_rate": 0.12, "cost_per_acquisition": 25.0}
            },
            {
                "domain": "economics",
                "intervention_type": "policy_intervention",
                "strategy": "phased_implementation",
                "success_factors": ["stakeholder_buy_in", "resource_allocation", "monitoring_system"],
                "typical_outcomes": {"policy_adoption": 0.65, "unintended_consequences": 0.20}
            },
            {
                "domain": "education",
                "intervention_type": "curriculum_change",
                "strategy": "pilot_and_scale",
                "success_factors": ["teacher_training", "student_engagement", "parental_support"],
                "typical_outcomes": {"learning_improvement": 0.18, "implementation_cost": 15000}
            }
        ]
    
    async def optimize(self, 
                      variables: Dict[str, str],
                      causal_graph: Any,
                      target_outcome: str,
                      constraints: List[OptimizationConstraint],
                      data: Optional[pd.DataFrame] = None,
                      domain_context: str = "",
                      **kwargs) -> OptimizationResult:
        """Optimize interventions using LLM guidance."""
        self.logger.info("Starting LLM-guided intervention optimization")
        
        start_time = time.time()
        reasoning_trace = []
        
        try:
            # Step 1: Analyze causal structure and identify leverage points
            leverage_points = await self._identify_leverage_points(
                variables, causal_graph, target_outcome, reasoning_trace
            )
            
            # Step 2: Generate intervention candidates
            intervention_candidates = await self._generate_intervention_candidates(
                leverage_points, variables, constraints, domain_context, reasoning_trace
            )
            
            # Step 3: Evaluate and optimize intervention plans
            optimal_plan, alternative_plans = await self._optimize_intervention_plans(
                intervention_candidates, constraints, target_outcome, data, reasoning_trace
            )
            
            # Step 4: Validate and refine plans
            optimal_plan, alternative_plans = await self._validate_and_refine_plans(
                optimal_plan, alternative_plans, constraints, reasoning_trace
            )
            
            # Calculate optimization metrics
            optimization_metrics = {
                "leverage_points_identified": len(leverage_points),
                "candidates_generated": len(intervention_candidates),
                "plans_evaluated": len(alternative_plans) + 1,
                "constraint_satisfaction_rate": self._calculate_constraint_satisfaction(optimal_plan, constraints),
                "expected_roi": optimal_plan.expected_outcome / max(optimal_plan.total_cost, 0.01)
            }
            
            convergence_info = {
                "method": "llm_guided_search",
                "iterations": 1,
                "convergence_criterion": "llm_confidence",
                "final_confidence": optimal_plan.confidence_score
            }
            
            end_time = time.time()
            
            return OptimizationResult(
                optimal_plan=optimal_plan,
                alternative_plans=alternative_plans,
                optimization_metrics=optimization_metrics,
                convergence_info=convergence_info,
                reasoning_trace=reasoning_trace,
                time_taken=end_time - start_time,
                iterations=1
            )
            
        except Exception as e:
            self.logger.error(f"LLM-guided optimization failed: {e}")
            raise
    
    async def _identify_leverage_points(self, variables: Dict[str, str],
                                      causal_graph: Any, 
                                      target_outcome: str,
                                      reasoning_trace: List[str]) -> List[Dict[str, Any]]:
        """Identify high-leverage intervention points using LLM analysis."""
        reasoning_trace.append("Identifying causal leverage points")
        
        # Extract graph information
        if hasattr(causal_graph, 'graph'):
            edges = list(causal_graph.graph.edges())
            nodes = list(causal_graph.graph.nodes())
        else:
            edges = []
            nodes = list(variables.keys())
        
        prompt = f"""
        You are an expert in causal intervention design. Analyze the causal system to identify
        the most effective intervention points.
        
        CAUSAL SYSTEM:
        - Target outcome: {target_outcome}
        - Variables: {list(variables.keys())}
        - Causal edges: {edges}
        
        VARIABLE DESCRIPTIONS:
        """
        
        for var, desc in variables.items():
            prompt += f"\n- {var}: {desc}"
        
        prompt += f"""
        
        TASK: Identify the top 3-5 variables that would be most effective intervention points
        to influence '{target_outcome}'. Consider:
        1. Direct causal paths to the outcome
        2. Indirect effects through mediating variables  
        3. Potential for high impact with reasonable effort
        4. Feasibility of intervention
        
        Respond with a JSON list of leverage points:
        [
          {{
            "variable": "variable_name",
            "leverage_score": 0.8,
            "reasoning": "Why this is a good intervention point",
            "intervention_type": "direct/indirect/mediating",
            "expected_impact": "high/medium/low",
            "feasibility": "high/medium/low"
          }}
        ]
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
                leverage_points = json.loads(json_match.group())
                reasoning_trace.append(f"Identified {len(leverage_points)} leverage points")
                
                for point in leverage_points:
                    reasoning_trace.append(
                        f"Leverage point: {point['variable']} "
                        f"(score: {point.get('leverage_score', 0):.2f}, "
                        f"impact: {point.get('expected_impact', 'unknown')})"
                    )
                
                return leverage_points
            else:
                reasoning_trace.append("Warning: Could not parse leverage points from LLM response")
                return []
        
        except Exception as e:
            self.logger.error(f"Failed to identify leverage points: {e}")
            reasoning_trace.append(f"Error identifying leverage points: {e}")
            return []
    
    async def _generate_intervention_candidates(self, leverage_points: List[Dict[str, Any]],
                                              variables: Dict[str, str],
                                              constraints: List[OptimizationConstraint],
                                              domain_context: str,
                                              reasoning_trace: List[str]) -> List[InterventionPlan]:
        """Generate intervention candidates based on leverage points."""
        reasoning_trace.append("Generating intervention candidates")
        
        if not leverage_points:
            reasoning_trace.append("No leverage points available, generating default candidates")
            # Generate simple single-variable interventions for all variables
            candidates = []
            for var in variables.keys():
                plan = InterventionPlan(
                    actions=[InterventionAction(variable=var, value="optimized", cost=100.0)],
                    total_cost=100.0,
                    expected_outcome=0.5,
                    confidence_score=0.3,
                    risk_score=0.5,
                    plan_type=InterventionType.SINGLE_VARIABLE,
                    reasoning=f"Default intervention on {var}"
                )
                candidates.append(plan)
            return candidates
        
        # Create constraint summary for prompting
        constraint_summary = []
        for constraint in constraints:
            constraint_summary.append(f"{constraint.constraint_type.value}: {constraint.description}")
        
        prompt = f"""
        Design specific intervention plans based on the identified leverage points.
        
        LEVERAGE POINTS:
        """
        
        for point in leverage_points:
            prompt += f"\n- {point['variable']}: {point['reasoning']} (impact: {point.get('expected_impact', 'unknown')})"
        
        if domain_context:
            prompt += f"\n\nDOMAIN CONTEXT: {domain_context}"
        
        if constraint_summary:
            prompt += f"\n\nCONSTRAINTS:\n" + "\n".join(f"- {c}" for c in constraint_summary)
        
        prompt += """
        
        TASK: Design 3-5 specific intervention plans. Include:
        1. Single-variable interventions (focus on one leverage point)
        2. Multi-variable interventions (combine leverage points)
        3. At least one low-cost/low-risk option
        4. At least one high-impact option
        
        Respond with JSON:
        [
          {
            "plan_name": "descriptive name",
            "actions": [
              {
                "variable": "variable_name", 
                "intervention": "specific intervention description",
                "estimated_cost": 1000,
                "feasibility": 0.8,
                "expected_effect": 0.6
              }
            ],
            "total_cost": 1000,
            "expected_outcome": 0.7,
            "confidence": 0.8,
            "risk_level": 0.3,
            "reasoning": "why this plan would be effective"
          }
        ]
        """
        
        try:
            if hasattr(self.llm_client, 'generate_response'):
                response = await self.llm_client.generate_response(prompt)
            else:
                response = await asyncio.to_thread(self.llm_client.generate, prompt)
            
            # Parse and convert to InterventionPlan objects
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                plans_data = json.loads(json_match.group())
                
                candidates = []
                for plan_data in plans_data:
                    actions = []
                    for action_data in plan_data.get("actions", []):
                        action = InterventionAction(
                            variable=action_data["variable"],
                            value=action_data.get("intervention", "optimized"),
                            cost=action_data.get("estimated_cost", 0.0),
                            feasibility_score=action_data.get("feasibility", 1.0),
                            expected_effect=action_data.get("expected_effect")
                        )
                        actions.append(action)
                    
                    plan = InterventionPlan(
                        actions=actions,
                        total_cost=plan_data.get("total_cost", 0.0),
                        expected_outcome=plan_data.get("expected_outcome", 0.5),
                        confidence_score=plan_data.get("confidence", 0.5),
                        risk_score=plan_data.get("risk_level", 0.5),
                        plan_type=InterventionType.MULTI_VARIABLE if len(actions) > 1 else InterventionType.SINGLE_VARIABLE,
                        reasoning=plan_data.get("reasoning", "LLM-generated intervention plan")
                    )
                    candidates.append(plan)
                
                reasoning_trace.append(f"Generated {len(candidates)} intervention candidates")
                return candidates
            
            else:
                reasoning_trace.append("Could not parse intervention candidates from LLM response")
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to generate intervention candidates: {e}")
            reasoning_trace.append(f"Error generating candidates: {e}")
            return []
    
    async def _optimize_intervention_plans(self, candidates: List[InterventionPlan],
                                         constraints: List[OptimizationConstraint],
                                         target_outcome: str,
                                         data: Optional[pd.DataFrame],
                                         reasoning_trace: List[str]) -> Tuple[InterventionPlan, List[InterventionPlan]]:
        """Optimize and rank intervention plans."""
        reasoning_trace.append("Optimizing and ranking intervention plans")
        
        if not candidates:
            reasoning_trace.append("No candidates to optimize")
            # Return a default plan
            default_plan = InterventionPlan(
                actions=[InterventionAction(variable="default", value="none", cost=0.0)],
                total_cost=0.0,
                expected_outcome=0.1,
                confidence_score=0.1,
                risk_score=0.9,
                plan_type=InterventionType.SINGLE_VARIABLE,
                reasoning="Default plan due to no candidates available"
            )
            return default_plan, []
        
        # Score each plan based on objective and constraints
        scored_plans = []
        for plan in candidates:
            score = await self._score_intervention_plan(plan, constraints, target_outcome, data)
            scored_plans.append((score, plan))
            reasoning_trace.append(f"Plan scored {score:.3f}: {plan.reasoning[:50]}...")
        
        # Sort by score (descending)
        scored_plans.sort(key=lambda x: x[0], reverse=True)
        
        optimal_plan = scored_plans[0][1]
        alternative_plans = [plan for _, plan in scored_plans[1:]]
        
        reasoning_trace.append(f"Selected optimal plan with score {scored_plans[0][0]:.3f}")
        
        return optimal_plan, alternative_plans
    
    async def _score_intervention_plan(self, plan: InterventionPlan,
                                     constraints: List[OptimizationConstraint],
                                     target_outcome: str,
                                     data: Optional[pd.DataFrame]) -> float:
        """Score an intervention plan based on objective and constraints."""
        
        base_score = 0.0
        
        # Objective-based scoring
        if self.objective == OptimizationObjective.MAXIMIZE_OUTCOME:
            base_score = plan.expected_outcome
        elif self.objective == OptimizationObjective.MINIMIZE_COST:
            base_score = 1.0 / (1.0 + plan.total_cost / 1000.0)  # Normalize cost
        elif self.objective == OptimizationObjective.MAXIMIZE_UTILITY:
            utility = plan.expected_outcome - (plan.total_cost / 10000.0) - plan.risk_score * 0.5
            base_score = max(0, utility)
        elif self.objective == OptimizationObjective.MINIMIZE_RISK:
            base_score = 1.0 - plan.risk_score
        
        # Apply constraint penalties
        constraint_penalty = 0.0
        for constraint in constraints:
            violation_penalty = await self._calculate_constraint_violation(plan, constraint)
            constraint_penalty += violation_penalty * constraint.penalty_weight
        
        # Confidence bonus
        confidence_bonus = plan.confidence_score * 0.2
        
        final_score = base_score - constraint_penalty + confidence_bonus
        return max(0, final_score)
    
    async def _calculate_constraint_violation(self, plan: InterventionPlan,
                                            constraint: OptimizationConstraint) -> float:
        """Calculate constraint violation penalty."""
        
        if constraint.constraint_type == ConstraintType.BUDGET:
            budget_limit = float(constraint.value)
            if plan.total_cost > budget_limit:
                excess = (plan.total_cost - budget_limit) / budget_limit
                return excess if not constraint.soft_constraint else excess * 0.5
        
        elif constraint.constraint_type == ConstraintType.FEASIBILITY:
            min_feasibility = float(constraint.value)
            avg_feasibility = sum(action.feasibility_score for action in plan.actions) / len(plan.actions)
            if avg_feasibility < min_feasibility:
                return (min_feasibility - avg_feasibility) * 2.0
        
        elif constraint.constraint_type == ConstraintType.ETHICAL:
            min_ethical_score = float(constraint.value)
            avg_ethical = sum(action.ethical_score for action in plan.actions) / len(plan.actions)
            if avg_ethical < min_ethical_score:
                return (min_ethical_score - avg_ethical) * 3.0  # High penalty for ethical violations
        
        return 0.0
    
    async def _validate_and_refine_plans(self, optimal_plan: InterventionPlan,
                                       alternative_plans: List[InterventionPlan],
                                       constraints: List[OptimizationConstraint],
                                       reasoning_trace: List[str]) -> Tuple[InterventionPlan, List[InterventionPlan]]:
        """Validate and refine intervention plans."""
        reasoning_trace.append("Validating and refining intervention plans")
        
        # Check constraint satisfaction for optimal plan
        constraint_satisfaction = {}
        for constraint in constraints:
            violation = await self._calculate_constraint_violation(optimal_plan, constraint)
            constraint_satisfaction[constraint.constraint_type] = violation == 0.0
        
        optimal_plan.constraints_satisfied = constraint_satisfaction
        
        # Generate execution timeline for optimal plan
        optimal_plan.execution_timeline = self._generate_execution_timeline(optimal_plan)
        
        # Adjust confidence based on constraint satisfaction
        satisfaction_rate = sum(constraint_satisfaction.values()) / len(constraint_satisfaction) if constraint_satisfaction else 1.0
        optimal_plan.confidence_score *= satisfaction_rate
        
        reasoning_trace.append(f"Constraint satisfaction rate: {satisfaction_rate:.2f}")
        reasoning_trace.append(f"Final confidence: {optimal_plan.confidence_score:.3f}")
        
        return optimal_plan, alternative_plans
    
    def _generate_execution_timeline(self, plan: InterventionPlan) -> List[Tuple[str, str]]:
        """Generate execution timeline for intervention plan."""
        timeline = []
        
        # Simple timeline based on intervention complexity
        for i, action in enumerate(plan.actions):
            if len(plan.actions) == 1:
                timeline.append(("T+0", f"Implement intervention on {action.variable}"))
            else:
                timeline.append((f"T+{i*2}", f"Implement intervention on {action.variable}"))
                if i < len(plan.actions) - 1:
                    timeline.append((f"T+{i*2+1}", "Monitor and assess intermediate effects"))
        
        timeline.append(("T+final", "Evaluate overall intervention outcomes"))
        
        return timeline
    
    def _calculate_constraint_satisfaction(self, plan: InterventionPlan, 
                                         constraints: List[OptimizationConstraint]) -> float:
        """Calculate overall constraint satisfaction rate."""
        if not constraints:
            return 1.0
        
        satisfied_constraints = sum(plan.constraints_satisfied.values()) if plan.constraints_satisfied else 0
        return satisfied_constraints / len(constraints)


class AdaptiveInterventionOptimizer:
    """Adaptive optimizer that learns from intervention outcomes."""
    
    def __init__(self, base_optimizer: InterventionOptimizer, learning_rate: float = 0.1):
        self.base_optimizer = base_optimizer
        self.learning_rate = learning_rate
        self.logger = get_logger("causalllm.adaptive_intervention_optimizer")
        
        # Adaptive state
        self.adaptive_state = AdaptiveState(
            current_outcomes={},
            intervention_history=[],
            learned_parameters={},
            environment_changes=[],
            adaptation_triggers=[],
            confidence_evolution=[]
        )
        
        # Learning parameters
        self.outcome_prediction_model = {}
        self.intervention_effectiveness = {}
        self.environment_drift_detector = deque(maxlen=10)  # Recent outcome tracking
    
    async def adaptive_optimize(self, 
                              variables: Dict[str, str],
                              causal_graph: Any,
                              target_outcome: str,
                              constraints: List[OptimizationConstraint],
                              current_state: Dict[str, float],
                              data: Optional[pd.DataFrame] = None,
                              **kwargs) -> OptimizationResult:
        """Perform adaptive optimization considering historical performance."""
        self.logger.info("Starting adaptive intervention optimization")
        
        # Update environment understanding
        await self._update_environment_model(current_state)
        
        # Adapt optimization parameters based on learning
        adapted_constraints = await self._adapt_constraints(constraints)
        
        # Run base optimization with adaptations
        result = await self.base_optimizer.optimize(
            variables=variables,
            causal_graph=causal_graph,
            target_outcome=target_outcome,
            constraints=adapted_constraints,
            data=data,
            **kwargs
        )
        
        # Apply adaptive refinements
        result = await self._apply_adaptive_refinements(result, current_state)
        
        # Store result for learning
        self.adaptive_state.intervention_history.append(result.optimal_plan)
        self.adaptive_state.confidence_evolution.append(result.optimal_plan.confidence_score)
        
        return result
    
    async def update_with_outcome(self, intervention_plan: InterventionPlan, 
                                actual_outcome: float,
                                outcome_context: Dict[str, Any] = None):
        """Update the adaptive model with actual intervention outcomes."""
        self.logger.info(f"Updating adaptive model with outcome: {actual_outcome}")
        
        # Store actual outcome
        plan_key = self._get_plan_key(intervention_plan)
        self.adaptive_state.current_outcomes[plan_key] = actual_outcome
        
        # Update prediction model
        predicted_outcome = intervention_plan.expected_outcome
        prediction_error = abs(actual_outcome - predicted_outcome)
        
        # Learn from prediction error
        if plan_key not in self.outcome_prediction_model:
            self.outcome_prediction_model[plan_key] = {"predictions": [], "actuals": [], "errors": []}
        
        self.outcome_prediction_model[plan_key]["predictions"].append(predicted_outcome)
        self.outcome_prediction_model[plan_key]["actuals"].append(actual_outcome)
        self.outcome_prediction_model[plan_key]["errors"].append(prediction_error)
        
        # Update intervention effectiveness
        for action in intervention_plan.actions:
            if action.variable not in self.intervention_effectiveness:
                self.intervention_effectiveness[action.variable] = {"outcomes": [], "costs": [], "roi": []}
            
            self.intervention_effectiveness[action.variable]["outcomes"].append(actual_outcome)
            self.intervention_effectiveness[action.variable]["costs"].append(action.cost)
            
            roi = (actual_outcome - predicted_outcome) / max(action.cost, 1.0)
            self.intervention_effectiveness[action.variable]["roi"].append(roi)
        
        # Environment drift detection
        self.environment_drift_detector.append(actual_outcome)
        await self._detect_environment_drift()
        
        # Update learned parameters
        self._update_learned_parameters()
    
    async def _update_environment_model(self, current_state: Dict[str, float]):
        """Update model of the environment based on current state."""
        # Simple environment change detection
        if hasattr(self, 'previous_state'):
            state_changes = {}
            for var, value in current_state.items():
                if var in self.previous_state:
                    change = abs(value - self.previous_state[var])
                    state_changes[var] = change
            
            # Detect significant changes
            significant_changes = {var: change for var, change in state_changes.items() if change > 0.1}
            
            if significant_changes:
                self.adaptive_state.environment_changes.append({
                    "timestamp": time.time(),
                    "changes": significant_changes
                })
                self.logger.info(f"Detected environment changes: {list(significant_changes.keys())}")
        
        self.previous_state = current_state.copy()
    
    async def _adapt_constraints(self, original_constraints: List[OptimizationConstraint]) -> List[OptimizationConstraint]:
        """Adapt constraints based on learned experiences."""
        adapted_constraints = []
        
        for constraint in original_constraints:
            adapted_constraint = constraint
            
            # Adapt budget constraints based on ROI learning
            if constraint.constraint_type == ConstraintType.BUDGET:
                if self.intervention_effectiveness:
                    avg_roi = np.mean([
                        np.mean(data["roi"]) for data in self.intervention_effectiveness.values()
                        if data["roi"]
                    ])
                    
                    if avg_roi > 1.5:  # Good ROI, relax budget slightly
                        adapted_constraint.value = float(constraint.value) * 1.1
                    elif avg_roi < 0.5:  # Poor ROI, tighten budget
                        adapted_constraint.value = float(constraint.value) * 0.9
            
            adapted_constraints.append(adapted_constraint)
        
        return adapted_constraints
    
    async def _apply_adaptive_refinements(self, result: OptimizationResult, 
                                        current_state: Dict[str, float]) -> OptimizationResult:
        """Apply adaptive refinements to optimization results."""
        
        # Adjust confidence based on historical accuracy
        if self.outcome_prediction_model:
            avg_error = np.mean([
                np.mean(data["errors"]) for data in self.outcome_prediction_model.values()
                if data["errors"]
            ])
            
            # Reduce confidence if predictions have been poor
            confidence_adjustment = max(0.5, 1.0 - avg_error)
            result.optimal_plan.confidence_score *= confidence_adjustment
        
        # Adjust expected outcome based on intervention effectiveness
        for action in result.optimal_plan.actions:
            if action.variable in self.intervention_effectiveness:
                effectiveness_data = self.intervention_effectiveness[action.variable]
                if effectiveness_data["outcomes"]:
                    avg_effectiveness = np.mean(effectiveness_data["outcomes"])
                    # Blend prediction with historical effectiveness
                    action.expected_effect = (
                        0.7 * (action.expected_effect or 0.5) + 
                        0.3 * avg_effectiveness
                    )
        
        return result
    
    async def _detect_environment_drift(self):
        """Detect if the environment has significantly changed."""
        if len(self.environment_drift_detector) < 5:
            return
        
        recent_outcomes = list(self.environment_drift_detector)
        recent_avg = np.mean(recent_outcomes[-3:])
        historical_avg = np.mean(recent_outcomes[:-3])
        
        # Detect significant drift
        if abs(recent_avg - historical_avg) > 0.2:
            self.adaptive_state.adaptation_triggers.append(f"Environment drift detected at {time.time()}")
            self.logger.info("Environment drift detected - adapting parameters")
            
            # Reset some learned parameters due to environment change
            self._reset_outdated_learning()
    
    def _reset_outdated_learning(self):
        """Reset learning parameters that may be outdated due to environment changes."""
        # Keep only recent data
        for var in self.intervention_effectiveness:
            data = self.intervention_effectiveness[var]
            if len(data["outcomes"]) > 10:
                # Keep only most recent 70%
                keep_n = int(len(data["outcomes"]) * 0.7)
                data["outcomes"] = data["outcomes"][-keep_n:]
                data["costs"] = data["costs"][-keep_n:]
                data["roi"] = data["roi"][-keep_n:]
    
    def _update_learned_parameters(self):
        """Update learned parameters based on accumulated experience."""
        # Update effectiveness scores
        for var, data in self.intervention_effectiveness.items():
            if data["roi"]:
                self.adaptive_state.learned_parameters[f"{var}_effectiveness"] = np.mean(data["roi"])
                self.adaptive_state.learned_parameters[f"{var}_reliability"] = 1.0 - np.std(data["roi"])
        
        # Update prediction accuracy
        if self.outcome_prediction_model:
            overall_accuracy = 1.0 - np.mean([
                np.mean(data["errors"]) for data in self.outcome_prediction_model.values()
                if data["errors"]
            ])
            self.adaptive_state.learned_parameters["prediction_accuracy"] = overall_accuracy
    
    def _get_plan_key(self, plan: InterventionPlan) -> str:
        """Generate a unique key for an intervention plan."""
        action_keys = []
        for action in plan.actions:
            action_keys.append(f"{action.variable}:{action.value}")
        return "|".join(sorted(action_keys))
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptive learning progress."""
        return {
            "interventions_tried": len(self.adaptive_state.intervention_history),
            "environment_changes_detected": len(self.adaptive_state.environment_changes),
            "adaptation_triggers": len(self.adaptive_state.adaptation_triggers),
            "learned_parameters": dict(self.adaptive_state.learned_parameters),
            "intervention_effectiveness": {
                var: {
                    "avg_outcome": np.mean(data["outcomes"]) if data["outcomes"] else 0,
                    "avg_roi": np.mean(data["roi"]) if data["roi"] else 0,
                    "reliability": 1.0 - np.std(data["outcomes"]) if len(data["outcomes"]) > 1 else 0
                }
                for var, data in self.intervention_effectiveness.items()
            },
            "current_prediction_accuracy": self.adaptive_state.learned_parameters.get("prediction_accuracy", 0.5)
        }


# Convenience functions
def create_intervention_optimizer(llm_client=None,
                                objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_UTILITY) -> InterventionOptimizer:
    """Create an intervention optimizer."""
    if llm_client is None:
        raise ValueError("LLM client required for intervention optimization")
    
    return LLMGuidedOptimizer(llm_client, objective)


async def optimize_intervention(variables: Dict[str, str],
                              causal_graph: Any,
                              target_outcome: str,
                              constraints: Optional[List[OptimizationConstraint]] = None,
                              llm_client=None,
                              objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_UTILITY,
                              **kwargs) -> OptimizationResult:
    """Quick function to optimize interventions."""
    optimizer = create_intervention_optimizer(llm_client, objective)
    return await optimizer.optimize(
        variables=variables,
        causal_graph=causal_graph,
        target_outcome=target_outcome,
        constraints=constraints or [],
        **kwargs
    )