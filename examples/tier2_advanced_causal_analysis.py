"""
Comprehensive example demonstrating Tier 2 advanced causal analysis capabilities.

This example showcases:
- Advanced causal discovery with LLM guidance
- Adaptive intervention optimization
- Temporal causal modeling
- Sophisticated causal explanations
- Integration with external libraries
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalllm.core import CausalLLMCore
from causalllm.llm_client import get_llm_client
from causalllm.logging import get_logger

# Tier 2 imports
from causalllm.causal_discovery import (
    AdvancedCausalDiscovery, 
    DiscoveryMethod,
    discover_causal_structure
)
from causalllm.intervention_optimizer import (
    LLMGuidedOptimizer,
    AdaptiveInterventionOptimizer,
    OptimizationObjective,
    OptimizationConstraint,
    ConstraintType,
    optimize_intervention
)
from causalllm.temporal_causal_modeling import (
    AdvancedTemporalAnalyzer,
    TimeUnit,
    analyze_temporal_causation
)
from causalllm.causal_explanation_generator import (
    LLMExplanationEngine,
    ExplanationType,
    ExplanationAudience,
    ExplanationModality,
    generate_causal_explanation
)
from causalllm.external_integrations import (
    UniversalExternalIntegrator,
    ExternalLibrary,
    IntegrationMethod
)

warnings.filterwarnings('ignore')


async def demonstrate_causal_discovery():
    """Demonstrate advanced causal discovery capabilities."""
    logger = get_logger("tier2_example.causal_discovery")
    
    print("=== Advanced Causal Discovery Demo ===\n")
    
    # Create synthetic healthcare dataset
    np.random.seed(42)
    n_patients = 500
    
    # Simulate realistic healthcare data
    age = np.random.normal(55, 15, n_patients)
    age = np.clip(age, 18, 90)
    
    # Comorbidities influenced by age
    comorbidities = np.random.poisson(0.1 * age / 10, n_patients)
    
    # Treatment assignment (somewhat influenced by age and comorbidities)
    treatment_prob = 0.3 + 0.01 * age / 10 + 0.05 * comorbidities
    treatment = np.random.binomial(1, np.clip(treatment_prob, 0.1, 0.9), n_patients)
    
    # Medication adherence influenced by age and treatment
    adherence = np.where(treatment == 1, 
                        np.random.beta(2, 1, n_patients),  # Higher adherence when treated
                        np.random.beta(1, 2, n_patients))  # Lower when not treated
    
    # Recovery outcome influenced by treatment, adherence, age, and comorbidities
    recovery_score = (
        30 * treatment * adherence +      # Strong treatment effect with adherence
        -0.2 * age +                     # Age negatively affects recovery
        -5 * comorbidities +             # Comorbidities hurt recovery
        np.random.normal(0, 10, n_patients)  # Random noise
    )
    recovery_score = np.clip(recovery_score, 0, 100)
    
    # Create DataFrame
    data = pd.DataFrame({
        'patient_age': age,
        'comorbidities': comorbidities,
        'treatment_assigned': treatment,
        'medication_adherence': adherence,
        'recovery_score': recovery_score
    })
    
    variables = {
        'patient_age': 'Age of patient in years',
        'comorbidities': 'Number of comorbid conditions',
        'treatment_assigned': 'Whether patient was assigned to treatment (1) or control (0)',
        'medication_adherence': 'Medication adherence rate (0-1)',
        'recovery_score': 'Patient recovery score (0-100)'
    }
    
    print(f"Created synthetic healthcare dataset with {len(data)} patients")
    print(f"Variables: {list(variables.keys())}\n")
    
    try:
        llm_client = get_llm_client()
        
        # Initialize causal discovery system
        discovery_system = AdvancedCausalDiscovery(llm_client)
        
        print("🔍 Running LLM-guided causal discovery...")
        
        # Discover causal structure
        result = await discovery_system.discover(
            data=data,
            variables=variables,
            method=DiscoveryMethod.HYBRID_LLM,
            domain_context="healthcare treatment effectiveness study"
        )
        
        print("✅ Causal discovery completed!\n")
        
        print("📊 DISCOVERY RESULTS")
        print("=" * 50)
        print(f"Method used: {result.method_used.value}")
        print(f"Discovered edges: {len(result.discovered_edges)}")
        print(f"Rejected edges: {len(result.rejected_edges)}")
        print(f"Time taken: {result.time_taken:.2f} seconds\n")
        
        print("🔗 DISCOVERED CAUSAL EDGES:")
        for edge in result.discovered_edges:
            print(f"  {edge.cause} → {edge.effect}")
            print(f"    Confidence: {edge.confidence:.3f}")
            print(f"    Reasoning: {edge.reasoning[:80]}...")
            print()
        
        print("📈 DISCOVERY METRICS:")
        for metric, value in result.discovery_metrics.items():
            print(f"  {metric}: {value}")
        
        logger.info("Causal discovery demonstration completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Causal discovery failed: {e}")
        print(f"❌ Discovery failed: {e}")
        return None


async def demonstrate_intervention_optimization():
    """Demonstrate adaptive intervention optimization."""
    logger = get_logger("tier2_example.intervention_optimization")
    
    print("\n\n=== Adaptive Intervention Optimization Demo ===\n")
    
    # Define marketing campaign scenario
    variables = {
        'budget_allocation': 'Marketing budget allocation strategy',
        'channel_mix': 'Mix of marketing channels (digital vs traditional)',
        'message_personalization': 'Level of message personalization',
        'timing_optimization': 'Campaign timing and frequency',
        'customer_acquisition': 'Number of new customers acquired',
        'customer_lifetime_value': 'Average customer lifetime value'
    }
    
    # Simple DAG for marketing scenario
    dag_edges = [
        ('budget_allocation', 'channel_mix'),
        ('budget_allocation', 'message_personalization'),
        ('channel_mix', 'customer_acquisition'),
        ('message_personalization', 'customer_acquisition'),
        ('timing_optimization', 'customer_acquisition'),
        ('customer_acquisition', 'customer_lifetime_value')
    ]
    
    print("📱 Marketing Campaign Optimization Scenario")
    print(f"Variables: {list(variables.keys())}")
    print(f"Target outcome: customer_lifetime_value\n")
    
    try:
        llm_client = get_llm_client()
        
        # Create core for graph structure (simplified)
        core = CausalLLMCore(
            context="Marketing campaign optimization for e-commerce platform",
            variables=variables,
            dag_edges=dag_edges,
            llm_client=llm_client
        )
        
        # Define optimization constraints
        constraints = [
            OptimizationConstraint(
                constraint_type=ConstraintType.BUDGET,
                description="Total marketing budget limit",
                value=50000.0,
                soft_constraint=False
            ),
            OptimizationConstraint(
                constraint_type=ConstraintType.FEASIBILITY,
                description="Minimum feasibility score",
                value=0.7,
                soft_constraint=True,
                penalty_weight=2.0
            )
        ]
        
        print("🎯 Running intervention optimization...")
        
        # Run optimization
        result = await optimize_intervention(
            variables=variables,
            causal_graph=core.dag,
            target_outcome="customer_lifetime_value",
            constraints=constraints,
            llm_client=llm_client,
            objective=OptimizationObjective.MAXIMIZE_UTILITY,
            domain_context="e-commerce marketing campaign"
        )
        
        print("✅ Optimization completed!\n")
        
        print("📊 OPTIMIZATION RESULTS")
        print("=" * 50)
        print(f"Optimal plan expected outcome: {result.optimal_plan.expected_outcome:.3f}")
        print(f"Total cost: ${result.optimal_plan.total_cost:,.2f}")
        print(f"Confidence: {result.optimal_plan.confidence_score:.3f}")
        print(f"Risk score: {result.optimal_plan.risk_score:.3f}\n")
        
        print("🎯 RECOMMENDED INTERVENTIONS:")
        for i, action in enumerate(result.optimal_plan.actions, 1):
            print(f"  {i}. {action.variable}")
            print(f"     Action: {action.value}")
            print(f"     Cost: ${action.cost:,.2f}")
            print(f"     Feasibility: {action.feasibility_score:.3f}")
            print()
        
        print("💡 REASONING:")
        print(f"  {result.optimal_plan.reasoning}\n")
        
        print("📈 OPTIMIZATION METRICS:")
        for metric, value in result.optimization_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
        
        # Demonstrate adaptive optimization
        print("\n🤖 ADAPTIVE OPTIMIZATION:")
        base_optimizer = LLMGuidedOptimizer(llm_client, OptimizationObjective.MAXIMIZE_UTILITY)
        adaptive_optimizer = AdaptiveInterventionOptimizer(base_optimizer)
        
        # Simulate some outcomes for learning
        print("  Simulating outcomes for adaptive learning...")
        await adaptive_optimizer.update_with_outcome(
            result.optimal_plan, 
            actual_outcome=0.85,  # Good outcome
            outcome_context={"campaign_month": "January"}
        )
        
        adaptation_summary = adaptive_optimizer.get_adaptation_summary()
        print(f"  Interventions tried: {adaptation_summary['interventions_tried']}")
        print(f"  Prediction accuracy: {adaptation_summary['current_prediction_accuracy']:.3f}")
        
        logger.info("Intervention optimization demonstration completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Intervention optimization failed: {e}")
        print(f"❌ Optimization failed: {e}")
        return None


async def demonstrate_temporal_analysis():
    """Demonstrate temporal causal modeling."""
    logger = get_logger("tier2_example.temporal_analysis")
    
    print("\n\n=== Temporal Causal Modeling Demo ===\n")
    
    # Generate synthetic time series data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    n_days = len(dates)
    
    # Simulate business metrics over time
    base_marketing_spend = 1000 + 200 * np.sin(2 * np.pi * np.arange(n_days) / 365)  # Seasonal
    marketing_spend = base_marketing_spend + np.random.normal(0, 100, n_days)
    
    # Website traffic influenced by marketing (with lag)
    website_traffic = np.zeros(n_days)
    for i in range(n_days):
        lag_effect = sum(marketing_spend[max(0, i-7):i] * np.exp(-0.2 * np.arange(min(7, i))))
        website_traffic[i] = 500 + 0.3 * lag_effect + np.random.normal(0, 50)
    
    # Conversions influenced by traffic (same day) and marketing (lagged)
    conversions = np.zeros(n_days)
    for i in range(n_days):
        marketing_lag = marketing_spend[max(0, i-3)] if i >= 3 else 0
        conversions[i] = (
            10 + 
            0.05 * website_traffic[i] + 
            0.01 * marketing_lag +
            np.random.normal(0, 5)
        )
    conversions = np.maximum(0, conversions)
    
    # Revenue influenced by conversions and external factors
    external_factor = 100 * np.sin(2 * np.pi * np.arange(n_days) / 30)  # Monthly cycle
    revenue = 1000 * conversions + external_factor + np.random.normal(0, 500, n_days)
    
    # Create temporal dataset
    temporal_data = pd.DataFrame({
        'timestamp': dates,
        'marketing_spend': marketing_spend,
        'website_traffic': website_traffic,
        'conversions': conversions,
        'revenue': revenue
    })
    
    variables = {
        'marketing_spend': 'Daily marketing expenditure',
        'website_traffic': 'Daily unique website visitors',
        'conversions': 'Daily conversion events',
        'revenue': 'Daily revenue generated'
    }
    
    print("📈 E-commerce Time Series Analysis")
    print(f"Time period: {dates[0].date()} to {dates[-1].date()}")
    print(f"Data points: {len(temporal_data)}")
    print(f"Variables: {list(variables.keys())}\n")
    
    try:
        llm_client = get_llm_client()
        
        print("⏱️ Running temporal causal analysis...")
        
        # Analyze temporal causation
        result = await analyze_temporal_causation(
            temporal_data=temporal_data,
            variables=variables,
            llm_client=llm_client,
            time_column='timestamp',
            time_unit=TimeUnit.DAYS,
            max_lag=7
        )
        
        print("✅ Temporal analysis completed!\n")
        
        print("📊 TEMPORAL ANALYSIS RESULTS")
        print("=" * 50)
        print(f"Temporal edges discovered: {len(result.temporal_edges)}")
        print(f"Data timespan: {result.analysis_metadata['data_timespan']['duration']} days")
        print(f"Max lag analyzed: {result.analysis_metadata['max_lag_analyzed']}")
        print(f"Temporal confounders: {len(result.temporal_confounders)}\n")
        
        print("⏰ TEMPORAL CAUSAL EDGES:")
        for edge in result.temporal_edges:
            print(f"  {edge.cause} → {edge.effect}")
            print(f"    Lag: {edge.lag} {edge.time_unit.value}")
            print(f"    Strength: {edge.strength:.3f}")
            print(f"    Confidence: {edge.confidence:.3f}")
            print(f"    Mechanism: {edge.mechanism.value}")
            if edge.evidence and 'llm_reasoning' in edge.evidence:
                reasoning = edge.evidence['llm_reasoning'][:60] + "..."
                print(f"    Reasoning: {reasoning}")
            print()
        
        print("🔍 TEMPORAL CONFOUNDERS:")
        for confounder in result.temporal_confounders:
            print(f"  - {confounder}")
        
        print("\n📈 CAUSAL FORECASTS:")
        for var, trajectory in result.causal_forecasts.items():
            if len(trajectory.states) > 0:
                print(f"  {var}: {len(trajectory.states)} forecast steps")
        
        print("\n🎯 INTERVENTION WINDOWS:")
        for var, windows in result.intervention_windows.items():
            if windows:
                print(f"  {var}: {len(windows)} optimal intervention periods")
        
        logger.info("Temporal analysis demonstration completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Temporal analysis failed: {e}")
        print(f"❌ Temporal analysis failed: {e}")
        return None


async def demonstrate_causal_explanations():
    """Demonstrate advanced causal explanation generation."""
    logger = get_logger("tier2_example.causal_explanations")
    
    print("\n\n=== Advanced Causal Explanation Generation Demo ===\n")
    
    # Healthcare scenario for explanations
    causal_data = {
        "statistical_evidence": {
            "correlation": 0.65,
            "p_value": 0.001,
            "effect_size": 0.42
        },
        "experimental_data": {
            "sample_size": 1200,
            "effect_size": 0.38,
            "limitations": ["Short follow-up period", "Limited to urban population"]
        },
        "domain_knowledge": {
            "description": "Exercise has well-established cardiovascular benefits through multiple physiological pathways",
            "confidence": 0.9,
            "source": "American Heart Association guidelines"
        }
    }
    
    print("💊 Healthcare Causal Explanation Scenario")
    print("Explaining: Exercise → Cardiovascular Health")
    print(f"Available evidence: {list(causal_data.keys())}\n")
    
    try:
        llm_client = get_llm_client()
        
        # Test different explanation types and audiences
        explanation_tests = [
            {
                "type": ExplanationType.MECHANISM,
                "audience": ExplanationAudience.PRACTITIONER,
                "description": "Mechanism explanation for practitioners"
            },
            {
                "type": ExplanationType.COUNTERFACTUAL,
                "audience": ExplanationAudience.GENERAL,
                "description": "Counterfactual explanation for general public"
            },
            {
                "type": ExplanationType.NECESSITY,
                "audience": ExplanationAudience.EXPERT,
                "description": "Necessity analysis for experts"
            }
        ]
        
        explanations = []
        
        for test in explanation_tests:
            print(f"🔍 Generating {test['description']}...")
            
            explanation = await generate_causal_explanation(
                cause_variable="regular_exercise",
                effect_variable="cardiovascular_health",
                explanation_type=test["type"],
                audience=test["audience"],
                context="Clinical research on lifestyle interventions for cardiovascular disease prevention",
                causal_data=causal_data,
                llm_client=llm_client,
                desired_length="medium"
            )
            
            explanations.append((test["description"], explanation))
            
            print(f"✅ Generated {test['description']}\n")
        
        print("📚 EXPLANATION RESULTS")
        print("=" * 50)
        
        for desc, explanation in explanations:
            print(f"\n📖 {desc.upper()}")
            print("-" * len(desc))
            print(f"Audience: {explanation.request.audience.value}")
            print(f"Type: {explanation.request.explanation_type.value}")
            print(f"Certainty: {explanation.certainty_level.value}")
            print(f"Confidence: {explanation.confidence_score:.3f}")
            
            print(f"\nMain Explanation:")
            print(f"{explanation.main_explanation[:300]}...")
            
            if explanation.supporting_details:
                print(f"\nSupporting Details:")
                for i, detail in enumerate(explanation.supporting_details[:2], 1):
                    print(f"  {i}. {detail[:100]}...")
            
            if explanation.analogies:
                print(f"\nAnalogies:")
                for analogy in explanation.analogies[:2]:
                    print(f"  - {analogy}")
            
            if explanation.limitations:
                print(f"\nLimitations:")
                for limitation in explanation.limitations[:2]:
                    print(f"  - {limitation}")
            
            print()
        
        # Demonstrate adaptive explanation system
        print("🤖 ADAPTIVE EXPLANATION SYSTEM:")
        from causalllm.causal_explanation_generator import AdaptiveExplanationGenerator
        
        adaptive_generator = AdaptiveExplanationGenerator(llm_client)
        
        # Record feedback for learning
        sample_feedback = {
            "rating": 4,
            "length_feedback": "appropriate", 
            "jargon_feedback": "appropriate",
            "analogies_helpful": True
        }
        
        adaptive_generator.record_user_feedback(explanations[0][1], sample_feedback)
        
        stats = adaptive_generator.get_explanation_statistics()
        print(f"  Total explanations: {stats['total_explanations_generated']}")
        print(f"  Overall success rate: {stats['overall_success_rate']:.3f}")
        print(f"  Feedback entries: {stats['feedback_entries']}")
        
        logger.info("Causal explanation demonstration completed successfully")
        return explanations
        
    except Exception as e:
        logger.error(f"Causal explanation failed: {e}")
        print(f"❌ Explanation generation failed: {e}")
        return None


async def demonstrate_external_integrations():
    """Demonstrate integration with external causal libraries."""
    logger = get_logger("tier2_example.external_integrations")
    
    print("\n\n=== External Library Integration Demo ===\n")
    
    try:
        llm_client = get_llm_client()
        
        # Initialize universal integrator
        integrator = UniversalExternalIntegrator(llm_client)
        
        print("🔧 Checking available external libraries...")
        
        # Get integration summary
        summary = integrator.get_integration_summary()
        
        print("📚 LIBRARY AVAILABILITY:")
        print("=" * 40)
        print(f"Total supported libraries: {summary['total_libraries']}")
        print(f"Available libraries: {summary['available_libraries']}\n")
        
        for lib_name, status in summary["library_status"].items():
            print(f"📦 {lib_name.upper()}:")
            print(f"  Status: {'✅ Available' if status['available'] else '❌ Not available'}")
            if status["available"]:
                print(f"  Version: {status['version']}")
                print(f"  Methods: {', '.join(status['methods'][:3])}...")
                print(f"  Strengths: {', '.join(status['strengths'][:2])}...")
            print()
        
        # Generate synthetic data for integration testing
        np.random.seed(42)
        n_samples = 300
        
        # Treatment assignment
        treatment = np.random.binomial(1, 0.4, n_samples)
        
        # Confounders
        age = np.random.normal(50, 15, n_samples)
        income = np.random.normal(50000, 20000, n_samples)
        
        # Outcome influenced by treatment and confounders
        outcome = (
            2.5 * treatment +           # Treatment effect
            0.01 * age +               # Age effect
            0.00001 * income +         # Income effect  
            np.random.normal(0, 1, n_samples)  # Noise
        )
        
        integration_data = pd.DataFrame({
            'treatment': treatment,
            'age': age,
            'income': income,
            'outcome': outcome
        })
        
        variables = {
            'treatment': 'Binary treatment assignment',
            'age': 'Patient age in years',
            'income': 'Annual household income',
            'outcome': 'Primary outcome measure'
        }
        
        print("🧪 Testing Integration with Synthetic Data")
        print(f"Dataset: {len(integration_data)} samples")
        print(f"Variables: {list(variables.keys())}")
        
        # Check which libraries are available
        available_libs = [
            lib for lib, status in summary["library_status"].items() 
            if status["available"]
        ]
        
        if not available_libs:
            print("\n⚠️ No external libraries available for integration testing")
            print("To enable integrations, install:")
            print("  pip install dowhy econml pgmpy")
            return None
        
        print(f"\n🔄 Available for testing: {', '.join(available_libs)}")
        
        # Test integration with first available library
        if "dowhy" in available_libs:
            print("\n🧮 Testing DoWhy Integration...")
            
            try:
                result = await integrator.integrate_with_library(
                    library=ExternalLibrary.DOWHY,
                    method=IntegrationMethod.WRAP_ESTIMATOR,
                    data=integration_data,
                    variables=variables,
                    treatment_variable="treatment",
                    outcome_variable="outcome",
                    confounders=["age", "income"]
                )
                
                print("✅ DoWhy integration successful!")
                print(f"  Causal estimate: {result.results['causal_estimate']:.4f}")
                print(f"  Method: {result.results['estimation_method']}")
                print(f"  Refutation tests: {len(result.results['refutation_tests'])}")
                
                for rec in result.recommendations[:2]:
                    print(f"  - {rec}")
                    
            except Exception as e:
                print(f"❌ DoWhy integration failed: {e}")
        
        # Simulate CausalLLM results for validation
        causalllm_results = {
            "causal_estimate": 2.3,  # Close to true effect of 2.5
            "method": "llm_guided_analysis",
            "confidence": 0.8
        }
        
        print(f"\n🔍 Validating CausalLLM results with external libraries...")
        
        validation_results = await integrator.validate_causalllm_with_external(
            causalllm_results=causalllm_results,
            data=integration_data,
            variables=variables,
            treatment_variable="treatment",
            outcome_variable="outcome",
            confounders=["age", "income"]
        )
        
        if validation_results:
            print(f"✅ Validation completed with {len(validation_results)} libraries")
            
            for result in validation_results:
                print(f"\n📊 {result.library_used.value.upper()} Validation:")
                for insight in result.combined_insights[:2]:
                    print(f"  - {insight}")
                
                for metric, score in result.validation_scores.items():
                    print(f"  {metric}: {score:.3f}")
        else:
            print("⚠️ No validation results available")
        
        logger.info("External integration demonstration completed successfully")
        return summary
        
    except Exception as e:
        logger.error(f"External integration failed: {e}")
        print(f"❌ External integration failed: {e}")
        return None


async def main():
    """Run comprehensive Tier 2 demonstrations."""
    logger = get_logger("tier2_example")
    logger.info("Starting Tier 2 advanced causal analysis demonstrations")
    
    print("🚀 CausalLLM Tier 2 Advanced Causal Analysis Demonstrations")
    print("=" * 70)
    print("This example showcases cutting-edge causal inference capabilities:")
    print("• Advanced causal discovery with LLM guidance")
    print("• Adaptive intervention optimization")
    print("• Temporal causal modeling and forecasting")
    print("• Sophisticated causal explanation generation")
    print("• Integration with external causal libraries")
    print()
    
    # Track demonstration results
    results = {}
    
    # 1. Causal Discovery
    discovery_result = await demonstrate_causal_discovery()
    results['causal_discovery'] = discovery_result is not None
    
    # 2. Intervention Optimization  
    optimization_result = await demonstrate_intervention_optimization()
    results['intervention_optimization'] = optimization_result is not None
    
    # 3. Temporal Analysis
    temporal_result = await demonstrate_temporal_analysis()
    results['temporal_analysis'] = temporal_result is not None
    
    # 4. Causal Explanations
    explanation_result = await demonstrate_causal_explanations()
    results['causal_explanations'] = explanation_result is not None
    
    # 5. External Integrations
    integration_result = await demonstrate_external_integrations()
    results['external_integrations'] = integration_result is not None
    
    # Summary
    print("\n\n🎉 TIER 2 DEMONSTRATIONS SUMMARY")
    print("=" * 50)
    
    successful_demos = sum(results.values())
    total_demos = len(results)
    
    for demo_name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        demo_display = demo_name.replace('_', ' ').title()
        print(f"{demo_display}: {status}")
    
    print(f"\nOverall Success Rate: {successful_demos}/{total_demos} ({100*successful_demos/total_demos:.0f}%)")
    
    if successful_demos == total_demos:
        print("\n🎊 All Tier 2 capabilities demonstrated successfully!")
    elif successful_demos > 0:
        print(f"\n⚠️ Some demonstrations completed successfully.")
    else:
        print("\n❌ Demonstrations encountered issues.")
    
    print("\n📋 NEXT STEPS:")
    print("1. Set up your LLM client (OpenAI API key, etc.)")
    print("2. Install optional dependencies:")
    print("   pip install dowhy econml sentence-transformers")
    print("3. Explore individual Tier 2 modules in your projects")
    print("4. Check the documentation for advanced configuration options")
    
    logger.info(f"Tier 2 demonstrations completed - {successful_demos}/{total_demos} successful")


if __name__ == "__main__":
    asyncio.run(main())