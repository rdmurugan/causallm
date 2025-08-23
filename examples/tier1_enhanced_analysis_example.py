"""
Example demonstrating Tier 1 enhanced causal analysis capabilities.

This example showcases how to use the new Tier 1 LLM enhancements including:
- Intelligent prompt engineering with few-shot learning
- Multi-agent collaborative causal analysis
- Dynamic RAG for causal knowledge retrieval
- Comprehensive enhanced analysis workflows
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalllm.core import CausalLLMCore
from causalllm.llm_client import get_llm_client
from causalllm.logging import get_logger


async def demonstrate_enhanced_counterfactual_analysis():
    """Demonstrate enhanced counterfactual analysis with Tier 1 features."""
    logger = get_logger("tier1_example.counterfactual")
    
    print("=== Enhanced Counterfactual Analysis Demo ===\n")
    
    # Define a healthcare scenario
    context = """
    A clinical study examining the effectiveness of a new blood pressure medication.
    The study involves 1000 patients with hypertension, randomly assigned to treatment
    and control groups. Key variables include patient demographics, baseline health,
    treatment assignment, and blood pressure outcomes after 6 months.
    """
    
    variables = {
        "age": "Patient age in years",
        "baseline_bp": "Baseline systolic blood pressure",
        "treatment": "Assignment to new medication (1) or placebo (0)", 
        "outcome_bp": "Blood pressure after 6 months of treatment",
        "comorbidities": "Number of other health conditions"
    }
    
    dag_edges = [
        ("age", "baseline_bp"),
        ("age", "comorbidities"),
        ("baseline_bp", "treatment"),
        ("comorbidities", "treatment"),
        ("treatment", "outcome_bp"),
        ("baseline_bp", "outcome_bp"),
        ("age", "outcome_bp")
    ]
    
    # Initialize core with LLM client
    try:
        llm_client = get_llm_client()
        core = CausalLLMCore(
            context=context,
            variables=variables, 
            dag_edges=dag_edges,
            llm_client=llm_client
        )
        
        print("✅ CausalLLMCore initialized with Tier 1 enhancements")
        
    except Exception as e:
        print(f"❌ Failed to initialize core: {e}")
        print("Note: This example requires a properly configured LLM client.")
        print("Please set up environment variables for OpenAI, Anthropic, or other supported providers.")
        return
    
    # Define counterfactual scenario
    factual = "A 65-year-old patient with high baseline blood pressure received the new medication"
    intervention = "The same patient received placebo instead of the new medication"
    
    print(f"Factual scenario: {factual}")
    print(f"Counterfactual: {intervention}\n")
    
    try:
        # Run enhanced counterfactual analysis
        print("🔄 Running enhanced counterfactual analysis...")
        result = await core.enhanced_counterfactual_analysis(
            factual=factual,
            intervention=intervention,
            domain="healthcare"
        )
        
        print("✅ Analysis completed!\n")
        
        # Display results
        print("📊 ANALYSIS RESULTS")
        print("=" * 50)
        
        print(f"Overall Confidence: {result['overall_confidence']:.2f}")
        print(f"Domain: {result['domain']}")
        
        print("\n🔍 RAG Analysis:")
        rag = result["rag_analysis"]
        print(f"  • Documents Retrieved: {rag['retrieved_documents']}")
        print(f"  • Confidence Score: {rag['confidence_score']:.2f}")
        print(f"  • Knowledge Gaps: {len(rag['knowledge_gaps'])}")
        
        print("\n🎯 Intelligent Prompting:")
        prompting = result["intelligent_prompting"] 
        print(f"  • Reasoning Strategy: {prompting['reasoning_strategy']}")
        print(f"  • Examples Used: {prompting['examples_used']}")
        print(f"  • Quality Score: {prompting['quality_score']:.2f}")
        
        print("\n👥 Multi-Agent Analysis:")
        agents = result["multi_agent_analysis"]
        print(f"  • Confidence Score: {agents['confidence_score']:.2f}")
        print(f"  • Key Assumptions: {len(agents['key_assumptions'])}")
        print(f"  • Recommendations: {len(agents['recommendations'])}")
        
        print(f"\n📝 Synthesized Conclusion:")
        print(f"  {agents['synthesized_conclusion'][:200]}...")
        
        print("\n🎯 Key Recommendations:")
        for i, rec in enumerate(agents['recommendations'][:3], 1):
            print(f"  {i}. {rec}")
            
        logger.info("Enhanced counterfactual analysis demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {e}")
        print(f"❌ Analysis failed: {e}")


async def demonstrate_enhanced_treatment_effect_analysis():
    """Demonstrate enhanced treatment effect analysis."""
    logger = get_logger("tier1_example.treatment_effect")
    
    print("\n\n=== Enhanced Treatment Effect Analysis Demo ===\n")
    
    # Define a marketing scenario
    context = """
    An e-commerce company testing the impact of personalized email recommendations
    on customer purchase behavior. The experiment involves 10,000 customers randomly
    assigned to receive either personalized recommendations or generic promotional emails.
    """
    
    variables = {
        "customer_segment": "Customer value tier (high/medium/low)",
        "purchase_history": "Number of purchases in past 6 months", 
        "email_type": "Personalized recommendations (1) vs generic (0)",
        "email_opens": "Number of emails opened during campaign",
        "purchase_amount": "Total purchase amount during campaign period"
    }
    
    dag_edges = [
        ("customer_segment", "purchase_history"),
        ("customer_segment", "email_type"),
        ("purchase_history", "email_type"),
        ("email_type", "email_opens"),
        ("email_type", "purchase_amount"),
        ("email_opens", "purchase_amount"),
        ("customer_segment", "purchase_amount")
    ]
    
    try:
        llm_client = get_llm_client()
        core = CausalLLMCore(
            context=context,
            variables=variables,
            dag_edges=dag_edges,
            llm_client=llm_client
        )
        
        print("✅ Marketing scenario core initialized")
        
    except Exception as e:
        print(f"❌ Failed to initialize core: {e}")
        return
    
    # Define treatment effect analysis
    treatment = "Personalized email recommendations based on browsing history"
    outcome = "Customer purchase amount during 30-day campaign period"
    
    print(f"Treatment: {treatment}")
    print(f"Outcome: {outcome}\n")
    
    try:
        print("🔄 Running enhanced treatment effect analysis...")
        result = await core.enhanced_treatment_effect_analysis(
            treatment=treatment,
            outcome=outcome,
            domain="marketing"
        )
        
        print("✅ Treatment effect analysis completed!\n")
        
        # Display results
        print("📊 TREATMENT EFFECT RESULTS")
        print("=" * 50)
        
        assessment = result["overall_assessment"]
        print(f"Overall Confidence: {assessment['confidence']:.2f}")
        
        print("\n🔍 RAG Enhancement:")
        rag = result["rag_enhancement"]
        print(f"  • Confidence Score: {rag['confidence_score']:.2f}")
        print(f"  • Knowledge Gaps: {len(rag['knowledge_gaps'])}")
        
        print("\n👥 Collaborative Analysis:")
        collab = result["collaborative_analysis"]
        print(f"  • Confidence: {collab['confidence_score']:.2f}")
        print(f"  • Key Assumptions: {len(collab['key_assumptions'])}")
        
        print(f"\n📝 Key Findings:")
        print(f"  {assessment['key_findings'][:200]}...")
        
        print("\n🎯 Methodological Recommendations:")
        for i, rec in enumerate(assessment['methodological_recommendations'][:3], 1):
            print(f"  {i}. {rec}")
            
        logger.info("Enhanced treatment effect analysis demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Treatment effect analysis failed: {e}")
        print(f"❌ Analysis failed: {e}")


async def demonstrate_intelligent_prompting():
    """Demonstrate standalone intelligent prompting capabilities."""
    print("\n\n=== Intelligent Prompting Demo ===\n")
    
    try:
        from causalllm.llm_prompting import IntelligentCausalPrompting
        
        llm_client = get_llm_client()
        prompting_system = IntelligentCausalPrompting(llm_client)
        
        print("✅ Intelligent prompting system initialized")
        
        # Demonstrate enhanced prompt generation
        context = "A social media platform testing the impact of algorithm changes on user engagement"
        factual = "Users see chronological timeline"
        intervention = "Users see algorithm-curated timeline"
        domain = "technology"
        
        print(f"Context: {context}")
        print(f"Factual: {factual}")
        print(f"Intervention: {intervention}\n")
        
        print("🔄 Generating enhanced prompt with few-shot learning...")
        
        enhanced_prompt = await prompting_system.generate_enhanced_counterfactual_prompt(
            context=context,
            factual=factual,
            intervention=intervention,
            domain=domain
        )
        
        print("✅ Enhanced prompt generated!\n")
        
        print("📊 PROMPTING RESULTS")
        print("=" * 50)
        print(f"Reasoning Strategy: {enhanced_prompt.reasoning_strategy}")
        print(f"Examples Used: {len(enhanced_prompt.examples_used)}")
        print(f"Quality Score: {enhanced_prompt.quality_score:.2f}")
        print(f"Performance Metrics: {enhanced_prompt.performance_metrics}")
        
        print(f"\n📝 Enhanced Prompt (first 300 chars):")
        print(f"  {enhanced_prompt.enhanced_prompt[:300]}...")
        
    except Exception as e:
        print(f"❌ Intelligent prompting demo failed: {e}")


async def demonstrate_multi_agent_analysis():
    """Demonstrate standalone multi-agent analysis."""
    print("\n\n=== Multi-Agent Analysis Demo ===\n")
    
    try:
        from causalllm.llm_agents import MultiAgentCausalAnalyzer
        
        llm_client = get_llm_client()
        analyzer = MultiAgentCausalAnalyzer(llm_client)
        
        print("✅ Multi-agent analyzer initialized")
        
        context = "Educational intervention study measuring impact of tutoring on student performance"
        factual = "Students receive standard classroom instruction"
        intervention = "Students receive additional one-on-one tutoring"
        
        print(f"Context: {context}")
        print(f"Factual: {factual}")
        print(f"Intervention: {intervention}\n")
        
        print("🔄 Running multi-agent collaborative analysis...")
        
        result = await analyzer.analyze_counterfactual(
            context=context,
            factual=factual,
            intervention=intervention
        )
        
        print("✅ Multi-agent analysis completed!\n")
        
        print("📊 MULTI-AGENT RESULTS")
        print("=" * 50)
        print(f"Overall Confidence: {result.confidence_score:.2f}")
        print(f"Key Assumptions: {len(result.key_assumptions)}")
        print(f"Recommendations: {len(result.recommendations)}")
        
        print(f"\n🎓 Domain Expert Analysis:")
        print(f"  {result.domain_expert_analysis[:150]}...")
        
        print(f"\n📊 Statistical Analysis:")
        print(f"  {result.statistical_analysis[:150]}...")
        
        print(f"\n🤔 Skeptic Analysis:")  
        print(f"  {result.skeptic_analysis[:150]}...")
        
        print(f"\n🎯 Synthesized Conclusion:")
        print(f"  {result.synthesized_conclusion[:200]}...")
        
    except Exception as e:
        print(f"❌ Multi-agent analysis demo failed: {e}")


async def main():
    """Run all Tier 1 enhancement demonstrations."""
    logger = get_logger("tier1_example")
    logger.info("Starting Tier 1 enhanced analysis demonstrations")
    
    print("🚀 CausalLLM Tier 1 Enhanced Analysis Demonstrations")
    print("=" * 60)
    print("This example showcases advanced LLM capabilities for causal inference:")
    print("• Intelligent prompt engineering with few-shot learning")
    print("• Multi-agent collaborative analysis")  
    print("• Dynamic RAG for causal knowledge retrieval")
    print("• Comprehensive enhanced analysis workflows\n")
    
    # Run demonstrations
    await demonstrate_enhanced_counterfactual_analysis()
    await demonstrate_enhanced_treatment_effect_analysis()
    await demonstrate_intelligent_prompting()
    await demonstrate_multi_agent_analysis()
    
    print("\n\n🎉 All demonstrations completed!")
    print("\nTo run this example:")
    print("1. Set up your LLM client (OpenAI API key, etc.)")
    print("2. Install dependencies: pip install sentence-transformers (optional)")
    print("3. Run: python examples/tier1_enhanced_analysis_example.py")
    
    logger.info("Tier 1 demonstrations completed successfully")


if __name__ == "__main__":
    asyncio.run(main())