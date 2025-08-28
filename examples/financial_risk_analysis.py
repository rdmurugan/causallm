#!/usr/bin/env python3
"""
Financial Risk Analysis Example

This example demonstrates how to use CausalLLM to analyze financial risk
and explore the causal factors behind loan portfolio performance.

Run: python examples/financial_risk_analysis.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causallm.core.causal_llm_core import CausalLLMCore


def main():
    """Run financial risk causal analysis example."""
    print("🏦 " + "="*60)
    print("   CAUSALLM FINANCIAL RISK ANALYSIS EXAMPLE")
    print("="*63)
    print()

    # Define banking context
    finance_context = """
    A retail bank manages a loan portfolio across economic cycles.
    Interest rates influence both loan demand and borrower default rates.
    Macroeconomic conditions such as unemployment and GDP growth affect
    credit risk and profitability.
    The bank seeks to understand how policy changes impact default risk
    and overall portfolio performance.
    """

    # Define current financial state
    finance_variables = {
        "interest_rate": "3% baseline lending rate",
        "unemployment_rate": "5% unemployment",
        "gdp_growth": "2% annual GDP growth",
        "loan_demand": "stable loan applications",
        "default_rate": "1.5% default rate",
        "average_credit_score": "700 average credit score",
        "portfolio_profit": "$5M quarterly profit"
    }

    # Define financial causal relationships
    finance_dag = [
        ("interest_rate", "loan_demand"),
        ("interest_rate", "default_rate"),
        ("unemployment_rate", "default_rate"),
        ("gdp_growth", "unemployment_rate"),
        ("loan_demand", "portfolio_profit"),
        ("default_rate", "portfolio_profit"),
        ("average_credit_score", "default_rate")
    ]

    print("📉 Setting up financial risk model...")
    try:
        core = CausalLLMCore(finance_context, finance_variables, finance_dag)
        print("   ✅ Financial causal model initialized")
        print(f"   ✅ DAG created with {len(finance_dag)} relationships")
        print()

        # Scenario 1: Rising interest rates
        print("📈 SCENARIO 1: Interest Rate Increase")
        print("-" * 45)
        rate_hike = core.simulate_do({
            "interest_rate": "6% lending rate due to policy tightening"
        })
        print(rate_hike)
        print()

        # Scenario 2: Economic downturn
        print("📉 SCENARIO 2: Economic Downturn")
        print("-" * 42)
        downturn = core.simulate_do({
            "unemployment_rate": "9% unemployment in recession",
            "gdp_growth": "-1% GDP contraction"
        })
        print(downturn)
        print()

        # Scenario 3: Stricter credit policy
        print("🛡️ SCENARIO 3: Stricter Credit Policy")
        print("-" * 44)
        stricter_credit = core.simulate_do({
            "average_credit_score": "730 minimum credit score requirement"
        })
        print(stricter_credit)
        print()

        # Generate risk management prompt
        print("🔍 RISK MANAGEMENT ANALYSIS")
        print("-" * 40)
        risk_task = (
            "recommend strategies to minimize default risk while maintaining profitability"
        )
        risk_prompt = core.generate_reasoning_prompt(risk_task)
        print("Generated financial risk prompt:")
        print(risk_prompt)
        print()

    except Exception as e:
        print(f"   ❌ Error in financial analysis: {e}")
        return

    # Financial insights
    print("💡 FINANCIAL INSIGHTS")
    print("-" * 25)
    print("💰 Interest Rates:")
    print("   • Higher rates may reduce demand but raise default risk")
    print("   • Balance pricing with borrower affordability")
    print()
    print("📊 Economic Conditions:")
    print("   • Rising unemployment increases default likelihood")
    print("   • GDP growth stabilizes credit performance")
    print()
    print("🛡️ Credit Policy:")
    print("   • Stricter credit improves portfolio quality")
    print("   • May limit loan growth if too restrictive")
    print()

    print("🎯 KEY TAKEAWAYS")
    print("-" * 20)
    print("✅ CausalLLM clarified financial risk relationships")
    print("✅ Simulated macroeconomic and policy scenarios")
    print("✅ Generated structured reasoning for risk management")
    print()
    print("📚 For more examples, see: USAGE_EXAMPLES.md")


if __name__ == "__main__":
    main()
