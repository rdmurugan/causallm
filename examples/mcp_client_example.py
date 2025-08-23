#!/usr/bin/env python3
"""
Example MCP client usage for CausalLLM.

This script demonstrates how to use the CausalLLM MCP client to connect
to MCP servers and perform causal reasoning tasks.

Usage:
    python mcp_client_example.py [config_path]
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path so we can import causalllm
sys.path.insert(0, str(Path(__file__).parent.parent))

from causalllm.mcp.client import MCPClient, MCPLLMClient
from causalllm.mcp.config import load_mcp_config
from causalllm.llm_client import get_llm_client
from causalllm.core import CausalLLMCore
from causalllm.logging import setup_package_logging


async def demonstrate_raw_mcp_client():
    """Demonstrate raw MCP client usage."""
    print("\n🔧 Raw MCP Client Example")
    print("=" * 40)
    
    try:
        # Create and connect MCP client
        client = MCPClient()
        await client.connect()
        
        print("✅ Connected to MCP server")
        
        # List available tools
        print("\n📋 Available Tools:")
        tools = await client.list_tools()
        for tool in tools:
            print(f"   - {tool['name']}: {tool['description']}")
        
        # Example: Simulate counterfactual
        print("\n🎭 Counterfactual Simulation Example:")
        result = await client.call_tool("simulate_counterfactual", {
            "context": "A student is preparing for an exam",
            "factual": "The student studied for 2 hours and got a B grade",
            "intervention": "The student studied for 6 hours instead"
        })
        
        print("Result:")
        content = result.get("content", [])
        if content and content[0].get("type") == "text":
            print(f"   {content[0]['text']}")
        else:
            print(f"   {result}")
        
        # Example: Extract causal edges
        print("\n🔗 Causal Edge Extraction Example:")
        result = await client.call_tool("extract_causal_edges", {
            "scenario_description": "Smoking causes lung cancer. Exercise improves health. Stress leads to poor sleep."
        })
        
        print("Result:")
        content = result.get("content", [])
        if content and content[0].get("type") == "text":
            print(f"   {content[0]['text']}")
        
        await client.disconnect()
        print("✅ Disconnected from MCP server")
        
    except Exception as e:
        print(f"❌ Raw MCP client error: {e}")


async def demonstrate_llm_mcp_client():
    """Demonstrate LLM-integrated MCP client."""
    print("\n🤖 LLM MCP Client Example")
    print("=" * 40)
    
    try:
        # Create LLM-integrated MCP client
        llm_client = MCPLLMClient()
        await llm_client.connect()
        
        print("✅ Connected to MCP server via LLM client")
        
        # Example: Counterfactual simulation
        print("\n🎭 Counterfactual via LLM Client:")
        result = await llm_client.simulate_counterfactual(
            context="A company is deciding on marketing strategy",
            factual="They spent $10k on social media ads and got 100 customers",
            intervention="They spent $10k on TV ads instead",
            temperature=0.7
        )
        print(f"Result: {result}")
        
        # Example: Do-calculus prompt generation
        print("\n📐 Do-Calculus Prompt Generation:")
        result = await llm_client.generate_do_prompt(
            context="Medical treatment effectiveness study",
            variables={
                "Treatment": "Medication A vs Medication B",
                "Outcome": "Patient recovery time",
                "Confounder": "Patient age"
            },
            intervention={"Treatment": "Medication A"}
        )
        print(f"Result: {result}")
        
        # Example: Extract causal edges
        print("\n🔗 Causal Edge Extraction:")
        edges = await llm_client.extract_causal_edges(
            scenario_description="Education leads to higher income. Higher income enables better healthcare. Better healthcare improves life expectancy."
        )
        print(f"Extracted edges: {edges}")
        
        await llm_client.disconnect()
        print("✅ Disconnected from MCP server")
        
    except Exception as e:
        print(f"❌ LLM MCP client error: {e}")


def demonstrate_core_with_mcp():
    """Demonstrate CausalLLMCore with MCP client."""
    print("\n🧠 CausalLLMCore with MCP Example")
    print("=" * 40)
    
    try:
        # Create MCP client for core
        mcp_client = get_llm_client("mcp", "counterfactual")
        print(f"✅ Created MCP LLM client: {type(mcp_client).__name__}")
        
        # Create CausalLLMCore with MCP client
        core = CausalLLMCore(
            context="Healthcare system analysis",
            variables={
                "treatment": "Medical intervention",
                "outcome": "Patient health",
                "cost": "Healthcare cost"
            },
            dag_edges=[("treatment", "outcome"), ("treatment", "cost")],
            llm_client=mcp_client
        )
        
        print(f"✅ Created CausalLLMCore with MCP integration")
        print(f"   MCP features enabled: {core.is_mcp_client}")
        
        # Demonstrate MCP-specific methods
        if core.is_mcp_client:
            print("\n🔧 Available MCP Tools:")
            tools = core.get_mcp_tools()
            for tool in tools:
                print(f"   - {tool}")
            
            # Create MCP core representation
            mcp_config = core.create_causal_mcp_core()
            print(f"\n📋 MCP Core Config:")
            print(f"   Variables: {len(mcp_config['variables'])}")
            print(f"   DAG edges: {len(mcp_config['dag_edges'])}")
            print(f"   Capabilities: {list(mcp_config['capabilities'].keys())}")
        
        # Regular core operations work with MCP client
        print("\n🎭 Counterfactual Simulation:")
        result = core.simulate_counterfactual(
            factual="Patient received standard treatment and recovered in 10 days",
            intervention="Patient received experimental treatment",
            instruction="Compare recovery times and side effects"
        )
        print(f"Result: {result[:200]}...")
        
        print("✅ CausalLLMCore with MCP completed successfully")
        
    except Exception as e:
        print(f"❌ Core with MCP error: {e}")


async def main():
    """Main entry point for MCP client examples."""
    print("🚀 CausalLLM MCP Client Examples")
    print("=" * 50)
    
    # Set up logging
    setup_package_logging(level="INFO", log_to_file=False)
    
    print("💡 This example assumes an MCP server is running")
    print("   You can start one with: python mcp_server_example.py")
    
    try:
        # Demonstrate different MCP client approaches
        await demonstrate_raw_mcp_client()
        await demonstrate_llm_mcp_client()
        demonstrate_core_with_mcp()
        
        print("\n🎉 All MCP client examples completed!")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Example error: {e}")


if __name__ == "__main__":
    asyncio.run(main())