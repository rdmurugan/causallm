#!/usr/bin/env python3
"""
Complete MCP integration demonstration for CausalLLM.

This script shows a complete workflow using MCP for causal reasoning,
including server setup, client connections, and various causal analysis tasks.

Usage:
    python mcp_integration_demo.py
"""

import asyncio
import sys
import json
from pathlib import Path

# Add parent directory to path so we can import causalllm
sys.path.insert(0, str(Path(__file__).parent.parent))

from causalllm.mcp.server import MCPServer
from causalllm.mcp.client import MCPLLMClient
from causalllm.mcp.config import MCPConfig, MCPServerConfig, MCPClientConfig, MCPToolConfig
from causalllm.llm_client import get_llm_client
from causalllm.core import CausalLLMCore
from causalllm.logging import setup_package_logging


class MCPIntegrationDemo:
    """Comprehensive MCP integration demonstration."""
    
    def __init__(self):
        self.server = None
        self.client = None
        
    async def create_test_config(self) -> MCPConfig:
        """Create test configuration for demo."""
        print("📋 Creating test MCP configuration")
        
        config = MCPConfig(
            server=MCPServerConfig(
                name="causalllm-demo-server",
                version="1.0.0",
                description="Demo server for CausalLLM MCP integration",
                transport="websocket",
                host="localhost",
                port=8001  # Use different port for demo
            ),
            client=MCPClientConfig(
                server_name="causalllm-demo-server",
                transport="websocket",
                host="localhost",
                port=8001
            ),
            tools=MCPToolConfig(
                enabled_tools=[
                    "simulate_counterfactual",
                    "generate_do_prompt",
                    "extract_causal_edges",
                    "generate_reasoning_prompt"
                ]
            )
        )
        
        print(f"✅ Configuration created: {config.server.name}")
        return config
        
    async def start_server(self, config: MCPConfig):
        """Start MCP server for demo."""
        print("\n🔌 Starting MCP Server")
        print("-" * 30)
        
        try:
            # Create LLM client for server
            llm_client = get_llm_client("grok")
            print(f"🤖 Server using LLM client: {type(llm_client).__name__}")
            
            # Create and start server
            self.server = MCPServer(config, llm_client)
            await self.server.start()
            
            print(f"✅ MCP server started on {config.server.transport}://{config.server.host}:{config.server.port}")
            
            # Give server time to fully start
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            raise
            
    async def connect_client(self, config: MCPConfig):
        """Connect MCP client."""
        print("\n🔗 Connecting MCP Client")
        print("-" * 30)
        
        try:
            self.client = MCPLLMClient(config)
            await self.client.connect()
            print("✅ MCP client connected successfully")
            
        except Exception as e:
            print(f"❌ Failed to connect client: {e}")
            raise
            
    async def demonstrate_counterfactual_analysis(self):
        """Demonstrate counterfactual analysis via MCP."""
        print("\n🎭 Counterfactual Analysis Demo")
        print("-" * 40)
        
        scenarios = [
            {
                "name": "Education Impact",
                "context": "A student's academic performance analysis",
                "factual": "The student attended 80% of classes and got a GPA of 3.2",
                "intervention": "The student attended 95% of classes"
            },
            {
                "name": "Marketing Campaign",
                "context": "E-commerce marketing effectiveness",
                "factual": "Spent $5000 on social media ads and got 200 conversions",
                "intervention": "Spent $5000 on search engine ads instead"
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{i}. {scenario['name']}:")
            print(f"   Context: {scenario['context']}")
            print(f"   Factual: {scenario['factual']}")
            print(f"   Intervention: {scenario['intervention']}")
            
            try:
                result = await self.client.simulate_counterfactual(
                    context=scenario["context"],
                    factual=scenario["factual"],
                    intervention=scenario["intervention"],
                    temperature=0.7
                )
                
                print(f"   Result: {result[:150]}...")
                print("   ✅ Counterfactual analysis completed")
                
            except Exception as e:
                print(f"   ❌ Counterfactual analysis failed: {e}")
                
    async def demonstrate_causal_discovery(self):
        """Demonstrate causal edge discovery via MCP."""
        print("\n🔗 Causal Discovery Demo")
        print("-" * 40)
        
        scenarios = [
            "Smoking causes lung cancer. Exercise improves cardiovascular health. Stress leads to poor sleep quality.",
            "Education increases income. Higher income enables better healthcare access. Better healthcare improves life expectancy.",
            "Training improves employee skills. Better skills lead to higher productivity. Higher productivity results in company growth."
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{i}. Scenario: {scenario}")
            
            try:
                edges = await self.client.extract_causal_edges(
                    scenario_description=scenario,
                    temperature=0.3
                )
                
                print(f"   Extracted edges: {edges}")
                print("   ✅ Causal discovery completed")
                
            except Exception as e:
                print(f"   ❌ Causal discovery failed: {e}")
                
    async def demonstrate_do_calculus(self):
        """Demonstrate do-calculus prompt generation via MCP."""
        print("\n📐 Do-Calculus Demo")
        print("-" * 30)
        
        scenarios = [
            {
                "context": "Clinical drug trial effectiveness",
                "variables": {
                    "Drug": "Treatment medication (A or B)",
                    "Recovery": "Patient recovery time in days",
                    "Age": "Patient age group"
                },
                "intervention": {"Drug": "A"}
            },
            {
                "context": "Agricultural yield optimization",
                "variables": {
                    "Fertilizer": "Type of fertilizer used",
                    "Yield": "Crop yield per acre",
                    "Weather": "Weather conditions"
                },
                "intervention": {"Fertilizer": "Organic"}
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{i}. Context: {scenario['context']}")
            print(f"   Variables: {list(scenario['variables'].keys())}")
            print(f"   Intervention: {scenario['intervention']}")
            
            try:
                result = await self.client.generate_do_prompt(
                    context=scenario["context"],
                    variables=scenario["variables"],
                    intervention=scenario["intervention"]
                )
                
                print(f"   Result: {result[:200]}...")
                print("   ✅ Do-calculus prompt generated")
                
            except Exception as e:
                print(f"   ❌ Do-calculus generation failed: {e}")
                
    async def demonstrate_core_integration(self):
        """Demonstrate CausalLLMCore integration with MCP."""
        print("\n🧠 Core Integration Demo")
        print("-" * 40)
        
        try:
            # Create MCP-enabled core
            mcp_llm_client = get_llm_client("mcp", "counterfactual")
            
            core = CausalLLMCore(
                context="Supply chain optimization analysis",
                variables={
                    "supplier_reliability": "Supplier delivery consistency",
                    "inventory_cost": "Cost of maintaining inventory",
                    "customer_satisfaction": "Customer satisfaction score"
                },
                dag_edges=[
                    ("supplier_reliability", "inventory_cost"),
                    ("inventory_cost", "customer_satisfaction"),
                    ("supplier_reliability", "customer_satisfaction")
                ],
                llm_client=mcp_llm_client
            )
            
            print(f"✅ CausalLLMCore created with MCP support")
            print(f"   MCP features enabled: {core.is_mcp_client}")
            
            # Test MCP-specific features
            if core.is_mcp_client:
                tools = core.get_mcp_tools()
                print(f"   Available MCP tools: {len(tools)}")
                
                # Create MCP config representation
                mcp_config = core.create_causal_mcp_core()
                print(f"   Core config created: {len(mcp_config)} keys")
            
            # Test regular core operations
            do_result = core.simulate_do(
                {"supplier_reliability": "high"},
                "What is the impact on customer satisfaction?"
            )
            print(f"   Do-operation result: {len(do_result)} characters")
            
            counterfactual_result = core.simulate_counterfactual(
                factual="Current supplier has 85% reliability and costs are high",
                intervention="Switch to supplier with 95% reliability",
                instruction="Analyze impact on costs and customer satisfaction"
            )
            print(f"   Counterfactual result: {len(counterfactual_result)} characters")
            
            print("   ✅ Core integration completed successfully")
            
        except Exception as e:
            print(f"   ❌ Core integration failed: {e}")
            
    async def cleanup(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up resources")
        
        if self.client:
            try:
                await self.client.disconnect()
                print("✅ Client disconnected")
            except:
                pass
                
        if self.server:
            try:
                await self.server.stop()
                print("✅ Server stopped")
            except:
                pass
                
    async def run_demo(self):
        """Run the complete MCP integration demo."""
        print("🚀 CausalLLM MCP Integration Demo")
        print("=" * 50)
        
        try:
            # Create configuration
            config = await self.create_test_config()
            
            # Start server
            await self.start_server(config)
            
            # Connect client
            await self.connect_client(config)
            
            # Run demonstrations
            await self.demonstrate_counterfactual_analysis()
            await self.demonstrate_causal_discovery()
            await self.demonstrate_do_calculus()
            await self.demonstrate_core_integration()
            
            print("\n🎉 MCP Integration Demo Completed Successfully!")
            print("✨ All causal reasoning capabilities are now available via MCP")
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            return False
        finally:
            await self.cleanup()
            
        return True


async def main():
    """Main entry point for MCP integration demo."""
    # Set up logging
    setup_package_logging(level="INFO", log_to_file=False)
    
    demo = MCPIntegrationDemo()
    
    try:
        success = await demo.run_demo()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        await demo.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())