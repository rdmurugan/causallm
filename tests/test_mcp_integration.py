#!/usr/bin/env python3
"""
Comprehensive test suite for CausalLLM MCP integration.

This script tests all MCP components including configuration, transport,
tools, server, client, and integration with the core system.

Usage:
    python test_mcp_integration.py
"""

import asyncio
import sys
import unittest
import tempfile
import json
from pathlib import Path

# Add parent directory to path so we can import causalllm
sys.path.insert(0, str(Path(__file__).parent.parent))

from causalllm.mcp.config import MCPConfig, MCPServerConfig, MCPClientConfig, MCPToolConfig, load_mcp_config
from causalllm.mcp.transport import create_transport, StdioTransport, WebSocketTransport
from causalllm.mcp.tools import CausalTools
from causalllm.mcp.server import MCPServer
from causalllm.mcp.client import MCPClient, MCPLLMClient
from causalllm.llm_client import get_llm_client
from causalllm.core import CausalLLMCore
from causalllm.logging import setup_package_logging


class TestMCPConfiguration(unittest.TestCase):
    """Test MCP configuration management."""
    
    def test_default_config_creation(self):
        """Test creating default MCP configuration."""
        config = MCPConfig()
        
        self.assertIsNotNone(config.server)
        self.assertIsNotNone(config.client)
        self.assertIsNotNone(config.tools)
        
        self.assertEqual(config.server.name, "causalllm-server")
        self.assertEqual(config.server.transport, "stdio")
        self.assertEqual(config.client.server_name, "causalllm-server")
        self.assertGreater(len(config.tools.enabled_tools), 0)
        
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should pass
        config = MCPConfig()
        try:
            config.validate()
        except Exception as e:
            self.fail(f"Valid config failed validation: {e}")
        
        # Invalid transport should fail
        config.server.transport = "invalid"
        with self.assertRaises(ValueError):
            config.validate()
            
    def test_config_file_operations(self):
        """Test saving and loading configuration files."""
        config = MCPConfig()
        
        # Test saving to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.to_file(f.name)
            
            # Test loading from file
            loaded_config = MCPConfig.from_file(f.name)
            
            self.assertEqual(config.server.name, loaded_config.server.name)
            self.assertEqual(config.client.server_name, loaded_config.client.server_name)
            self.assertEqual(config.tools.enabled_tools, loaded_config.tools.enabled_tools)
            
            Path(f.name).unlink()  # Clean up


class TestMCPTransport(unittest.TestCase):
    """Test MCP transport implementations."""
    
    def test_transport_factory(self):
        """Test transport factory function."""
        # Test stdio transport creation
        stdio_transport = create_transport("stdio")
        self.assertIsInstance(stdio_transport, StdioTransport)
        
        # Test websocket transport creation
        ws_transport = create_transport("websocket", host="localhost", port=8000)
        self.assertIsInstance(ws_transport, WebSocketTransport)
        self.assertEqual(ws_transport.host, "localhost")
        self.assertEqual(ws_transport.port, 8000)
        
        # Test invalid transport
        with self.assertRaises(ValueError):
            create_transport("invalid")


class TestMCPTools(unittest.TestCase):
    """Test MCP tool definitions and execution."""
    
    def setUp(self):
        """Set up test environment."""
        # Use grok (mock) client for testing
        self.llm_client = get_llm_client("grok")
        self.tools = CausalTools(self.llm_client)
        
    def test_tools_initialization(self):
        """Test tools initialization."""
        self.assertIsNotNone(self.tools.llm_client)
        self.assertGreater(len(self.tools.tools), 0)
        
        expected_tools = {
            "simulate_counterfactual",
            "generate_do_prompt", 
            "extract_causal_edges",
            "generate_reasoning_prompt",
            "analyze_treatment_effect",
            "create_causal_core"
        }
        
        actual_tools = set(self.tools.tools.keys())
        self.assertTrue(expected_tools.issubset(actual_tools))
        
    def test_get_tool_list(self):
        """Test getting tool list in MCP format."""
        tool_list = self.tools.get_tool_list()
        
        self.assertIsInstance(tool_list, list)
        self.assertGreater(len(tool_list), 0)
        
        for tool in tool_list:
            self.assertIn("name", tool)
            self.assertIn("description", tool)
            self.assertIn("inputSchema", tool)
            
    def test_tool_schemas(self):
        """Test tool input schemas."""
        for tool_name, tool in self.tools.tools.items():
            schema = tool.input_schema
            
            self.assertIn("type", schema)
            self.assertEqual(schema["type"], "object")
            self.assertIn("properties", schema)
            self.assertIn("required", schema)


class TestMCPIntegration(unittest.IsolatedAsyncioTestCase):
    """Test MCP server and client integration."""
    
    async def asyncSetUp(self):
        """Set up async test environment."""
        self.config = MCPConfig(
            server=MCPServerConfig(
                transport="websocket",
                host="localhost", 
                port=8002  # Use unique port for tests
            ),
            client=MCPClientConfig(
                server_name="test-server",
                transport="websocket",
                host="localhost",
                port=8002
            )
        )
        
        self.llm_client = get_llm_client("grok")
        self.server = None
        self.client = None
        
    async def asyncTearDown(self):
        """Clean up async test environment."""
        if self.client:
            try:
                await self.client.disconnect()
            except:
                pass
                
        if self.server:
            try:
                await self.server.stop()
            except:
                pass
                
    async def test_server_startup_shutdown(self):
        """Test MCP server startup and shutdown."""
        server = MCPServer(self.config, self.llm_client)
        
        # Test startup
        await server.start()
        self.assertTrue(server._running)
        
        # Test shutdown
        await server.stop()
        self.assertFalse(server._running)
        
    async def test_client_connection(self):
        """Test MCP client connection."""
        # Start server
        self.server = MCPServer(self.config, self.llm_client)
        await self.server.start()
        
        # Give server time to start
        await asyncio.sleep(0.5)
        
        # Test client connection
        self.client = MCPClient(self.config)
        await self.client.connect()
        
        self.assertTrue(self.client.is_connected)
        
        # Test basic ping
        ping_result = await self.client.ping()
        self.assertTrue(ping_result)
        
    async def test_tool_listing_and_calls(self):
        """Test listing and calling MCP tools."""
        # Start server
        self.server = MCPServer(self.config, self.llm_client)
        await self.server.start()
        await asyncio.sleep(0.5)
        
        # Connect client
        self.client = MCPClient(self.config)
        await self.client.connect()
        
        # Test tool listing
        tools = await self.client.list_tools()
        self.assertIsInstance(tools, list)
        self.assertGreater(len(tools), 0)
        
        tool_names = [tool['name'] for tool in tools]
        self.assertIn("simulate_counterfactual", tool_names)
        
        # Test tool call
        result = await self.client.call_tool("simulate_counterfactual", {
            "context": "Test context",
            "factual": "Test factual scenario",
            "intervention": "Test intervention"
        })
        
        self.assertIn("content", result)
        self.assertIsInstance(result["content"], list)


class TestCoreIntegration(unittest.TestCase):
    """Test CausalLLMCore integration with MCP."""
    
    def test_mcp_client_detection(self):
        """Test MCP client detection in core."""
        # Test with regular client
        regular_client = get_llm_client("grok")
        core_regular = CausalLLMCore(
            context="Test context",
            variables={"X": "Variable X", "Y": "Variable Y"},
            dag_edges=[("X", "Y")],
            llm_client=regular_client
        )
        
        self.assertFalse(core_regular.is_mcp_client)
        
        # Test with MCP client
        mcp_client = get_llm_client("mcp")
        core_mcp = CausalLLMCore(
            context="Test context",
            variables={"X": "Variable X", "Y": "Variable Y"},
            dag_edges=[("X", "Y")],
            llm_client=mcp_client
        )
        
        self.assertTrue(core_mcp.is_mcp_client)
        
    def test_mcp_core_config_creation(self):
        """Test creating MCP core configuration."""
        mcp_client = get_llm_client("mcp")
        core = CausalLLMCore(
            context="Test context for MCP",
            variables={"treatment": "Medical treatment", "outcome": "Health outcome"},
            dag_edges=[("treatment", "outcome")],
            llm_client=mcp_client
        )
        
        mcp_config = core.create_causal_mcp_core()
        
        self.assertIn("context", mcp_config)
        self.assertIn("variables", mcp_config)
        self.assertIn("dag_edges", mcp_config)
        self.assertIn("capabilities", mcp_config)
        
        self.assertEqual(mcp_config["context"], "Test context for MCP")
        self.assertEqual(len(mcp_config["variables"]), 2)
        self.assertEqual(len(mcp_config["dag_edges"]), 1)
        
    def test_from_mcp_config(self):
        """Test creating core from MCP configuration."""
        mcp_config = {
            "context": "Test MCP context",
            "variables": {
                "cause": "Causal variable",
                "effect": "Effect variable"
            },
            "dag_edges": [["cause", "effect"]]
        }
        
        core = CausalLLMCore.from_mcp_config(mcp_config)
        
        self.assertEqual(core.context, "Test MCP context")
        self.assertEqual(len(core.variables), 2)
        self.assertEqual(len(core.dag.edges), 1)


class TestEndToEndMCP(unittest.IsolatedAsyncioTestCase):
    """End-to-end MCP integration tests."""
    
    async def test_complete_mcp_workflow(self):
        """Test complete MCP workflow from server to client."""
        # Create configuration
        config = MCPConfig(
            server=MCPServerConfig(transport="websocket", host="localhost", port=8003),
            client=MCPClientConfig(transport="websocket", host="localhost", port=8003),
            tools=MCPToolConfig(enabled_tools=["simulate_counterfactual"])
        )
        
        # Start server
        llm_client = get_llm_client("grok")
        server = MCPServer(config, llm_client)
        
        try:
            await server.start()
            await asyncio.sleep(0.5)
            
            # Connect LLM client
            llm_mcp_client = MCPLLMClient(config)
            await llm_mcp_client.connect()
            
            # Test counterfactual simulation
            result = await llm_mcp_client.simulate_counterfactual(
                context="Business decision analysis",
                factual="Company launched product A and got 1000 sales",
                intervention="Company launched product B instead",
                temperature=0.5
            )
            
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            
            await llm_mcp_client.disconnect()
            
        finally:
            await server.stop()


def run_tests():
    """Run all MCP integration tests."""
    print("🧪 Running CausalLLM MCP Integration Tests")
    print("=" * 50)
    
    # Set up logging
    setup_package_logging(level="WARNING", log_to_file=False)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMCPConfiguration,
        TestMCPTransport,
        TestMCPTools,
        TestMCPIntegration,
        TestCoreIntegration,
        TestEndToEndMCP
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"🧪 Test Results:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.splitlines()[-1]}")
            
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.splitlines()[-1]}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n✅ All MCP integration tests passed!")
    else:
        print(f"\n❌ {len(result.failures + result.errors)} tests failed")
        
    return success


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)