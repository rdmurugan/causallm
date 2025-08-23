#!/usr/bin/env python3
"""
Example MCP server implementation for CausalLLM.

This script demonstrates how to run a CausalLLM MCP server that exposes
causal reasoning capabilities via the Model Context Protocol.

Usage:
    python mcp_server_example.py [config_path]
"""

import asyncio
import sys
import signal
from pathlib import Path

# Add parent directory to path so we can import causalllm
sys.path.insert(0, str(Path(__file__).parent.parent))

from causalllm.mcp.server import MCPServer
from causalllm.mcp.config import load_mcp_config
from causalllm.llm_client import get_llm_client
from causalllm.logging import setup_package_logging


async def main():
    """Main entry point for MCP server example."""
    print("🚀 Starting CausalLLM MCP Server Example")
    
    # Set up logging
    setup_package_logging(level="INFO", log_to_file=False)
    
    # Load configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    config = load_mcp_config(config_path)
    
    print(f"📋 Server Configuration:")
    print(f"   Name: {config.server.name}")
    print(f"   Version: {config.server.version}")
    print(f"   Transport: {config.server.transport}")
    if config.server.transport == "websocket":
        print(f"   Address: {config.server.host}:{config.server.port}")
    print(f"   Enabled Tools: {', '.join(config.tools.enabled_tools)}")
    
    try:
        # Create LLM client (defaults to grok for demo)
        llm_client = get_llm_client("grok")
        print(f"🤖 Using LLM client: {type(llm_client).__name__}")
        
        # Create and start MCP server
        server = MCPServer(config, llm_client)
        
        print("🔌 Starting MCP server...")
        await server.start()
        
        print("✅ MCP server started successfully!")
        print("📡 Server ready to accept MCP connections")
        
        if config.server.transport == "stdio":
            print("📝 Using stdio transport - server will handle stdin/stdout")
        elif config.server.transport == "websocket":
            print(f"🌐 WebSocket server listening on ws://{config.server.host}:{config.server.port}")
        
        print("\n💡 You can now connect MCP clients to this server")
        print("🔧 Available tools:")
        for tool in config.tools.enabled_tools:
            print(f"   - {tool}")
        
        # Set up graceful shutdown
        def signal_handler():
            print("\n🛑 Shutdown signal received")
            asyncio.create_task(server.stop())
        
        # Register signal handlers
        if sys.platform != "win32":
            loop = asyncio.get_event_loop()
            for sig in [signal.SIGTERM, signal.SIGINT]:
                loop.add_signal_handler(sig, signal_handler)
        
        # Run server forever
        await server.run_forever()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)
    finally:
        print("🏁 MCP server shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())