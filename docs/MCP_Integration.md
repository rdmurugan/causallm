# Model Context Protocol (MCP) Integration

CausalLLM now supports the Model Context Protocol (MCP), enabling seamless integration with MCP-compatible tools and applications. This document provides comprehensive information about using MCP with CausalLLM.

## Overview

The Model Context Protocol (MCP) is a standardized way for applications to expose capabilities to language models and other AI systems. CausalLLM's MCP integration allows you to:

- **Expose causal reasoning capabilities** as MCP tools
- **Run MCP servers** that provide causal analysis services
- **Connect to MCP servers** from CausalLLM applications
- **Integrate seamlessly** with MCP-compatible environments

## Architecture

CausalLLM's MCP implementation consists of several key components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MCP Client    │◄──►│   Transport     │◄──►│   MCP Server    │
│                 │    │   (stdio/ws)    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                                              │
         ▼                                              ▼
┌─────────────────┐                          ┌─────────────────┐
│ CausalLLMCore   │                          │  Causal Tools   │
│   Integration   │                          │   Collection    │
└─────────────────┘                          └─────────────────┘
```

### Core Components

1. **MCP Server** (`causalllm.mcp.server`): Exposes CausalLLM capabilities as MCP tools
2. **MCP Client** (`causalllm.mcp.client`): Connects to MCP servers for causal reasoning
3. **Transport Layer** (`causalllm.mcp.transport`): Handles communication (stdio, WebSocket)
4. **Tool Definitions** (`causalllm.mcp.tools`): MCP tool implementations
5. **Configuration** (`causalllm.mcp.config`): MCP configuration management

## Available MCP Tools

CausalLLM exposes the following tools via MCP:

### 1. `simulate_counterfactual`
Simulate counterfactual scenarios by comparing factual and hypothetical situations.

**Parameters:**
- `context` (string): Background context or setting
- `factual` (string): The factual scenario that actually happened  
- `intervention` (string): The hypothetical intervention or change
- `instruction` (string, optional): Additional instruction for analysis
- `temperature` (number, optional): LLM generation temperature (default: 0.7)
- `chain_of_thought` (boolean, optional): Use chain-of-thought reasoning

### 2. `generate_do_prompt`
Generate do-calculus intervention prompts for causal analysis.

**Parameters:**
- `context` (string): Base context describing the system
- `variables` (object): Dictionary mapping variable names to descriptions
- `intervention` (object): Dictionary of variable interventions {var: new_value}
- `question` (string, optional): Question to include in analysis

### 3. `extract_causal_edges`
Extract causal relationships from text descriptions.

**Parameters:**
- `scenario_description` (string): Natural language scenario description
- `model` (string, optional): LLM model to use (default: "gpt-4")
- `temperature` (number, optional): Generation temperature (default: 0.3)

### 4. `generate_reasoning_prompt`
Generate reasoning prompts from causal DAG structures.

**Parameters:**
- `dag_edges` (array): List of directed edges as [source, target] pairs
- `task` (string, optional): The reasoning task or question

### 5. `analyze_treatment_effect`
Analyze treatment effects using causal inference templates.

**Parameters:**
- `context` (string): Experimental or observational context
- `treatment` (string): Description of the treatment variable
- `outcome` (string): Description of the outcome variable

### 6. `create_causal_core`
Create a complete CausalLLM core instance for complex analysis.

**Parameters:**
- `context` (string): System context description
- `variables` (object): Dictionary mapping variable names to descriptions
- `dag_edges` (array): Causal DAG edges as [source, target] pairs

## Configuration

MCP configuration is managed through the `MCPConfig` class and can be loaded from files or environment variables.

### Configuration File Example

```json
{
  "server": {
    "name": "causalllm-server",
    "version": "1.0.0",
    "description": "CausalLLM MCP Server for causal reasoning",
    "transport": "stdio",
    "host": "localhost",
    "port": 8000,
    "max_connections": 10,
    "timeout": 30,
    "log_level": "INFO"
  },
  "client": {
    "server_name": "causalllm-server",
    "transport": "stdio",
    "host": "localhost",
    "port": 8000,
    "timeout": 30,
    "retry_attempts": 3,
    "retry_delay": 1.0
  },
  "tools": {
    "enabled_tools": [
      "simulate_counterfactual",
      "generate_do_prompt",
      "extract_causal_edges",
      "generate_reasoning_prompt",
      "analyze_treatment_effect"
    ],
    "tool_timeout": 60,
    "max_prompt_length": 10000,
    "enable_streaming": false
  }
}
```

### Environment Variables

You can also configure MCP using environment variables:

```bash
# Server configuration
export MCP_SERVER_NAME="causalllm-server"
export MCP_TRANSPORT="stdio"
export MCP_HOST="localhost"
export MCP_PORT="8000"

# Client configuration  
export MCP_CLIENT_SERVER="causalllm-server"
export MCP_CLIENT_TRANSPORT="stdio"

# Tools configuration
export MCP_ENABLED_TOOLS="simulate_counterfactual,generate_do_prompt,extract_causal_edges"
export MCP_TOOL_TIMEOUT="60"
```

## Transport Options

CausalLLM MCP supports multiple transport protocols:

### 1. Standard I/O (stdio)
Default transport for command-line integration.

```json
{
  "transport": "stdio"
}
```

### 2. WebSocket
For network-based communication.

```json
{
  "transport": "websocket",
  "host": "localhost",
  "port": 8080
}
```

**Note:** WebSocket support requires the `websockets` package:
```bash
pip install websockets
```

### 3. HTTP (Future)
HTTP transport is planned for future releases.

## Usage Examples

### Running an MCP Server

```python
import asyncio
from causalllm.mcp.server import MCPServer
from causalllm.mcp.config import load_mcp_config
from causalllm.llm_client import get_llm_client

async def run_server():
    # Load configuration
    config = load_mcp_config("mcp_config.json")
    
    # Create LLM client
    llm_client = get_llm_client("grok")
    
    # Create and start server
    server = MCPServer(config, llm_client)
    await server.start()
    
    # Run forever
    await server.run_forever()

# Run the server
asyncio.run(run_server())
```

### Using MCP Client

```python
import asyncio
from causalllm.mcp.client import MCPClient

async def use_mcp_client():
    # Create and connect client
    client = MCPClient()
    await client.connect()
    
    # List available tools
    tools = await client.list_tools()
    print(f"Available tools: {[t['name'] for t in tools]}")
    
    # Call counterfactual simulation
    result = await client.call_tool("simulate_counterfactual", {
        "context": "A student is preparing for an exam",
        "factual": "The student studied for 2 hours and got a B",
        "intervention": "The student studied for 6 hours instead"
    })
    
    print("Counterfactual result:", result)
    
    await client.disconnect()

asyncio.run(use_mcp_client())
```

### Integration with CausalLLMCore

```python
from causalllm.llm_client import get_llm_client
from causalllm.core import CausalLLMCore

# Create MCP-enabled LLM client
mcp_client = get_llm_client("mcp", "counterfactual")

# Create core with MCP integration
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

# Use MCP-specific features
if core.is_mcp_client:
    # Get available MCP tools
    tools = core.get_mcp_tools()
    
    # Create MCP core configuration
    mcp_config = core.create_causal_mcp_core()
    
    # Call MCP tools directly
    result = core.call_mcp_tool("simulate_counterfactual", {
        "context": core.context,
        "factual": "Current treatment has 80% success rate",
        "intervention": "New treatment with 95% success rate"
    })

# Regular core operations work seamlessly
counterfactual = core.simulate_counterfactual(
    factual="Patient received standard treatment",
    intervention="Patient received experimental treatment"
)
```

## Command Line Usage

### Start MCP Server
```bash
python -m causalllm.mcp.server mcp_config.json
```

### Run Examples
```bash
# Run server example
python examples/mcp_server_example.py

# Run client example  
python examples/mcp_client_example.py

# Run integration demo
python examples/mcp_integration_demo.py
```

## Testing

Run MCP integration tests:

```bash
python tests/test_mcp_integration.py
```

The test suite covers:
- Configuration management
- Transport layer functionality
- Tool definitions and execution
- Server-client communication
- Core integration features

## Error Handling

CausalLLM MCP includes comprehensive error handling:

### Common Error Scenarios

1. **Missing Dependencies**
   ```
   ImportError: WebSocket transport requires 'websockets' package
   Solution: pip install websockets
   ```

2. **Connection Failures**
   ```
   RuntimeError: MCP connection failed
   Solution: Ensure MCP server is running and accessible
   ```

3. **Invalid Configuration**
   ```
   ValueError: Invalid transport: invalid_transport
   Solution: Use 'stdio', 'websocket', or 'http'
   ```

4. **Tool Execution Errors**
   ```
   RuntimeError: Tool execution failed: <specific error>
   Solution: Check tool arguments and server logs
   ```

## Logging

MCP operations are logged using CausalLLM's centralized logging system:

```python
from causalllm.logging import setup_package_logging

# Enable MCP logging
setup_package_logging(level="INFO", log_to_file=True)
```

Log categories:
- `causalllm.mcp.server`: Server operations
- `causalllm.mcp.client`: Client operations  
- `causalllm.mcp.transport`: Transport layer
- `causalllm.mcp.tools`: Tool execution
- `causalllm.mcp.config`: Configuration management

## Integration with MCP Clients

CausalLLM MCP servers are compatible with standard MCP clients like:

- Claude Desktop
- MCP Inspector
- Custom MCP applications

### Claude Desktop Integration

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "causalllm": {
      "command": "python",
      "args": ["-m", "causalllm.mcp.server"],
      "env": {
        "MCP_SERVER_NAME": "causalllm-server"
      }
    }
  }
}
```

## Best Practices

1. **Use appropriate transport**: stdio for CLI tools, WebSocket for services
2. **Configure timeouts**: Set reasonable timeouts for long-running analyses
3. **Enable logging**: Use structured logging for production deployments
4. **Error handling**: Implement proper error handling in client applications
5. **Resource management**: Properly close connections and clean up resources
6. **Security**: Use authentication and encryption for network transports

## Troubleshooting

### Common Issues

1. **Server won't start**
   - Check port availability
   - Verify configuration file syntax
   - Ensure required dependencies are installed

2. **Client connection fails**
   - Verify server is running
   - Check network connectivity
   - Validate configuration parameters

3. **Tool calls timeout**
   - Increase tool timeout in configuration
   - Check LLM client connectivity
   - Verify input parameters are valid

4. **WebSocket connection issues**
   - Install websockets package: `pip install websockets`
   - Check firewall settings
   - Verify host/port configuration

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
from causalllm.logging import setup_package_logging
setup_package_logging(level="DEBUG", log_to_file=True)
```

## Future Enhancements

Planned MCP features:
- HTTP transport support
- Streaming responses
- Authentication mechanisms
- Tool discovery and metadata
- Performance optimizations
- Additional causal reasoning tools

## Contributing

To contribute to MCP integration:

1. Follow the existing code structure in `causalllm/mcp/`
2. Add tests for new functionality in `tests/test_mcp_integration.py`
3. Update documentation for new features
4. Ensure compatibility with MCP specification

## References

- [Model Context Protocol Specification](https://spec.modelcontextprotocol.io/)
- [CausalLLM Documentation](../README.md)
- [Examples](../examples/)
- [Test Suite](../tests/test_mcp_integration.py)