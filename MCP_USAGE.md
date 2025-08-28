# CausalLLM MCP Server Usage Guide

CausalLLM provides Model Context Protocol (MCP) server capabilities to expose causal reasoning tools to compatible clients like Claude Desktop, VS Code, and other MCP-enabled applications.

## Overview

The MCP server exposes CausalLLM's causal analysis capabilities as standardized tools that can be called by MCP clients. This allows you to perform causal reasoning tasks directly within your preferred applications.

## Available Tools

### 1. simulate_counterfactual
Generate counterfactual scenarios by comparing what actually happened with hypothetical alternatives.

**Input:**
```json
{
  "context": "Background context or setting",
  "factual": "What actually happened",
  "intervention": "Hypothetical change to consider"
}
```

**Example:**
```json
{
  "context": "A clinical trial testing a new drug for diabetes patients",
  "factual": "Patients received standard insulin therapy",
  "intervention": "Patients received the experimental drug instead"
}
```

### 2. generate_do_prompt
Create do-calculus intervention prompts for formal causal analysis.

**Input:**
```json
{
  "context": "System description",
  "variables": {"var1": "description", "var2": "description"},
  "intervention": {"variable": "new_value"}
}
```

**Example:**
```json
{
  "context": "Healthcare treatment effectiveness study",
  "variables": {
    "age": "Patient age in years",
    "treatment": "Type of treatment received",
    "outcome": "Recovery time in days"
  },
  "intervention": {"treatment": "intensive_therapy"}
}
```

### 3. extract_causal_edges
Extract causal relationships from natural language descriptions.

**Input:**
```json
{
  "scenario_description": "Natural language text describing causal relationships"
}
```

**Example:**
```json
{
  "scenario_description": "Older patients tend to have worse health outcomes. The treatment received affects recovery time. Patients with higher socioeconomic status have better access to treatments, which leads to better outcomes."
}
```

### 4. generate_reasoning_prompt
Create structured reasoning prompts from causal graph structures.

**Input:**
```json
{
  "dag_edges": [["cause1", "effect1"], ["cause2", "effect2"]],
  "task": "Reasoning task description"
}
```

### 5. analyze_treatment_effect
Analyze treatment effectiveness in various domains.

**Input:**
```json
{
  "treatment": "Treatment variable",
  "outcome": "Outcome variable", 
  "data_description": "Description of the data context"
}
```

## Setup Instructions

### 1. Test MCP Tools Locally

```bash
# Test MCP tools functionality
python examples/mcp_server_example.py --test

# View configuration
python examples/mcp_server_example.py --config
```

### 2. Claude Desktop Integration

1. Locate your Claude Desktop configuration file:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

2. Add CausalLLM MCP server configuration:

```json
{
  "mcpServers": {
    "causallm": {
      "command": "python",
      "args": [
        "/full/path/to/causallm/examples/mcp_server_example.py",
        "--server"
      ],
      "env": {}
    }
  }
}
```

3. Restart Claude Desktop

4. Look for CausalLLM tools in the tools menu

### 3. VS Code Integration

For VS Code with MCP support:

1. Install an MCP-compatible extension
2. Configure the extension to use CausalLLM MCP server
3. Point to the server script: `examples/mcp_server_example.py --server`

## Usage Examples

### Example 1: Healthcare Counterfactual Analysis

**Scenario**: Analyze what would have happened if a different treatment was used.

1. Use the `simulate_counterfactual` tool
2. Provide context about the clinical scenario
3. Describe the actual treatment given
4. Specify the alternative treatment to consider
5. Review the generated counterfactual analysis

### Example 2: Business Process Analysis

**Scenario**: Extract causal relationships from a business process description.

1. Use the `extract_causal_edges` tool
2. Provide a description of your business process
3. Review the extracted causal relationships
4. Use the relationships for further analysis

### Example 3: Policy Impact Assessment

**Scenario**: Generate prompts for analyzing policy interventions.

1. Use the `generate_do_prompt` tool
2. Define your policy context and variables
3. Specify the intervention you want to analyze
4. Use the generated prompt with statistical analysis tools

## Configuration Options

The MCP server can be configured through the configuration object:

```python
{
  "server": {
    "name": "causallm-server",
    "version": "1.0.0",
    "transport": "stdio",  # or "websocket", "http"
    "host": "localhost",
    "port": 8000,
    "timeout": 30
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
    "max_prompt_length": 10000
  }
}
```

## Limitations and Notes

1. **Development Status**: The MCP implementation is currently in development and may have some import issues that need to be resolved.

2. **Mock Implementation**: The example provided includes mock implementations to demonstrate functionality until the full MCP integration is completed.

3. **Tool Reliability**: Results are demonstrations of the intended functionality. Production use would require more robust error handling and validation.

4. **Performance**: Tools are designed for interactive use rather than high-volume batch processing.

## Troubleshooting

### Common Issues

1. **Import Errors**: If you encounter import errors, ensure all dependencies are installed and paths are correct.

2. **Configuration Not Found**: Make sure the Claude Desktop configuration file exists and is properly formatted.

3. **Tools Not Appearing**: Restart Claude Desktop after making configuration changes.

4. **Path Issues**: Use absolute paths in MCP configuration to avoid path resolution problems.

### Debug Steps

1. Test tools locally first: `python examples/mcp_server_example.py --test`
2. Verify configuration: `python examples/mcp_server_example.py --config`  
3. Check Claude Desktop logs for error messages
4. Ensure Python environment is accessible to Claude Desktop

## Contributing

The MCP integration is actively being developed. Contributions are welcome for:

- Fixing import issues in the MCP module
- Implementing additional causal reasoning tools
- Improving error handling and validation
- Adding support for more MCP clients
- Performance optimizations

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.