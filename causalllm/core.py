from typing import Any, Dict, List, Optional, Tuple
from causalllm.dag_parser import DAGParser
from causalllm.do_operator import DoOperatorSimulator
from causalllm.prompt_templates import PromptTemplates
from causalllm.counterfactual_engine import CounterfactualEngine
from causalllm.scm_explainer import SCMExplainer
from causalllm.llm_client import BaseLLMClient, get_llm_client
from causalllm.logging import get_logger, get_structured_logger


class CausalLLMCore:
    """
    Core class that coordinates various modules to perform causal reasoning
    using DAG parsing, do-calculus simulation, counterfactuals, and SCM explanations.
    """

    def __init__(
        self,
        context: str,
        variables: Dict[str, str],
        dag_edges: List[Tuple[str, str]],
        llm_client: Optional[BaseLLMClient] = None,
    ) -> None:
        """
        Initializes the core engine for causal LLM tasks.

        Args:
            context (str): The textual context of the data or system.
            variables (Dict[str, str]): Mapping of variable names to their descriptions.
            dag_edges (List[Tuple[str, str]]): Directed edges representing the DAG structure.
            llm_client (Optional[BaseLLMClient]): A language model client implementing the chat interface.
                                                 If not provided, it is inferred from environment.
        """
        self.logger = get_logger("causalllm.core")
        self.struct_logger = get_structured_logger("core")
        
        self.logger.info("Initializing CausalLLMCore")
        self.logger.debug(f"Context length: {len(context)}, Variables: {list(variables.keys())}, DAG edges: {len(dag_edges)}")
        
        self.context = context
        self.variables = variables
        self.dag = DAGParser(dag_edges)
        self.do_operator = DoOperatorSimulator(context, variables)
        self.templates = PromptTemplates()
        
        # Default to configured LLM if none provided
        self.llm_client = llm_client or get_llm_client()
        self.logger.info(f"Using LLM client: {type(self.llm_client).__name__}")
        
        # Check if we're using MCP client for special handling
        self.is_mcp_client = type(self.llm_client).__name__ == "MCPClient"
        if self.is_mcp_client:
            self.logger.info("MCP client detected - enabling MCP-specific features")
        
        self.counterfactual = CounterfactualEngine(self.llm_client)
        self.scm = SCMExplainer(self.llm_client)
        
        self.struct_logger.log_interaction(
            "initialization",
            {
                "variables_count": len(variables),
                "dag_edges_count": len(dag_edges),
                "context_length": len(context),
                "llm_client_type": type(self.llm_client).__name__
            }
        )
        
        self.logger.info("CausalLLMCore initialization completed")

    def simulate_do(self, intervention: Dict[str, str], question: Optional[str] = None) -> str:
        """
        Generates a prompt simulating an intervention using do-calculus.

        Args:
            intervention (Dict[str, str]): Variable intervention values.
            question (Optional[str]): Optional question to include in the prompt.

        Returns:
            str: Generated prompt simulating the do-intervention.
        """
        self.logger.info(f"Simulating do-operation with intervention: {intervention}")
        self.logger.debug(f"Question: {question}")
        
        try:
            result = self.do_operator.generate_do_prompt(intervention, question)
            
            self.struct_logger.log_interaction(
                "do_simulation",
                {
                    "intervention": intervention,
                    "question": question,
                    "result_length": len(result)
                }
            )
            
            self.logger.info("Do-operation simulation completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in do-operation simulation: {e}")
            self.struct_logger.log_error(e, {"intervention": intervention, "question": question})
            raise

    def simulate_counterfactual(
        self, 
        factual: str, 
        intervention: str, 
        instruction: Optional[str] = None
    ) -> str:
        """
        Generates a counterfactual explanation based on a factual scenario and intervention.

        Args:
            factual (str): The factual statement.
            intervention (str): The hypothetical intervention.
            instruction (Optional[str]): Additional instruction for the model.

        Returns:
            str: Counterfactual explanation or reasoning prompt.
        """
        self.logger.info("Starting counterfactual simulation")
        self.logger.debug(f"Factual: {factual[:100]}..., Intervention: {intervention[:100]}...")
        
        try:
            result = self.counterfactual.simulate_counterfactual(
                self.context, factual, intervention, instruction
            )
            
            self.struct_logger.log_interaction(
                "counterfactual_simulation",
                {
                    "factual_length": len(factual),
                    "intervention_length": len(intervention),
                    "instruction": instruction,
                    "result_length": len(result)
                }
            )
            
            self.logger.info("Counterfactual simulation completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in counterfactual simulation: {e}")
            self.struct_logger.log_error(e, {
                "factual_length": len(factual),
                "intervention_length": len(intervention),
                "instruction": instruction
            })
            raise

    def generate_reasoning_prompt(self, task: str = "") -> str:
        """
        Generates a reasoning prompt from the DAG structure and task.

        Args:
            task (str): Optional reasoning task name.

        Returns:
            str: Prompt derived from the DAG.
        """
        self.logger.info(f"Generating reasoning prompt for task: {task}")
        
        try:
            result = self.dag.to_prompt(task)
            
            self.struct_logger.log_interaction(
                "reasoning_prompt_generation",
                {
                    "task": task,
                    "result_length": len(result)
                }
            )
            
            self.logger.info("Reasoning prompt generated successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating reasoning prompt: {e}")
            self.struct_logger.log_error(e, {"task": task})
            raise

    def get_mcp_tools(self) -> List[str]:
        """
        Get list of available MCP tools if using MCP client.
        
        Returns:
            List[str]: List of available MCP tool names.
        """
        if not self.is_mcp_client:
            self.logger.warning("get_mcp_tools called on non-MCP client")
            return []
            
        self.logger.info("Retrieving available MCP tools")
        
        try:
            # Access the underlying MCP client
            if hasattr(self.llm_client, 'mcp_client'):
                import asyncio
                tools = asyncio.run(self.llm_client.mcp_client.list_tools())
                tool_names = [tool['name'] for tool in tools]
                
                self.logger.info(f"Found {len(tool_names)} MCP tools: {tool_names}")
                return tool_names
            else:
                self.logger.warning("MCP client does not have expected mcp_client attribute")
                return []
                
        except Exception as e:
            self.logger.error(f"Error retrieving MCP tools: {e}")
            self.struct_logger.log_error(e, {"operation": "get_mcp_tools"})
            return []

    def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Direct call to MCP tool if using MCP client.
        
        Args:
            tool_name (str): Name of the MCP tool to call.
            arguments (Dict[str, Any]): Arguments to pass to the tool.
            
        Returns:
            Dict[str, Any]: Tool execution result.
        """
        if not self.is_mcp_client:
            raise ValueError("call_mcp_tool can only be used with MCP client")
            
        self.logger.info(f"Calling MCP tool: {tool_name}")
        self.logger.debug(f"Arguments: {arguments}")
        
        try:
            # Access the underlying MCP client
            if hasattr(self.llm_client, 'mcp_client'):
                import asyncio
                result = asyncio.run(self.llm_client.mcp_client.call_tool(tool_name, arguments))
                
                self.struct_logger.log_interaction(
                    "mcp_tool_call",
                    {
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "result_type": type(result).__name__
                    }
                )
                
                self.logger.info(f"MCP tool {tool_name} completed successfully")
                return result
            else:
                raise RuntimeError("MCP client does not have expected mcp_client attribute")
                
        except Exception as e:
            self.logger.error(f"Error calling MCP tool {tool_name}: {e}")
            self.struct_logger.log_error(e, {
                "tool_name": tool_name,
                "arguments": arguments,
                "operation": "call_mcp_tool"
            })
            raise

    def create_causal_mcp_core(self) -> Dict[str, Any]:
        """
        Create a comprehensive causal core representation for MCP.
        Useful for initializing MCP servers with this core's configuration.
        
        Returns:
            Dict[str, Any]: Core configuration suitable for MCP tools.
        """
        self.logger.info("Creating causal MCP core representation")
        
        try:
            config = {
                "context": self.context,
                "variables": self.variables,
                "dag_edges": list(self.dag.graph.edges()),
                "capabilities": {
                    "simulate_do": True,
                    "simulate_counterfactual": True,
                    "generate_reasoning_prompt": True,
                    "extract_causal_edges": True,
                    "mcp_integration": self.is_mcp_client
                }
            }
            
            self.struct_logger.log_interaction(
                "mcp_core_creation",
                {
                    "variables_count": len(self.variables),
                    "dag_edges_count": self.dag.graph.number_of_edges(),
                    "context_length": len(self.context),
                    "is_mcp_client": self.is_mcp_client
                }
            )
            
            self.logger.info("Causal MCP core representation created")
            return config
            
        except Exception as e:
            self.logger.error(f"Error creating causal MCP core: {e}")
            self.struct_logger.log_error(e, {"operation": "create_causal_mcp_core"})
            raise

    @classmethod
    def from_mcp_config(cls, mcp_config: Dict[str, Any], llm_client: Optional[BaseLLMClient] = None) -> "CausalLLMCore":
        """
        Create CausalLLMCore instance from MCP configuration.
        
        Args:
            mcp_config (Dict[str, Any]): MCP configuration dictionary.
            llm_client (Optional[BaseLLMClient]): Optional LLM client to use.
            
        Returns:
            CausalLLMCore: Initialized core instance.
        """
        logger = get_logger("causalllm.core.from_mcp")
        logger.info("Creating CausalLLMCore from MCP configuration")
        
        try:
            # Extract required fields from MCP config
            context = mcp_config.get("context", "")
            variables = mcp_config.get("variables", {})
            dag_edges = [tuple(edge) for edge in mcp_config.get("dag_edges", [])]
            
            if not context:
                raise ValueError("MCP configuration missing required 'context' field")
            if not variables:
                raise ValueError("MCP configuration missing required 'variables' field")
            if not dag_edges:
                raise ValueError("MCP configuration missing required 'dag_edges' field")
            
            # Create core instance
            core = cls(
                context=context,
                variables=variables,
                dag_edges=dag_edges,
                llm_client=llm_client
            )
            
            logger.info("CausalLLMCore created successfully from MCP configuration")
            return core
            
        except Exception as e:
            logger.error(f"Error creating CausalLLMCore from MCP config: {e}")
            raise RuntimeError(f"Failed to create core from MCP config: {e}") from e
