from typing import Dict, List, Optional, Tuple
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
