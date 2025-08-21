from typing import Dict, List, Optional, Tuple
from causalllm.dag_parser import DAGParser
from causalllm.do_operator import DoOperatorSimulator
from causalllm.prompt_templates import PromptTemplates
from causalllm.counterfactual_engine import CounterfactualEngine
from causalllm.scm_explainer import SCMExplainer
from causalllm.llm_client import BaseLLMClient, get_llm_client


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
        self.context = context
        self.variables = variables
        self.dag = DAGParser(dag_edges)
        self.do_operator = DoOperatorSimulator(context, variables)
        self.templates = PromptTemplates()
        
        # Default to configured LLM if none provided
        self.llm_client = llm_client or get_llm_client()
        self.counterfactual = CounterfactualEngine(self.llm_client)
        self.scm = SCMExplainer(self.llm_client)

    def simulate_do(self, intervention: Dict[str, str], question: Optional[str] = None) -> str:
        """
        Generates a prompt simulating an intervention using do-calculus.

        Args:
            intervention (Dict[str, str]): Variable intervention values.
            question (Optional[str]): Optional question to include in the prompt.

        Returns:
            str: Generated prompt simulating the do-intervention.
        """
        return self.do_operator.generate_do_prompt(intervention, question)

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
        return self.counterfactual.simulate_counterfactual(
            self.context, factual, intervention, instruction
        )

    def generate_reasoning_prompt(self, task: str = "") -> str:
        """
        Generates a reasoning prompt from the DAG structure and task.

        Args:
            task (str): Optional reasoning task name.

        Returns:
            str: Prompt derived from the DAG.
        """
        return self.dag.to_prompt(task)
