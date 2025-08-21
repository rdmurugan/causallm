# integrations/langchain_agent.py

from langchain.agents import Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

from causalllm.counterfactual_engine import CounterfactualEngine
from causalllm.do_operator import DoOperatorSimulator
from causalllm.prompt_templates import PromptTemplates


class CausalAgent:
    """
    LangChain-compatible causal reasoning agent using CausalLLM tools.
    """

    def __init__(self, llm=None, temperature=0.7):
        self.llm = llm or ChatOpenAI(temperature=temperature, model="gpt-4")
        self.counterfactual_engine = CounterfactualEngine(llm_client=self.llm)
        self.agent = self._build_agent()

    def _build_agent(self):
        tools = [
            Tool(
                name="SimulateCounterfactual",
                func=self._simulate_counterfactual,
                description="Use to simulate 'what if' scenarios given a factual story and a hypothetical change."
            ),
            Tool(
                name="RunDoOperator",
                func=self._run_do_operator,
                description="Use to apply do(X=x) interventions and observe the expected outcome."
            )
        ]

        return initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def _simulate_counterfactual(self, input: str) -> str:
        try:
            context, factual, intervention = input.split(";")
            return self.counterfactual_engine.simulate_counterfactual(
                context=context.strip(),
                factual=factual.strip(),
                intervention=intervention.strip()
            )
        except Exception as e:
            return f"Error parsing input: {str(e)}"

    def _run_do_operator(self, input: str) -> str:
        try:
            parts = input.split("|")
            base_context = parts[0].strip()
            interventions = {
                k.strip(): v.strip() for k, v in (pair.split("=") for pair in parts[1].split(","))
            }
            question = parts[2].strip() if len(parts) > 2 else None

            do_sim = DoOperatorSimulator(base_context, variables={})
            return do_sim.generate_do_prompt(interventions, question)
        except Exception as e:
            return f"Error parsing input: {str(e)}"

    def run(self, query: str):
        return self.agent.run(query)
