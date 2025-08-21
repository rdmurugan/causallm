
from typing import List, Tuple
import re
from causalllm.llm_client import BaseLLMClient

class SCMExplainer:
    def __init__(self, llm_client: BaseLLMClient, model: str = "gpt-4") -> None:
        self.llm_client = llm_client
        self.model = model

    def extract_variables_and_edges(self, scenario_description: str) -> List[Tuple[str, str]]:
        prompt = f"""
You're a causal inference modeler.

Read the following scenario and extract causal relationships in the form of edges (A -> B).

Respond only with a list of pairs like:
(A, B)
(B, C)

Scenario:
{scenario_description.strip()}
        """.strip()

        response = self.llm_client.chat(prompt, model=self.model, temperature=0.3)
        return self._parse_edges(response)

    def _parse_edges(self, raw_text: str) -> List[Tuple[str, str]]:
        pattern = r"\(([^,]+),\s*([^)]+)\)"
        return re.findall(pattern, raw_text)
