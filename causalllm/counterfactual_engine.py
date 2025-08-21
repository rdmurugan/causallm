
from typing import Dict, Optional, Any
from causalllm.llm_client import BaseLLMClient

class CounterfactualEngine:
    def __init__(self, llm_client: BaseLLMClient, model: str = "gpt-4"):
        self.llm_client = llm_client
        self.model = model

    def simulate_counterfactual(
        self,
        context: str,
        factual: str,
        intervention: str,
        instruction: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        prompt = self._build_prompt(context, factual, intervention, instruction)
        response = self.llm_client.chat(prompt, model=self.model, temperature=temperature)
        return response

    def _build_prompt(
        self,
        context: str,
        factual: str,
        intervention: str,
        instruction: Optional[str]
    ) -> str:
        base_prompt = f"""
You are a causal reasoning expert.

Context:
{context.strip()}

Factual Scenario:
{factual.strip()}

Counterfactual Intervention:
{intervention.strip()}

Please describe the most plausible counterfactual outcome based on this change.
"""
        if instruction:
            base_prompt += f"\nAdditional Instruction: {instruction.strip()}"
        return base_prompt.strip()
