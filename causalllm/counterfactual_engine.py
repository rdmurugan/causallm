import json
from typing import Dict, Optional
from datetime import datetime
from causalllm.llm_client import BaseLLMClient

class CounterfactualEngine:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        model: str = "gpt-4",
        log_file: str = "logs/counterfactual_log.jsonl"
    ):
        self.llm_client = llm_client
        self.model = model
        self.log_file = log_file

    def simulate_counterfactual(
        self,
        context: str,
        factual: str,
        intervention: str,
        instruction: Optional[str] = None,
        temperature: float = 0.7,
        chain_of_thought: bool = False,
    ) -> str:
        prompt = self._build_prompt(context, factual, intervention, instruction, chain_of_thought)
        response = self.llm_client.chat(prompt, model=self.model, temperature=temperature)
        self._log_interaction(prompt, response, temperature, self.model)
        return response

    def _build_prompt(
        self,
        context: str,
        factual: str,
        intervention: str,
        instruction: Optional[str],
        chain_of_thought: bool,
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
            base_prompt += f"\n\nInstruction: {instruction.strip()}"

        if chain_of_thought:
            base_prompt += "\n\nThink step by step before giving your final answer."

        return base_prompt.strip()

    def _log_interaction(self, prompt: str, response: str, temperature: float, model: str) -> None:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "temperature": temperature,
            "prompt": prompt,
            "response": response,
        }
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Warning: Failed to log interaction: {e}")
