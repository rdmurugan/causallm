import json
from typing import Dict, Optional
from datetime import datetime
from causalllm.llm_client import BaseLLMClient
from causalllm.logging import get_logger, get_structured_logger

class CounterfactualEngine:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        model: str = "gpt-4",
        log_file: str = "logs/counterfactual_log.jsonl"
    ):
        self.logger = get_logger("causalllm.counterfactual_engine")
        self.struct_logger = get_structured_logger("counterfactual_engine", log_file)
        
        self.logger.info("Initializing CounterfactualEngine")
        
        self.llm_client = llm_client
        self.model = model
        self.log_file = log_file
        
        self.logger.info(f"CounterfactualEngine initialized with model: {model}")
        self.struct_logger.log_interaction(
            "engine_initialization",
            {
                "model": model,
                "log_file": log_file,
                "llm_client_type": type(llm_client).__name__
            }
        )

    def simulate_counterfactual(
        self,
        context: str,
        factual: str,
        intervention: str,
        instruction: Optional[str] = None,
        temperature: float = 0.7,
        chain_of_thought: bool = False,
    ) -> str:
        self.logger.info("Starting counterfactual simulation")
        self.logger.debug(f"Context length: {len(context)}, Factual length: {len(factual)}, Intervention length: {len(intervention)}")
        self.logger.debug(f"Temperature: {temperature}, Chain of thought: {chain_of_thought}")
        
        try:
            prompt = self._build_prompt(context, factual, intervention, instruction, chain_of_thought)
            
            self.logger.debug(f"Built prompt with length: {len(prompt)}")
            
            response = self.llm_client.chat(prompt, model=self.model, temperature=temperature)
            
            # Use centralized structured logging instead of custom _log_interaction
            self.struct_logger.log_interaction(
                "counterfactual_simulation",
                {
                    "context_length": len(context),
                    "factual_length": len(factual),
                    "intervention_length": len(intervention),
                    "instruction": instruction,
                    "temperature": temperature,
                    "chain_of_thought": chain_of_thought,
                    "prompt_length": len(prompt),
                    "response_length": len(response),
                    "model": self.model
                }
            )
            
            self.logger.info("Counterfactual simulation completed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in counterfactual simulation: {e}")
            self.struct_logger.log_error(e, {
                "context_length": len(context),
                "factual_length": len(factual),
                "intervention_length": len(intervention),
                "instruction": instruction,
                "temperature": temperature,
                "chain_of_thought": chain_of_thought,
                "model": self.model
            })
            raise

    def _build_prompt(
        self,
        context: str,
        factual: str,
        intervention: str,
        instruction: Optional[str],
        chain_of_thought: bool,
    ) -> str:
        self.logger.debug("Building counterfactual prompt")
        
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

        result = base_prompt.strip()
        self.logger.debug(f"Prompt built successfully with length: {len(result)}")
        return result

    def _log_interaction(self, prompt: str, response: str, temperature: float, model: str) -> None:
        """
        Legacy method maintained for backward compatibility.
        New code should use the centralized structured logging via self.struct_logger.
        """
        self.logger.warning("Using legacy _log_interaction method - consider migrating to centralized logging")
        
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
            self.logger.error(f"Failed to write legacy log: {e}")
            # Fallback to centralized logging
            self.struct_logger.log_interaction(
                "legacy_log_fallback",
                {
                    "model": model,
                    "temperature": temperature,
                    "prompt_length": len(prompt),
                    "response_length": len(response)
                }
            )
