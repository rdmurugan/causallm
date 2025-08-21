from typing import Dict, Optional

class DoOperatorSimulator:
    def __init__(self, base_context: str, variables: Dict[str, str]):
        self.base_context = base_context
        self.variables = variables.copy()

    def intervene(self, interventions: Dict[str, str]) -> str:
        modified_context = self.base_context

        for var, new_val in interventions.items():
            if var not in self.variables:
                raise ValueError(f"Variable '{var}' not in base context.")
            original_val = self.variables[var]
            modified_context = modified_context.replace(original_val, new_val)
            self.variables[var] = new_val

        return modified_context

    def generate_do_prompt(
        self,
        interventions: Dict[str, str],
        question: Optional[str] = None
    ) -> str:
        modified_context = self.intervene(interventions)
        intervention_desc = ", ".join([f"{k} := {v}" for k, v in interventions.items()])
        prompt = f"""
You are a causal inference model.

Base scenario:
{self.base_context}

Intervention applied:
do({intervention_desc})

Resulting scenario:
{modified_context}

{question if question else "What is the expected impact of this intervention?"}
"""
        return prompt.strip()
