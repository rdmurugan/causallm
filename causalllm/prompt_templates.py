
from typing import Dict, Any
from jinja2 import Template

class PromptTemplates:
    @staticmethod
    def treatment_effect_estimation(context: str, treatment: str, outcome: str) -> str:
        return f"""
You are an expert in causal inference.

Context:
{context.strip()}

Treatment Variable:
{treatment.strip()}

Outcome Variable:
{outcome.strip()}

Task:
Estimate the average treatment effect of the treatment on the outcome.
Explain your reasoning and any assumptions made.
""".strip()

    @staticmethod
    def counterfactual_reasoning(context: str, factual: str, intervention: str) -> str:
        return f"""
You are an AI system trained in counterfactual analysis.

Context:
{context.strip()}

Factual Scenario:
{factual.strip()}

Intervention (counterfactual change):
{intervention.strip()}

Describe what would likely happen under this counterfactual scenario, and explain why.
""".strip()

    @staticmethod
    def causal_chain_of_thought(goal: str, causal_steps: Dict[int, str]) -> str:
        steps_str = "\n".join([f"Step {i}: {desc}" for i, desc in causal_steps.items()])
        return f"""
Your goal is to achieve: {goal.strip()}

Causal reasoning steps:
{steps_str}

Explain how each step leads to the next and how the chain produces the outcome.
""".strip()

    @staticmethod
    def from_template_file(path: str, **kwargs: Any) -> str:
        with open(path) as f:
            tmpl = Template(f.read())
        return tmpl.render(**kwargs)

    @staticmethod
    def custom(template_name: str, **kwargs: Any) -> str:
        return f"[Template: {template_name}]\n" + "\n".join([f"{k}: {v}" for k, v in kwargs.items()])
