
from typing import Dict, Any
import os
from causalllm.logging import get_logger

class PromptTemplates:
    @staticmethod
    def treatment_effect_estimation(context: str, treatment: str, outcome: str) -> str:
        logger = get_logger("causalllm.prompt_templates")
        
        # Input validation
        if not context or not context.strip():
            logger.error("Empty context provided for treatment effect estimation")
            raise ValueError("Context cannot be empty")
        
        if not treatment or not treatment.strip():
            logger.error("Empty treatment variable provided")
            raise ValueError("Treatment variable cannot be empty")
        
        if not outcome or not outcome.strip():
            logger.error("Empty outcome variable provided")
            raise ValueError("Outcome variable cannot be empty")
        
        logger.debug("Generating treatment effect estimation prompt")
        
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
        logger = get_logger("causalllm.prompt_templates")
        
        # Input validation
        if not path or not path.strip():
            logger.error("Empty template file path provided")
            raise ValueError("Template file path cannot be empty")
        
        # Check if file exists
        if not os.path.exists(path):
            logger.error(f"Template file not found: {path}")
            raise FileNotFoundError(f"Template file not found: {path}")
        
        # Check if file is readable
        if not os.access(path, os.R_OK):
            logger.error(f"Template file not readable: {path}")
            raise PermissionError(f"Cannot read template file: {path}")
        
        logger.debug(f"Loading template from file: {path}")
        
        try:
            from jinja2 import Template, TemplateError
        except ImportError as e:
            logger.error("Jinja2 package not available")
            raise ImportError("Jinja2 package is required but not installed. Run: pip install jinja2") from e
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                template_content = f.read()
                
            if not template_content.strip():
                logger.warning(f"Template file is empty: {path}")
                return ""
            
            tmpl = Template(template_content)
            result = tmpl.render(**kwargs)
            
            logger.debug(f"Successfully rendered template from {path}")
            return result
            
        except TemplateError as e:
            logger.error(f"Jinja2 template error in {path}: {e}")
            raise ValueError(f"Template rendering error in {path}: {e}") from e
        except Exception as e:
            logger.error(f"Failed to load template from {path}: {e}")
            raise RuntimeError(f"Failed to load template from {path}: {e}") from e

    @staticmethod
    def custom(template_name: str, **kwargs: Any) -> str:
        return f"[Template: {template_name}]\n" + "\n".join([f"{k}: {v}" for k, v in kwargs.items()])
