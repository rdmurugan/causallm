"""
CausalLLM Core Package

This module provides tools for causal reasoning, counterfactual analysis,
interventions using the do-operator, and SCM extraction integrated with LLMs.
"""
from .dag_parser import DAGParser
from .counterfactual_engine import CounterfactualEngine
from .prompt_templates import PromptTemplates
from .do_operator import DoOperatorSimulator
from .scm_explainer import SCMExplainer
from .utils import load_yaml, save_yaml, load_json, save_json
from .llm_client import BaseLLMClient, OpenAIClient
from .core import CausalLLMCore  # <-- new class

__all__ = [
    "DAGParser",
    "CounterfactualEngine",
    "PromptTemplates",
    "DoOperatorSimulator",
    "SCMExplainer",
    "CausalLLMCore"
]
