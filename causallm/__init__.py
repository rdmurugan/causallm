"""
CausalLLM - Open Source Causal Inference Library

Discover cause-and-effect relationships in your data using Large Language Models 
and statistical validation.

MIT License - Free for commercial and non-commercial use.
For enterprise features, visit: https://causallm.com/enterprise
"""

from .core import (
    CausalLLM,
    CausalDiscoveryEngine,
    DAGParser,
    DoOperator,
    CounterfactualEngine,
    DiscoveryMethod,
    CausalGraph,
    InterventionResult
)

# Version info
__version__ = "3.0.0"
__license__ = "MIT"
__author__ = "CausalLLM Team"
__email__ = "opensource@causallm.com"

# Main exports
__all__ = [
    'CausalLLM',
    'CausalDiscoveryEngine',
    'DAGParser', 
    'DoOperator',
    'CounterfactualEngine',
    'DiscoveryMethod',
    'CausalGraph',
    'InterventionResult'
]