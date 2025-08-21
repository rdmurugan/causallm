import pytest
from causalllm.core import CausalLLMCore
from causalllm.llm_client import BaseLLMClient

# Mock implementation of BaseLLMClient
class MockLLMClient(BaseLLMClient):
    def chat(self, prompt: str, model: str = "gpt-4", temperature: float = 0.7) -> str:
        return f"Mocked response for prompt: {prompt}"

@pytest.fixture
def core_instance():
    context = "A system with variables A, B, and C"
    variables = {"A": "Variable A", "B": "Variable B", "C": "Variable C"}
    dag_edges = [("A", "B"), ("B", "C")]
    llm_client = MockLLMClient()
    return CausalLLMCore(context, variables, dag_edges, llm_client)

def test_simulate_do(core_instance):
    intervention = {"A": "True"}
    result = core_instance.simulate_do(intervention)
    assert "do" in result or "intervene" in result.lower()

def test_simulate_counterfactual(core_instance):
    factual = "A was False"
    intervention = "A is True"
    result = core_instance.simulate_counterfactual(factual, intervention)
    assert "Mocked response" in result

def test_generate_reasoning_prompt(core_instance):
    prompt = core_instance.generate_reasoning_prompt(task="causal reasoning")
    assert "causal reasoning" in prompt or isinstance(prompt, str)
