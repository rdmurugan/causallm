import pytest
from causalllm.counterfactual_engine import CounterfactualEngine
from causalllm.llm_client import BaseLLMClient

class MockLLMClient(BaseLLMClient):
    def chat(self, prompt: str, model: str = "gpt-4", temperature: float = 0.7) -> str:
        return "Mocked counterfactual response"

def test_simulate_counterfactual_basic():
    llm_client = MockLLMClient()
    engine = CounterfactualEngine(llm_client)

    response = engine.simulate_counterfactual(
        context="A person is in good health.",
        factual="The person exercised regularly.",
        intervention="The person did not exercise.",
        instruction="Focus on cardiovascular health effects."
    )

    assert isinstance(response, str)
    assert "Mocked counterfactual response" in response
