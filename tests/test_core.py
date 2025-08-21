from causalllm.core import CausalLLMCore
from causalllm.llm_client import BaseLLMClient

class MockLLMClient(BaseLLMClient):
    def chat(self, prompt: str, model: str = "gpt-4", temperature: float = 0.7) -> str:
        return "Mocked response"

def test_init():
    mock_client = MockLLMClient()
    engine = CausalLLMCore("context", {"X": "variable"}, [("X", "Y")], mock_client)
    assert engine.context == "context"
    assert engine.variables == {"X": "variable"}
