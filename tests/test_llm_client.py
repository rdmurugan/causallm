from causalllm.llm_client import OpenAIClient

def test_chat_response(monkeypatch):
    class DummyResponse:
        class Choice:
            def __init__(self):
                self.message = type("Message", (), {"content": "Hello World!"})
        choices = [Choice()]
    
    class DummyClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(*args, **kwargs):
                    return DummyResponse()
            completions = Completions()
        chat = Chat()
    
    monkeypatch.setattr("causalllm.llm_client.OpenAI", lambda: DummyClient())

    client = OpenAIClient()
    result = client.chat("Say hello")
    assert result == "Hello World!"
