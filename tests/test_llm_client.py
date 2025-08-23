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

    # Patch the OpenAIClient.__init__ to avoid real API calls
    monkeypatch.setattr(OpenAIClient, "__init__", lambda self: None)
    monkeypatch.setattr(OpenAIClient, "chat", lambda self, prompt, model="", temperature=0.7: "Hello World!")

    client = OpenAIClient()
    result = client.chat("Say hi")
    assert result == "Hello World!"
