from typing import Protocol, Any
import os

# Base interface for all LLM clients
class BaseLLMClient(Protocol):
    def chat(self, prompt: str, model: str = "", temperature: float = 0.7) -> str: ...


# OpenAI Client (Project API Key version)
class OpenAIClient:
    def __init__(self, model: str = "gpt-4") -> None:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY", "your-default-api-key")
        project_id = os.getenv("OPENAI_PROJECT_ID", "your-default-project-id")

        if not api_key or not project_id:
            raise ValueError("OPENAI_API_KEY and OPENAI_PROJECT_ID must be set.")

        self.client = OpenAI(api_key=api_key, project=project_id)
        self.default_model = model

    def chat(self, prompt: str, model: str = "", temperature: float = 0.7) -> str:
        use_model = model or self.default_model
        response: Any = self.client.chat.completions.create(
            model=use_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()


# Local LLaMA (example: running Ollama or similar)
class LLaMAClient:
    def __init__(self, model: str = "llama3") -> None:
        import requests
        self.base_url = os.getenv("LLAMA_API_URL", "http://localhost:11434")
        self.default_model = model

    def chat(self, prompt: str, model: str = "", temperature: float = 0.7) -> str:
        import requests
        use_model = model or self.default_model
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": use_model, "prompt": prompt, "temperature": temperature},
            timeout=60
        )
        result = response.json()
        return (result.get("response", "") or "").strip()


# Grok Client (Mock/Example)
class GrokClient:
    def __init__(self, model: str = "grok-1") -> None:
        self.api_key = os.getenv("GROK_API_KEY", "dummy-key")
        if not self.api_key:
            raise ValueError("GROK_API_KEY must be set.")
        self.default_model = model

    def chat(self, prompt: str, model: str = "", temperature: float = 0.7) -> str:
        use_model = model or self.default_model
        return f"[Grok-{use_model}]: {prompt} (simulated)"


# Factory function
def get_llm_client(provider: str = "openai", model: str = "") -> BaseLLMClient:
    client: BaseLLMClient
    provider = provider.lower()

    if provider == "openai":
        client = OpenAIClient(model=model)
    elif provider == "llama":
        client = LLaMAClient(model=model)
    elif provider == "grok":
        client = GrokClient(model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return client
