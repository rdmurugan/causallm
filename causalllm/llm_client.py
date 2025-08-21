from typing import Protocol
from openai import OpenAI


class BaseLLMClient(Protocol):
    def chat(self, prompt: str, model: str = "gpt-4", temperature: float = 0.7) -> str:
        ...


class OpenAIClient:
    def __init__(self) -> None:
        self.client = OpenAI()

    def chat(self, prompt: str, model: str = "gpt-4", temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        # Use .content directly and guard against None
        content = response.choices[0].message.content
        return content.strip() if content else ""
