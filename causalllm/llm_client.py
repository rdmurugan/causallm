from typing import Protocol, Any
import os
import time
from causalllm.logging import get_logger, get_structured_logger

# Base interface for all LLM clients
class BaseLLMClient(Protocol):
    def chat(self, prompt: str, model: str = "", temperature: float = 0.7) -> str: ...


# === OpenAI Client (Project API Key version) ===
class OpenAIClient:
    def __init__(self, model: str = "gpt-4") -> None:
        from openai import OpenAI

        self.logger = get_logger("causalllm.llm_client.openai")
        self.struct_logger = get_structured_logger("llm_client_openai")
        
        self.logger.info("Initializing OpenAI client")
        
        api_key = os.getenv("OPENAI_API_KEY")
        project_id = os.getenv("OPENAI_PROJECT_ID")

        if not api_key or not project_id:
            self.logger.error("OPENAI_API_KEY and OPENAI_PROJECT_ID must be set")
            raise ValueError("OPENAI_API_KEY and OPENAI_PROJECT_ID must be set.")

        self.client = OpenAI(api_key=api_key, project=project_id)
        self.default_model = model
        
        self.logger.info(f"OpenAI client initialized with model: {model}")
        self.struct_logger.log_interaction(
            "client_initialization",
            {"default_model": model, "has_project_id": bool(project_id)}
        )

    def chat(self, prompt: str, model: str = "", temperature: float = 0.7) -> str:
        use_model = model or self.default_model
        
        self.logger.info(f"Making OpenAI chat request with model: {use_model}")
        self.logger.debug(f"Prompt length: {len(prompt)}, Temperature: {temperature}")
        
        start_time = time.time()
        try:
            response: Any = self.client.chat.completions.create(
                model=use_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            
            duration = time.time() - start_time
            result = (response.choices[0].message.content or "").strip()
            
            self.logger.info(f"OpenAI chat completed in {duration:.2f}s")
            
            self.struct_logger.log_interaction(
                "chat_completion",
                {
                    "model": use_model,
                    "temperature": temperature,
                    "prompt_length": len(prompt),
                    "response_length": len(result),
                    "duration_seconds": duration,
                    "tokens_used": getattr(response.usage, 'total_tokens', None) if hasattr(response, 'usage') else None
                }
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"OpenAI chat failed after {duration:.2f}s: {e}")
            self.struct_logger.log_error(e, {
                "model": use_model,
                "temperature": temperature,
                "prompt_length": len(prompt),
                "duration_seconds": duration
            })
            raise


# === Local LLaMA (via Ollama or similar) ===
class LLaMAClient:
    def __init__(self, model: str = "llama3") -> None:
        self.logger = get_logger("causalllm.llm_client.llama")
        self.struct_logger = get_structured_logger("llm_client_llama")
        
        self.base_url = os.getenv("LLAMA_API_URL", "http://localhost:11434")
        self.default_model = model
        
        self.logger.info(f"Initializing LLaMA client with URL: {self.base_url}, model: {model}")
        self.struct_logger.log_interaction(
            "client_initialization",
            {"base_url": self.base_url, "default_model": model}
        )

    def chat(self, prompt: str, model: str = "", temperature: float = 0.7) -> str:
        import requests

        use_model = model or self.default_model
        
        self.logger.info(f"Making LLaMA chat request with model: {use_model}")
        self.logger.debug(f"Prompt length: {len(prompt)}, Temperature: {temperature}")
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": use_model, "prompt": prompt, "temperature": temperature},
                timeout=60
            )
            response.raise_for_status()
            
            duration = time.time() - start_time
            json_resp = response.json()
            result = (json_resp.get("response") or "").strip()
            
            self.logger.info(f"LLaMA chat completed in {duration:.2f}s")
            
            self.struct_logger.log_interaction(
                "chat_completion",
                {
                    "model": use_model,
                    "temperature": temperature,
                    "prompt_length": len(prompt),
                    "response_length": len(result),
                    "duration_seconds": duration,
                    "base_url": self.base_url
                }
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"LLaMA chat failed after {duration:.2f}s: {e}")
            self.struct_logger.log_error(e, {
                "model": use_model,
                "temperature": temperature,
                "prompt_length": len(prompt),
                "duration_seconds": duration,
                "base_url": self.base_url
            })
            raise


# === Grok Client (Mock/Example) ===
class GrokClient:
    def __init__(self, model: str = "grok-1") -> None:
        self.logger = get_logger("causalllm.llm_client.grok")
        self.struct_logger = get_structured_logger("llm_client_grok")
        
        self.api_key = os.getenv("GROK_API_KEY")
        if not self.api_key:
            self.logger.error("GROK_API_KEY must be set")
            raise ValueError("GROK_API_KEY must be set.")
        self.default_model = model
        
        self.logger.info(f"Initializing Grok client with model: {model}")
        self.struct_logger.log_interaction(
            "client_initialization",
            {"default_model": model, "has_api_key": bool(self.api_key)}
        )

    def chat(self, prompt: str, model: str = "", temperature: float = 0.7) -> str:
        use_model = model or self.default_model
        
        self.logger.info(f"Making Grok chat request with model: {use_model}")
        self.logger.debug(f"Prompt length: {len(prompt)}, Temperature: {temperature}")
        
        start_time = time.time()
        try:
            # Placeholder: replace with actual Grok API call when available
            result = f"[Grok-{use_model}]: {prompt} (simulated)"
            duration = time.time() - start_time
            
            self.logger.warning("Using simulated Grok response")
            
            self.struct_logger.log_interaction(
                "chat_completion",
                {
                    "model": use_model,
                    "temperature": temperature,
                    "prompt_length": len(prompt),
                    "response_length": len(result),
                    "duration_seconds": duration,
                    "simulated": True
                }
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Grok chat failed after {duration:.2f}s: {e}")
            self.struct_logger.log_error(e, {
                "model": use_model,
                "temperature": temperature,
                "prompt_length": len(prompt),
                "duration_seconds": duration
            })
            raise


# === Factory Function ===
def get_llm_client(provider: str = "openai", model: str = "") -> BaseLLMClient:
    logger = get_logger("causalllm.llm_client.factory")
    struct_logger = get_structured_logger("llm_client_factory")
    
    provider = provider.lower()
    
    logger.info(f"Creating LLM client for provider: {provider}, model: {model}")
    
    try:
        if provider == "openai":
            client = OpenAIClient(model=model)
        elif provider == "llama":
            client = LLaMAClient(model=model)
        elif provider == "grok":
            client = GrokClient(model=model)
        else:
            logger.error(f"Unsupported provider: {provider}")
            raise ValueError(f"Unsupported provider: {provider}")
        
        struct_logger.log_interaction(
            "client_creation",
            {
                "provider": provider,
                "model": model,
                "client_type": type(client).__name__
            }
        )
        
        logger.info(f"Successfully created {type(client).__name__}")
        return client
        
    except Exception as e:
        logger.error(f"Failed to create LLM client: {e}")
        struct_logger.log_error(e, {"provider": provider, "model": model})
        raise
