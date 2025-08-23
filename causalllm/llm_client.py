from typing import Protocol, Any
import os
import time
from causalllm.logging import get_logger, get_structured_logger

# Base interface for all LLM clients
class BaseLLMClient(Protocol):
    def chat(self, prompt: str, model: str = "", temperature: float = 0.7) -> str: ...


# OpenAI Client (Project API Key version)
class OpenAIClient:
    def __init__(self, model: str = "gpt-4") -> None:
        self.logger = get_logger("causalllm.llm_client.openai")
        self.struct_logger = get_structured_logger("llm_client_openai")
        
        self.logger.info("Initializing OpenAI client")
        
        try:
            from openai import OpenAI
        except ImportError as e:
            self.logger.error("OpenAI package not available")
            raise ImportError("OpenAI package is required but not installed. Run: pip install openai") from e

        api_key = os.getenv("OPENAI_API_KEY", "your-default-api-key")
        project_id = os.getenv("OPENAI_PROJECT_ID", "your-default-project-id")

        # Check for placeholder values
        if api_key in ("your-default-api-key", None, "") or project_id in ("your-default-project-id", None, ""):
            self.logger.error("Invalid OpenAI credentials - using default placeholders")
            raise ValueError("OPENAI_API_KEY and OPENAI_PROJECT_ID must be set to valid values, not placeholders.")

        try:
            self.client = OpenAI(api_key=api_key, project=project_id)
            self.default_model = model
            
            self.logger.info(f"OpenAI client initialized successfully with model: {model}")
            self.struct_logger.log_interaction(
                "client_initialization", 
                {"model": model, "has_project_id": bool(project_id)}
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.struct_logger.log_error(e, {"model": model})
            raise

    def chat(self, prompt: str, model: str = "", temperature: float = 0.7) -> str:
        use_model = model or self.default_model
        
        # Input validation
        if not prompt or not prompt.strip():
            self.logger.error("Empty prompt provided")
            raise ValueError("Prompt cannot be empty")
        
        if not (0.0 <= temperature <= 2.0):
            self.logger.error(f"Invalid temperature: {temperature}")
            raise ValueError("Temperature must be between 0.0 and 2.0")
        
        self.logger.info(f"Making OpenAI API call with model: {use_model}")
        self.logger.debug(f"Prompt length: {len(prompt)}, Temperature: {temperature}")
        
        start_time = time.time()
        try:
            response: Any = self.client.chat.completions.create(
                model=use_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            
            duration = time.time() - start_time
            
            # Handle potential empty response
            if not response.choices or not response.choices[0].message:
                self.logger.error("Empty response from OpenAI API")
                raise RuntimeError("Received empty response from OpenAI API")
            
            result = (response.choices[0].message.content or "").strip()
            
            self.logger.info(f"OpenAI API call completed in {duration:.2f}s")
            
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
            error_type = type(e).__name__
            
            self.logger.error(f"OpenAI API call failed after {duration:.2f}s: {error_type}: {e}")
            
            self.struct_logger.log_error(e, {
                "model": use_model,
                "temperature": temperature,
                "prompt_length": len(prompt),
                "duration_seconds": duration,
                "error_type": error_type
            })
            
            # Re-raise with more context for different error types
            if "rate limit" in str(e).lower():
                raise RuntimeError(f"OpenAI API rate limit exceeded: {e}") from e
            elif "invalid" in str(e).lower() and "model" in str(e).lower():
                raise ValueError(f"Invalid model '{use_model}': {e}") from e
            elif "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
                raise RuntimeError(f"OpenAI API authentication failed: {e}") from e
            else:
                raise RuntimeError(f"OpenAI API call failed: {e}") from e


# Local LLaMA (example: running Ollama or similar)
class LLaMAClient:
    def __init__(self, model: str = "llama3") -> None:
        self.logger = get_logger("causalllm.llm_client.llama")
        self.struct_logger = get_structured_logger("llm_client_llama")
        
        self.logger.info("Initializing LLaMA client")
        
        try:
            import requests
            self.requests = requests
        except ImportError as e:
            self.logger.error("Requests package not available")
            raise ImportError("Requests package is required but not installed. Run: pip install requests") from e
            
        self.base_url = os.getenv("LLAMA_API_URL", "http://localhost:11434")
        self.default_model = model
        
        self.logger.info(f"LLaMA client initialized - URL: {self.base_url}, model: {model}")
        self.struct_logger.log_interaction(
            "client_initialization",
            {"base_url": self.base_url, "model": model}
        )

    def chat(self, prompt: str, model: str = "", temperature: float = 0.7) -> str:
        use_model = model or self.default_model
        
        # Input validation
        if not prompt or not prompt.strip():
            self.logger.error("Empty prompt provided")
            raise ValueError("Prompt cannot be empty")
        
        if not (0.0 <= temperature <= 2.0):
            self.logger.error(f"Invalid temperature: {temperature}")
            raise ValueError("Temperature must be between 0.0 and 2.0")
        
        self.logger.info(f"Making LLaMA API call with model: {use_model}")
        self.logger.debug(f"Prompt length: {len(prompt)}, Temperature: {temperature}")
        
        start_time = time.time()
        try:
            response = self.requests.post(
                f"{self.base_url}/api/generate",
                json={"model": use_model, "prompt": prompt, "temperature": temperature},
                timeout=60
            )
            
            duration = time.time() - start_time
            
            # Check for HTTP errors
            response.raise_for_status()
            
            try:
                result_data = response.json()
            except ValueError as e:
                self.logger.error("Invalid JSON response from LLaMA API")
                raise RuntimeError("LLaMA API returned invalid JSON response") from e
            
            # Check for API error in response
            if "error" in result_data:
                error_msg = result_data.get("error", "Unknown error")
                self.logger.error(f"LLaMA API error: {error_msg}")
                raise RuntimeError(f"LLaMA API error: {error_msg}")
            
            result = (result_data.get("response", "") or "").strip()
            
            if not result:
                self.logger.warning("Empty response from LLaMA API")
                result = "[Empty response from LLaMA]"
            
            self.logger.info(f"LLaMA API call completed in {duration:.2f}s")
            
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
            
        except self.requests.exceptions.Timeout as e:
            duration = time.time() - start_time
            self.logger.error(f"LLaMA API call timed out after {duration:.2f}s")
            self.struct_logger.log_error(e, {
                "model": use_model,
                "base_url": self.base_url,
                "duration_seconds": duration,
                "error_type": "timeout"
            })
            raise RuntimeError(f"LLaMA API call timed out after 60 seconds") from e
            
        except self.requests.exceptions.ConnectionError as e:
            duration = time.time() - start_time
            self.logger.error(f"Failed to connect to LLaMA API at {self.base_url}")
            self.struct_logger.log_error(e, {
                "model": use_model,
                "base_url": self.base_url,
                "duration_seconds": duration,
                "error_type": "connection_error"
            })
            raise RuntimeError(f"Cannot connect to LLaMA API at {self.base_url}. Is the service running?") from e
            
        except self.requests.exceptions.HTTPError as e:
            duration = time.time() - start_time
            self.logger.error(f"LLaMA API HTTP error: {response.status_code}")
            self.struct_logger.log_error(e, {
                "model": use_model,
                "base_url": self.base_url,
                "duration_seconds": duration,
                "status_code": response.status_code,
                "error_type": "http_error"
            })
            raise RuntimeError(f"LLaMA API HTTP error {response.status_code}: {e}") from e
            
        except Exception as e:
            duration = time.time() - start_time
            error_type = type(e).__name__
            
            self.logger.error(f"LLaMA API call failed after {duration:.2f}s: {error_type}: {e}")
            self.struct_logger.log_error(e, {
                "model": use_model,
                "base_url": self.base_url,
                "duration_seconds": duration,
                "error_type": error_type
            })
            raise RuntimeError(f"LLaMA API call failed: {e}") from e


# Grok Client (Mock/Example)
class GrokClient:
    def __init__(self, model: str = "grok-1") -> None:
        self.logger = get_logger("causalllm.llm_client.grok")
        self.struct_logger = get_structured_logger("llm_client_grok")
        
        self.logger.info("Initializing Grok client (simulated)")
        
        self.api_key = os.getenv("GROK_API_KEY", "dummy-key")
        if self.api_key in ("dummy-key", None, ""):
            self.logger.warning("Using dummy Grok API key - this is a simulated client")
        
        self.default_model = model
        
        self.struct_logger.log_interaction(
            "client_initialization",
            {"model": model, "simulated": True}
        )

    def chat(self, prompt: str, model: str = "", temperature: float = 0.7) -> str:
        use_model = model or self.default_model
        
        # Input validation
        if not prompt or not prompt.strip():
            self.logger.error("Empty prompt provided")
            raise ValueError("Prompt cannot be empty")
        
        if not (0.0 <= temperature <= 2.0):
            self.logger.error(f"Invalid temperature: {temperature}")
            raise ValueError("Temperature must be between 0.0 and 2.0")
        
        self.logger.info(f"Simulating Grok API call with model: {use_model}")
        self.logger.warning("This is a simulated response - not a real Grok API call")
        
        start_time = time.time()
        
        # Simulate processing time
        time.sleep(0.1)
        
        result = f"[Grok-{use_model}]: {prompt} (simulated)"
        duration = time.time() - start_time
        
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


# Factory function
def get_llm_client(provider: str = "openai", model: str = "") -> BaseLLMClient:
    logger = get_logger("causalllm.llm_client.factory")
    struct_logger = get_structured_logger("llm_client_factory")
    
    provider = provider.lower()
    
    logger.info(f"Creating LLM client for provider: {provider}, model: {model}")
    
    # Input validation
    if not provider:
        logger.error("Empty provider specified")
        raise ValueError("Provider cannot be empty")
    
    try:
        client: BaseLLMClient
        
        if provider == "openai":
            client = OpenAIClient(model=model)
        elif provider == "llama":
            client = LLaMAClient(model=model)
        elif provider == "grok":
            client = GrokClient(model=model)
        else:
            valid_providers = ["openai", "llama", "grok"]
            logger.error(f"Unsupported provider: {provider}. Valid options: {valid_providers}")
            raise ValueError(f"Unsupported provider: '{provider}'. Valid options: {valid_providers}")
        
        struct_logger.log_interaction(
            "client_creation",
            {
                "provider": provider,
                "model": model,
                "client_type": type(client).__name__,
                "success": True
            }
        )
        
        logger.info(f"Successfully created {type(client).__name__}")
        return client
        
    except Exception as e:
        logger.error(f"Failed to create LLM client: {e}")
        struct_logger.log_error(e, {"provider": provider, "model": model})
        
        # Re-raise with context
        if isinstance(e, (ValueError, ImportError)):
            raise  # These are expected user errors
        else:
            raise RuntimeError(f"Failed to create {provider} client: {e}") from e
