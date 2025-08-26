"""
Test suite for LLM clients
Tests OpenAI, LLaMA, Grok, MCP clients and factory functions
"""
import pytest
import os
import time
from unittest.mock import Mock, patch, MagicMock
from causallm.core.llm_client import (
    OpenAIClient,
    LLaMAClient,
    GrokClient,
    MCPClient,
    get_llm_client
)


class TestOpenAIClient:
    """Test OpenAI client implementation."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_PROJECT_ID": "test-project"})
    @patch('causallm.core.llm_client.OpenAI')
    def test_openai_initialization_success(self, mock_openai_class):
        """Test successful OpenAI client initialization."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        client = OpenAIClient(model="gpt-4")
        
        assert client.default_model == "gpt-4"
        assert client.client is mock_client
        mock_openai_class.assert_called_once_with(api_key="test-key", project="test-project")
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "your-default-api-key"})
    def test_openai_initialization_invalid_key(self):
        """Test OpenAI initialization with invalid API key."""
        with pytest.raises(ValueError, match="must be set to valid values"):
            OpenAIClient()
    
    @patch.dict(os.environ, {}, clear=True)
    def test_openai_initialization_missing_env_vars(self):
        """Test OpenAI initialization with missing environment variables."""
        with pytest.raises(ValueError, match="must be set to valid values"):
            OpenAIClient()
    
    @patch('causallm.core.llm_client.OpenAI')
    def test_openai_missing_package(self, mock_openai_class):
        """Test OpenAI client when package is not installed."""
        mock_openai_class.side_effect = ImportError("OpenAI package not found")
        
        with pytest.raises(ImportError, match="OpenAI package is required"):
            OpenAIClient()
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_PROJECT_ID": "test-project"})
    def test_openai_chat_success(self, mock_openai_client):
        """Test successful OpenAI chat completion."""
        with patch('causallm.core.llm_client.OpenAI') as mock_openai_class:
            mock_openai_class.return_value = mock_openai_client
            
            client = OpenAIClient()
            response = client.chat("Test prompt", temperature=0.5)
            
            assert response == "Test response from OpenAI"
            mock_openai_client.chat.completions.create.assert_called_once()
            
            call_args = mock_openai_client.chat.completions.create.call_args
            assert call_args[1]["model"] == "gpt-4"
            assert call_args[1]["temperature"] == 0.5
            assert call_args[1]["messages"][0]["content"] == "Test prompt"
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_PROJECT_ID": "test-project"})
    def test_openai_chat_empty_prompt(self):
        """Test OpenAI chat with empty prompt."""
        with patch('causallm.core.llm_client.OpenAI'):
            client = OpenAIClient()
            
            with pytest.raises(ValueError, match="Prompt cannot be empty"):
                client.chat("")
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_PROJECT_ID": "test-project"})
    def test_openai_chat_invalid_temperature(self):
        """Test OpenAI chat with invalid temperature."""
        with patch('causallm.core.llm_client.OpenAI'):
            client = OpenAIClient()
            
            with pytest.raises(ValueError, match="Temperature must be between"):
                client.chat("Test", temperature=3.0)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_PROJECT_ID": "test-project"})
    def test_openai_chat_empty_response(self):
        """Test handling of empty response from OpenAI."""
        with patch('causallm.core.llm_client.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value.choices = []
            mock_openai_class.return_value = mock_client
            
            client = OpenAIClient()
            
            with pytest.raises(RuntimeError, match="empty response"):
                client.chat("Test prompt")
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_PROJECT_ID": "test-project"})
    def test_openai_chat_rate_limit_error(self):
        """Test handling of rate limit error."""
        with patch('causallm.core.llm_client.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("Rate limit exceeded")
            mock_openai_class.return_value = mock_client
            
            client = OpenAIClient()
            
            with pytest.raises(RuntimeError, match="rate limit exceeded"):
                client.chat("Test prompt")
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_PROJECT_ID": "test-project"})
    def test_openai_chat_authentication_error(self):
        """Test handling of authentication error."""
        with patch('causallm.core.llm_client.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("Authentication failed")
            mock_openai_class.return_value = mock_client
            
            client = OpenAIClient()
            
            with pytest.raises(RuntimeError, match="authentication failed"):
                client.chat("Test prompt")


class TestLLaMAClient:
    """Test LLaMA client implementation."""
    
    @patch('causallm.core.llm_client.requests')
    def test_llama_initialization_success(self, mock_requests):
        """Test successful LLaMA client initialization."""
        client = LLaMAClient(model="llama3")
        
        assert client.default_model == "llama3"
        assert client.base_url == "http://localhost:11434"
        assert client.requests is mock_requests
    
    @patch.dict(os.environ, {"LLAMA_API_URL": "http://custom:8080"})
    @patch('causallm.core.llm_client.requests')
    def test_llama_custom_url(self, mock_requests):
        """Test LLaMA client with custom URL."""
        client = LLaMAClient()
        assert client.base_url == "http://custom:8080"
    
    def test_llama_missing_requests_package(self):
        """Test LLaMA client when requests package is missing."""
        with patch('causallm.core.llm_client.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("requests not found")
            
            with pytest.raises(ImportError, match="Requests package is required"):
                LLaMAClient()
    
    @patch('causallm.core.llm_client.requests')
    def test_llama_chat_success(self, mock_requests):
        """Test successful LLaMA chat."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {"response": "LLaMA response"}
        mock_response.raise_for_status.return_value = None
        mock_requests.post.return_value = mock_response
        
        client = LLaMAClient()
        response = client.chat("Test prompt")
        
        assert response == "LLaMA response"
        mock_requests.post.assert_called_once()
        
        call_args = mock_requests.post.call_args
        assert "api/generate" in call_args[0][0]
        assert call_args[1]["json"]["prompt"] == "Test prompt"
    
    @patch('causallm.core.llm_client.requests')
    def test_llama_chat_empty_prompt(self, mock_requests):
        """Test LLaMA chat with empty prompt."""
        client = LLaMAClient()
        
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            client.chat("")
    
    @patch('causallm.core.llm_client.requests')
    def test_llama_chat_timeout(self, mock_requests):
        """Test LLaMA chat timeout handling."""
        mock_requests.post.side_effect = mock_requests.exceptions.Timeout("Timeout")
        
        client = LLaMAClient()
        
        with pytest.raises(RuntimeError, match="timed out"):
            client.chat("Test prompt")
    
    @patch('causallm.core.llm_client.requests')
    def test_llama_chat_connection_error(self, mock_requests):
        """Test LLaMA connection error handling."""
        mock_requests.post.side_effect = mock_requests.exceptions.ConnectionError("Connection failed")
        
        client = LLaMAClient()
        
        with pytest.raises(RuntimeError, match="Cannot connect"):
            client.chat("Test prompt")
    
    @patch('causallm.core.llm_client.requests')
    def test_llama_chat_http_error(self, mock_requests):
        """Test LLaMA HTTP error handling."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = mock_requests.exceptions.HTTPError("404 Not Found")
        mock_response.status_code = 404
        mock_requests.post.return_value = mock_response
        
        client = LLaMAClient()
        
        with pytest.raises(RuntimeError, match="HTTP error 404"):
            client.chat("Test prompt")
    
    @patch('causallm.core.llm_client.requests')
    def test_llama_chat_json_error(self, mock_requests):
        """Test LLaMA invalid JSON response."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_requests.post.return_value = mock_response
        
        client = LLaMAClient()
        
        with pytest.raises(RuntimeError, match="invalid JSON"):
            client.chat("Test prompt")
    
    @patch('causallm.core.llm_client.requests')
    def test_llama_chat_api_error(self, mock_requests):
        """Test LLaMA API error in response."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"error": "Model not found"}
        mock_requests.post.return_value = mock_response
        
        client = LLaMAClient()
        
        with pytest.raises(RuntimeError, match="LLaMA API error: Model not found"):
            client.chat("Test prompt")
    
    @patch('causallm.core.llm_client.requests')
    def test_llama_chat_empty_response(self, mock_requests):
        """Test LLaMA empty response handling."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"response": ""}
        mock_requests.post.return_value = mock_response
        
        client = LLaMAClient()
        response = client.chat("Test prompt")
        
        assert response == "[Empty response from LLaMA]"


class TestGrokClient:
    """Test Grok client implementation (simulated)."""
    
    def test_grok_initialization(self):
        """Test Grok client initialization."""
        client = GrokClient(model="grok-2")
        assert client.default_model == "grok-2"
        assert client.api_key == "dummy-key"
    
    @patch.dict(os.environ, {"GROK_API_KEY": "real-key"})
    def test_grok_initialization_real_key(self):
        """Test Grok initialization with real API key."""
        client = GrokClient()
        assert client.api_key == "real-key"
    
    def test_grok_chat_simulation(self):
        """Test Grok simulated chat."""
        client = GrokClient()
        response = client.chat("Test prompt")
        
        assert response.startswith("[Grok-grok-1]:")
        assert "Test prompt" in response
        assert "(simulated)" in response
    
    def test_grok_chat_empty_prompt(self):
        """Test Grok chat with empty prompt."""
        client = GrokClient()
        
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            client.chat("")
    
    def test_grok_chat_invalid_temperature(self):
        """Test Grok chat with invalid temperature."""
        client = GrokClient()
        
        with pytest.raises(ValueError, match="Temperature must be between"):
            client.chat("Test", temperature=-1)


class TestMCPClient:
    """Test MCP client implementation."""
    
    @patch('causallm.core.llm_client.load_mcp_config')
    @patch('causallm.core.llm_client.MCPLLMClient')
    def test_mcp_initialization_success(self, mock_mcp_client_class, mock_load_config):
        """Test successful MCP client initialization."""
        mock_config = Mock()
        mock_config.client.transport = "tcp"
        mock_config.client.server_name = "causal-server"
        mock_load_config.return_value = mock_config
        
        mock_mcp_client = Mock()
        mock_mcp_client_class.return_value = mock_mcp_client
        
        client = MCPClient(model="counterfactual")
        
        assert client.default_model == "counterfactual"
        assert client.config is mock_config
        assert client.mcp_client is mock_mcp_client
        assert client._connected is False
    
    def test_mcp_initialization_missing_module(self):
        """Test MCP initialization when module is missing."""
        with patch('causallm.core.llm_client.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("MCP module not found")
            
            with pytest.raises(ImportError, match="MCP module is required"):
                MCPClient()
    
    @patch('causallm.core.llm_client.load_mcp_config')
    @patch('causallm.core.llm_client.MCPLLMClient')
    def test_mcp_ensure_connected(self, mock_mcp_client_class, mock_load_config):
        """Test MCP connection establishment."""
        mock_config = Mock()
        mock_config.client.transport = "tcp"
        mock_config.client.server_name = "causal-server"
        mock_load_config.return_value = mock_config
        
        mock_mcp_client = Mock()
        mock_mcp_client.connect = Mock(return_value=None)
        mock_mcp_client_class.return_value = mock_mcp_client
        
        client = MCPClient()
        
        with patch('asyncio.run') as mock_run:
            client._ensure_connected()
            
            mock_run.assert_called_once()
            assert client._connected is True
    
    @patch('causallm.core.llm_client.load_mcp_config')
    @patch('causallm.core.llm_client.MCPLLMClient')
    def test_mcp_chat_counterfactual(self, mock_mcp_client_class, mock_load_config):
        """Test MCP chat with counterfactual model."""
        mock_config = Mock()
        mock_config.client.server_name = "causal-server"
        mock_load_config.return_value = mock_config
        
        mock_mcp_client = Mock()
        mock_mcp_client.simulate_counterfactual = Mock(return_value="Counterfactual result")
        mock_mcp_client_class.return_value = mock_mcp_client
        
        client = MCPClient()
        client._connected = True  # Skip connection
        
        with patch('asyncio.run', return_value="Counterfactual result"):
            response = client.chat("Test scenario", model="counterfactual")
            
            assert response == "Counterfactual result"
    
    @patch('causallm.core.llm_client.load_mcp_config')
    @patch('causallm.core.llm_client.MCPLLMClient')
    def test_mcp_chat_do_prompt(self, mock_mcp_client_class, mock_load_config):
        """Test MCP chat with do-calculus model."""
        mock_config = Mock()
        mock_config.client.server_name = "causal-server"
        mock_load_config.return_value = mock_config
        
        mock_mcp_client = Mock()
        mock_mcp_client_class.return_value = mock_mcp_client
        
        client = MCPClient()
        client._connected = True
        
        with patch('asyncio.run', return_value="Do-calculus result"):
            response = client.chat("Test context", model="do_prompt")
            
            assert response == "Do-calculus result"
    
    @patch('causallm.core.llm_client.load_mcp_config')
    @patch('causallm.core.llm_client.MCPLLMClient')
    def test_mcp_chat_causal_edges(self, mock_mcp_client_class, mock_load_config):
        """Test MCP chat with causal edges extraction."""
        mock_config = Mock()
        mock_config.client.server_name = "causal-server"
        mock_load_config.return_value = mock_config
        
        mock_mcp_client = Mock()
        mock_mcp_client_class.return_value = mock_mcp_client
        
        client = MCPClient()
        client._connected = True
        
        with patch('asyncio.run', return_value=[("A", "B"), ("B", "C")]):
            response = client.chat("Extract edges", model="causal_edges")
            
            assert "Extracted causal edges:" in response
            assert "('A', 'B')" in response
    
    @patch('causallm.core.llm_client.load_mcp_config')
    @patch('causallm.core.llm_client.MCPLLMClient')
    def test_mcp_chat_empty_response(self, mock_mcp_client_class, mock_load_config):
        """Test MCP chat with empty response."""
        mock_config = Mock()
        mock_config.client.server_name = "causal-server"
        mock_load_config.return_value = mock_config
        
        mock_mcp_client = Mock()
        mock_mcp_client_class.return_value = mock_mcp_client
        
        client = MCPClient()
        client._connected = True
        
        with patch('asyncio.run', return_value=""):
            response = client.chat("Test", model="counterfactual")
            
            assert response == "[Empty response from MCP server]"
    
    @patch('causallm.core.llm_client.load_mcp_config')
    @patch('causallm.core.llm_client.MCPLLMClient')
    def test_mcp_chat_connection_error(self, mock_mcp_client_class, mock_load_config):
        """Test MCP chat with connection error."""
        mock_config = Mock()
        mock_config.client.server_name = "causal-server"
        mock_load_config.return_value = mock_config
        
        mock_mcp_client = Mock()
        mock_mcp_client_class.return_value = mock_mcp_client
        
        client = MCPClient()
        client._connected = True
        
        with patch('asyncio.run', side_effect=RuntimeError("Connection failed")):
            with pytest.raises(RuntimeError, match="MCP tool call failed"):
                client.chat("Test", model="counterfactual")


class TestLLMClientFactory:
    """Test LLM client factory function."""
    
    @patch('causallm.core.llm_client.OpenAIClient')
    def test_get_llm_client_openai(self, mock_openai_class):
        """Test factory creating OpenAI client."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        client = get_llm_client("openai", "gpt-4")
        
        assert client is mock_client
        mock_openai_class.assert_called_once_with(model="gpt-4")
    
    @patch('causallm.core.llm_client.LLaMAClient')
    def test_get_llm_client_llama(self, mock_llama_class):
        """Test factory creating LLaMA client."""
        mock_client = Mock()
        mock_llama_class.return_value = mock_client
        
        client = get_llm_client("llama", "llama3")
        
        assert client is mock_client
        mock_llama_class.assert_called_once_with(model="llama3")
    
    @patch('causallm.core.llm_client.GrokClient')
    def test_get_llm_client_grok(self, mock_grok_class):
        """Test factory creating Grok client."""
        mock_client = Mock()
        mock_grok_class.return_value = mock_client
        
        client = get_llm_client("grok", "grok-1")
        
        assert client is mock_client
        mock_grok_class.assert_called_once_with(model="grok-1")
    
    @patch('causallm.core.llm_client.MCPClient')
    def test_get_llm_client_mcp(self, mock_mcp_class):
        """Test factory creating MCP client."""
        mock_client = Mock()
        mock_mcp_class.return_value = mock_client
        
        client = get_llm_client("mcp", "counterfactual")
        
        assert client is mock_client
        mock_mcp_class.assert_called_once_with(model="counterfactual")
    
    def test_get_llm_client_unsupported(self):
        """Test factory with unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            get_llm_client("unsupported")
    
    def test_get_llm_client_empty_provider(self):
        """Test factory with empty provider."""
        with pytest.raises(ValueError, match="Provider cannot be empty"):
            get_llm_client("")
    
    def test_get_llm_client_case_insensitive(self):
        """Test factory with case-insensitive provider names."""
        with patch('causallm.core.llm_client.OpenAIClient') as mock_class:
            mock_class.return_value = Mock()
            
            # Should work with different cases
            get_llm_client("OpenAI")
            get_llm_client("OPENAI")
            get_llm_client("openai")
            
            assert mock_class.call_count == 3
    
    @patch('causallm.core.llm_client.OpenAIClient')
    def test_get_llm_client_creation_error(self, mock_openai_class):
        """Test factory handling client creation errors."""
        mock_openai_class.side_effect = RuntimeError("API key invalid")
        
        with pytest.raises(RuntimeError, match="Failed to create openai client"):
            get_llm_client("openai")


class TestLLMClientIntegration:
    """Test integration scenarios for LLM clients."""
    
    def test_client_protocol_compliance(self):
        """Test that all clients implement the required protocol."""
        from causallm.core.llm_client import BaseLLMClient
        
        # All client classes should have the chat method
        clients = [OpenAIClient, LLaMAClient, GrokClient, MCPClient]
        
        for client_class in clients:
            # Check that chat method exists (we can't instantiate without proper setup)
            assert hasattr(client_class, 'chat')
    
    @patch('causallm.core.llm_client.time.time')
    def test_timing_measurement(self, mock_time):
        """Test that clients measure execution time."""
        mock_time.side_effect = [0.0, 1.5]  # Start and end times
        
        client = GrokClient()  # Use Grok as it's simulated
        
        with patch('causallm.core.llm_client.time.sleep'):  # Skip actual sleep
            response = client.chat("Test timing")
        
        # Should have measured time (called time.time() twice)
        assert mock_time.call_count >= 2
    
    def test_temperature_validation(self):
        """Test temperature validation across clients."""
        clients = [
            GrokClient(),  # Use Grok as it doesn't require external services
        ]
        
        for client in clients:
            # Valid temperatures should work
            try:
                client.chat("Test", temperature=0.0)
                client.chat("Test", temperature=1.0)
                client.chat("Test", temperature=2.0)
            except ValueError:
                pytest.fail(f"Valid temperature rejected by {type(client).__name__}")
            
            # Invalid temperatures should raise errors
            with pytest.raises(ValueError):
                client.chat("Test", temperature=-0.1)
            
            with pytest.raises(ValueError):
                client.chat("Test", temperature=2.1)
    
    def test_prompt_validation(self):
        """Test prompt validation across clients."""
        client = GrokClient()  # Use Grok as it's simulated
        
        # Valid prompts should work
        client.chat("Valid prompt")
        client.chat("Another valid prompt with numbers 123")
        
        # Invalid prompts should raise errors
        with pytest.raises(ValueError):
            client.chat("")
        
        with pytest.raises(ValueError):
            client.chat("   ")  # Whitespace only
    
    def test_error_message_consistency(self):
        """Test that error messages are consistent across clients."""
        client = GrokClient()
        
        # Test empty prompt error
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            client.chat("")
        
        # Test temperature error
        with pytest.raises(ValueError, match="Temperature must be between"):
            client.chat("Test", temperature=3.0)
    
    @pytest.mark.slow
    def test_client_performance(self):
        """Test client performance characteristics."""
        client = GrokClient()  # Simulated, so should be fast
        
        start_time = time.time()
        response = client.chat("Performance test prompt")
        duration = time.time() - start_time
        
        # Simulated client should respond quickly (under 1 second)
        assert duration < 1.0
        assert isinstance(response, str)
        assert len(response) > 0