"""
Test configuration and fixtures for CausalLLM test suite
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from typing import Dict, Any
from unittest.mock import Mock, MagicMock


@pytest.fixture
def sample_data():
    """Generate sample data for testing causal discovery and inference."""
    np.random.seed(42)
    n = 1000
    
    # Create causal structure: X1 -> X2 -> X3, X1 -> X3, X4 independent
    X1 = np.random.normal(0, 1, n)
    X2 = 0.5 * X1 + np.random.normal(0, 0.5, n)
    X3 = 0.3 * X1 + 0.4 * X2 + np.random.normal(0, 0.3, n)
    X4 = np.random.normal(0, 1, n)
    
    return pd.DataFrame({
        'X1': X1,
        'X2': X2, 
        'X3': X3,
        'X4': X4
    })


@pytest.fixture
def simple_data():
    """Simple two-variable dataset for basic testing."""
    np.random.seed(123)
    n = 500
    
    X = np.random.normal(0, 1, n)
    Y = 0.7 * X + np.random.normal(0, 0.3, n)
    
    return pd.DataFrame({'X': X, 'Y': Y})


@pytest.fixture
def variable_descriptions():
    """Variable descriptions for testing."""
    return {
        'X1': 'Treatment variable representing intervention',
        'X2': 'Mediator variable affected by treatment', 
        'X3': 'Outcome variable of interest',
        'X4': 'Control variable independent of treatment'
    }


@pytest.fixture
def simple_variable_descriptions():
    """Simple variable descriptions."""
    return {
        'X': 'Predictor variable',
        'Y': 'Outcome variable'
    }


@pytest.fixture
def dag_edges():
    """Sample DAG edges for testing."""
    return [('X1', 'X2'), ('X2', 'X3'), ('X1', 'X3')]


@pytest.fixture
def simple_dag_edges():
    """Simple DAG edges."""
    return [('X', 'Y')]


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing without actual API calls."""
    mock_client = Mock()
    mock_client.chat.return_value = """
    [
        {
            "cause": "X1",
            "effect": "X2", 
            "confidence": "high",
            "reasoning": "X1 directly affects X2 based on domain knowledge"
        },
        {
            "cause": "X2",
            "effect": "X3",
            "confidence": "medium", 
            "reasoning": "X2 influences X3 through mediation pathway"
        }
    ]
    """
    return mock_client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()
    
    # Mock successful response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Test response from OpenAI"
    mock_response.usage.total_tokens = 100
    
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def temp_dir():
    """Temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def causal_context():
    """Sample causal context for testing."""
    return """
    This is a healthcare scenario where we are studying the relationship between
    a new treatment (X1), patient response (X2), and health outcomes (X3).
    We also have a control variable (X4) representing baseline health status.
    """


@pytest.fixture
def intervention_data():
    """Sample intervention data for do-calculus testing."""
    return {
        "treatment": "1",
        "dosage": "high"
    }


@pytest.fixture
def counterfactual_scenario():
    """Sample counterfactual scenario."""
    return {
        "factual": "Patient received treatment A and showed improvement",
        "intervention": "Patient received treatment B instead",
        "instruction": "Analyze the potential difference in outcomes"
    }


@pytest.fixture
def discovery_config():
    """Configuration for causal discovery tests."""
    return {
        "method": "hybrid",
        "significance_level": 0.05,
        "max_conditioning_size": 3,
        "bootstrap_samples": 50,
        "stability_threshold": 0.8
    }


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory for test data files."""
    import pathlib
    test_dir = pathlib.Path(__file__).parent / "data"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture
def large_sample_data():
    """Larger sample dataset for performance testing."""
    np.random.seed(2024)
    n = 5000
    
    # Complex causal structure
    Z1 = np.random.normal(0, 1, n)  # Confounder
    Z2 = np.random.normal(0, 1, n)  # Confounder
    
    X = 0.3 * Z1 + 0.2 * Z2 + np.random.normal(0, 0.5, n)  # Treatment
    M = 0.4 * X + 0.3 * Z1 + np.random.normal(0, 0.4, n)   # Mediator
    Y = 0.5 * X + 0.6 * M + 0.2 * Z2 + np.random.normal(0, 0.3, n)  # Outcome
    
    return pd.DataFrame({
        'Z1': Z1, 'Z2': Z2, 'X': X, 'M': M, 'Y': Y
    })


@pytest.fixture
def time_series_data():
    """Time series data for Granger causality testing."""
    np.random.seed(456)
    n = 200
    
    # Create time series with causal relationship
    X = np.zeros(n)
    Y = np.zeros(n)
    
    for t in range(1, n):
        X[t] = 0.6 * X[t-1] + np.random.normal(0, 0.3)
        Y[t] = 0.4 * Y[t-1] + 0.3 * X[t-1] + np.random.normal(0, 0.25)
    
    return pd.DataFrame({
        'X': X,
        'Y': Y,
        'time': range(n)
    })


@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test for reproducibility."""
    np.random.seed(42)


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "llm: marks tests that require LLM API access"
    )
    config.addinivalue_line(
        "markers", "statistical: marks tests for statistical methods"
    )
    config.addinivalue_line(
        "markers", "visualization: marks tests for visualization features"
    )


# Environment setup helpers
def skip_if_no_openai():
    """Skip test if OpenAI credentials are not available."""
    return pytest.mark.skipif(
        not (os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_PROJECT_ID")),
        reason="OpenAI credentials not available"
    )


def skip_if_no_llama():
    """Skip test if LLaMA API is not available."""
    return pytest.mark.skipif(
        not os.getenv("LLAMA_API_URL"),
        reason="LLaMA API URL not configured"
    )


@pytest.fixture
def mock_statistical_test_results():
    """Mock results for statistical tests."""
    return {
        'partial_correlation': {'statistic': 2.5, 'p_value': 0.01},
        'mutual_information': {'statistic': 0.8, 'p_value': 0.03},
        'chi_square': {'statistic': 15.2, 'p_value': 0.001}
    }


@pytest.fixture
def sample_graph_edges():
    """Sample causal graph edges for testing."""
    return [
        {'source': 'income', 'target': 'education'},
        {'source': 'education', 'target': 'job_satisfaction'},
        {'source': 'income', 'target': 'job_satisfaction'},
        {'source': 'age', 'target': 'income'}
    ]