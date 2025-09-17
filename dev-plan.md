# dev-plan.md

## Repository Overview

CausalLLM is a high-performance Python library for causal inference that combines statistical methods with Large Language Model integration. It provides enterprise-grade performance with 10x faster computations and supports multiple domains (healthcare, marketing, insurance, education).

## Development Commands

### Testing
- **Run all tests**: `python run_tests.py` or `python run_tests.py --all`
- **Run specific test types**:
  - Basic functionality: `python run_tests.py --basic`
  - Statistical methods: `python run_tests.py --statistical` 
  - Causal discovery: `python run_tests.py --discovery`
  - LLM integration: `python run_tests.py --llm`
  - Integration tests: `python run_tests.py --integration`
- **Run with coverage**: `python run_tests.py --coverage`
- **Generate test report**: `python run_tests.py --report`
- **Validate documentation**: `python run_tests.py --validate`

### Testing with pytest directly
- `pytest tests/` - Run all tests
- `pytest tests/ -m "not slow and not llm"` - Skip slow and LLM tests
- `pytest tests/ --cov=causallm --cov-report=html` - Run with coverage
- `pytest tests/test_core_functionality.py -v` - Run specific test file

### Installation and Setup
- **Install for development**: `pip install -e .[dev]`
- **Install with testing dependencies**: `pip install -e .[testing]`
- **Install full version**: `pip install -e .[full]`

### Code Quality
- **Format code**: `black causallm/`
- **Lint code**: `flake8 causallm/`
- **Type checking**: `mypy causallm/`

### CLI Usage
- **Run CLI**: `causallm --help`
- **Launch web interface**: `causallm web --port 8080`
- **Discover relationships**: `causallm discover --data data.csv --variables "age,treatment,outcome"`
- **Estimate effects**: `causallm effect --data data.csv --treatment drug --outcome recovery`

## High-Level Architecture

### Core Components
- **causallm/core/**: Core causal inference functionality
  - `causal_llm_core.py`: Main engine for causal analysis
  - `causal_discovery.py`: Algorithms for discovering causal relationships  
  - `statistical_methods.py`: PC Algorithm and independence tests
  - `dag_parser.py`: Directed Acyclic Graph parsing and manipulation
  - `do_operator.py`: Do-calculus simulation and effect estimation
  - `llm_client.py`: LLM provider integration (OpenAI, Anthropic, etc.)

### Enhanced Features
- **enhanced_causallm.py**: Performance-optimized main interface with standardized parameters
- **core/statistical_inference.py**: Statistical causal inference methods
- **core/enhanced_causal_discovery.py**: Enhanced discovery algorithms with async processing

### Domain Packages
- **domains/**: Industry-specific implementations
  - `healthcare/`: Clinical trial analysis, treatment effectiveness
  - `marketing/`: Campaign attribution, ROI optimization  
  - `insurance/`: Risk assessment, premium optimization
  - `education/`: Student outcome analysis
  - `experimentation/`: A/B testing frameworks

### Monitoring & Testing
- **monitoring/**: Production monitoring and observability
  - `metrics.py`: Performance and usage metrics collection
  - `health.py`: System health checks and monitoring
  - `profiler.py`: Memory and execution profiling
- **testing/**: Advanced testing infrastructure
  - `property_based.py`: Property-based testing with Hypothesis
  - `benchmarks.py`: Performance benchmarking
  - `mutation.py`: Mutation testing for test quality

### Interfaces & CLI
- **cli.py**: Command-line interface for terminal-based analysis
- **web.py**: Streamlit-based web interface for GUI analysis
- **mcp/**: Model Context Protocol server integration

## Key Configuration

### Environment Variables
- `CAUSALLM_LLM_PROVIDER`: LLM provider (openai, anthropic)
- `CAUSALLM_USE_ASYNC`: Enable async processing (true/false)
- `CAUSALLM_CHUNK_SIZE`: Data chunk size for large datasets
- `CAUSALLM_CACHE_DIR`: Cache directory for performance optimization
- `OPENAI_API_KEY`: OpenAI API key (optional)
- `ANTHROPIC_API_KEY`: Anthropic API key (optional)

### Important Files
- `pyproject.toml`: Package configuration and dependencies
- `pytest.ini`: Test configuration with markers for different test types
- `requirements.txt`: Core dependencies
- `run_tests.py`: Comprehensive test runner script

## Development Guidelines

### Test Markers
Use pytest markers to categorize tests:
- `@pytest.mark.slow`: For long-running tests
- `@pytest.mark.llm`: For tests requiring LLM API access
- `@pytest.mark.integration`: For integration tests
- `@pytest.mark.statistical`: For statistical method tests
- `@pytest.mark.unit`: For unit tests

### Parameter Standardization
Use consistent parameter names across all components:
- `data`: Dataset parameter (not `df` or `dataset`)
- `treatment_variable`: Treatment variable name
- `outcome_variable`: Outcome variable name  
- `domain_context`: Domain context (healthcare, marketing, etc.)

### Performance Considerations
- Use async methods for large datasets (`use_async=True`)
- Enable caching for repeated analyses (`cache_dir` parameter)
- Use chunked processing for datasets > 100K rows
- Monitor memory usage with built-in profiling tools

### Module Structure
- Core algorithms in `causallm/core/`
- Domain-specific code in `causallm/domains/`
- Utilities and helpers in `causallm/core/utils/`
- Example scripts in `examples/`
- Comprehensive documentation in `docs/`
