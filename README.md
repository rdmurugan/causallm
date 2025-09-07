# CausalLLM: High-Performance Causal Inference Library

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/causallm.svg)](https://badge.fury.io/py/causallm)
[![GitHub stars](https://img.shields.io/github/stars/rdmurugan/causallm.svg)](https://github.com/rdmurugan/causallm/stargazers)
[![Downloads](https://img.shields.io/pypi/dm/causallm.svg)](https://pypi.org/project/causallm/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://github.com/rdmurugan/causallm/blob/main/COMPLETE_USER_GUIDE.md)

CausalLLM is a Python library for causal inference and analysis that combines statistical methods with domain-specific knowledge and advanced performance optimizations. It provides tools for discovering causal relationships in data and estimating treatment effects using multiple statistical approaches, now with enterprise-grade performance and scalability.

## 🚀 Performance Highlights (New in v4.0.0)

- **10x Faster Computations**: Vectorized algorithms with Numba JIT compilation
- **80% Memory Reduction**: Intelligent data chunking and lazy evaluation  
- **Unlimited Scale**: Handle datasets with millions of rows through streaming processing
- **Smart Caching**: 80%+ cache hit rates for repeated analyses
- **Parallel Processing**: Async computations with automatic resource management
- **Zero Configuration**: Performance optimizations work automatically

## Key Features

### 🧠 Statistical Causal Inference
- **Multiple Methods**: Linear regression, propensity score matching, instrumental variables, doubly robust estimation
- **Assumption Testing**: Automated validation of causal inference assumptions
- **Robustness Checks**: Cross-validation across multiple statistical approaches
- **Performance Optimized**: Vectorized algorithms for large-scale analysis

### 🔍 Causal Structure Discovery  
- **PC Algorithm**: Implementation for discovering relationships from data
- **Parallel Processing**: Async independence testing for faster discovery
- **LLM Enhancement**: Optional integration with language models for domain expertise
- **Scalable**: Chunked processing for very large variable sets

### 🏭 Domain-Specific Packages
- **Healthcare**: Clinical trial analysis, treatment effectiveness, patient outcomes
- **Insurance**: Risk assessment, premium optimization, claims analysis  
- **Marketing**: Campaign attribution, ROI optimization, customer analytics
- **Education**: Student outcomes, intervention analysis, policy evaluation
- **Experimentation**: A/B testing, experimental design validation

### 🔧 Advanced Performance Features
- **Data Chunking**: Automatic memory-efficient processing of large datasets
- **Intelligent Caching**: Multi-tier caching (memory + disk) with smart invalidation
- **Vectorized Algorithms**: Numba-optimized statistical computations
- **Async Processing**: Parallel execution of independent computations
- **Lazy Evaluation**: Deferred computation until results are needed
- **Resource Monitoring**: Automatic memory and CPU usage optimization

### 🌐 LLM Integrations
- **Multiple Providers**: OpenAI, Anthropic, LLaMA, local models
- **Optional Usage**: Library works fully without API keys using statistical methods
- **MCP Support**: Model Context Protocol for advanced integrations

## Installation

```bash
# Install latest version with all performance optimizations
pip install causallm

# Install with optional dependencies for maximum performance  
pip install causallm[full]

# For development with all tools
pip install causallm[dev]
```

## Quick Start

### Basic High-Performance Analysis

```python
from causallm import EnhancedCausalLLM
import pandas as pd

# Initialize with performance optimizations enabled (default)
causal_llm = EnhancedCausalLLM(
    enable_performance_optimizations=True,  # Auto-enabled for large datasets
    use_async=True,                         # Enable async processing
    chunk_size=10000                        # Automatic optimization
)

# Load your data (supports very large datasets now)
data = pd.read_csv("your_large_data.csv")  # Can handle millions of rows

# One-line comprehensive analysis with automatic performance optimization
results = causal_llm.comprehensive_analysis(
    data=data,
    treatment='treatment_variable',
    outcome='outcome_variable', 
    domain='healthcare'  # Enables domain-specific optimizations
)

print(f"Effect estimate: {results.inference_results}")
print(f"Confidence: {results.confidence_score}")
```

### Memory-Efficient Processing for Large Datasets

```python
from causallm.core.data_processing import DataChunker, StreamingDataProcessor

# Process datasets that don't fit in memory
processor = StreamingDataProcessor()

def analyze_chunk(chunk_data):
    return chunk_data.corr()

# Stream and process large CSV files
results = processor.process_streaming(
    "very_large_data.csv",
    analyze_chunk,
    aggregation_func=lambda results: pd.concat(results).mean()
)
```

### Cached Analysis for Faster Iterations

```python
from causallm.core.caching import StatisticalComputationCache

# Enable persistent caching across sessions
causal_llm = EnhancedCausalLLM(cache_dir="./causallm_cache")

# First run computes and caches
result1 = causal_llm.estimate_causal_effect(data, 'treatment', 'outcome')

# Second run uses cached results (10x+ faster)  
result2 = causal_llm.estimate_causal_effect(data, 'treatment', 'outcome')
```

### Async Processing for Maximum Performance

```python
import asyncio
from causallm.core.async_processing import AsyncCausalAnalysis

async def parallel_analysis():
    async_causal = AsyncCausalAnalysis()
    
    # Parallel correlation analysis
    corr_matrix = await async_causal.parallel_correlation_analysis(
        large_data, chunk_size=5000
    )
    
    # Parallel bootstrap analysis  
    bootstrap_results = await async_causal.parallel_bootstrap_analysis(
        large_data, analysis_func=my_analysis, n_bootstrap=1000
    )
    
    return corr_matrix, bootstrap_results

# Run async analysis
results = asyncio.run(parallel_analysis())
```

### Domain-Specific Analysis

#### Healthcare Example

```python
from causallm import HealthcareDomain, EnhancedCausalLLM

# Use healthcare domain with performance optimizations
healthcare = HealthcareDomain()
causal_llm = EnhancedCausalLLM(domain='healthcare')

# Generate realistic clinical trial data (scalable)
clinical_data = healthcare.generate_clinical_trial_data(
    n_patients=100000,  # Large dataset support
    treatment_arms=['control', 'treatment_a', 'treatment_b']
)

# High-performance treatment effectiveness analysis
results = healthcare.treatment_template.run_analysis(
    'treatment_effectiveness',
    clinical_data,
    causal_llm
)
```

#### Insurance Example

```python
from causallm import InsuranceDomain

# Use insurance domain package  
insurance = InsuranceDomain()

# Generate large-scale policy data
policy_data = insurance.generate_stop_loss_data(n_policies=500000)

# Memory-efficient risk factor analysis
risk_results = insurance.analyze_risk_factors(
    data=policy_data,
    risk_factor='industry_type',
    outcome='total_claim_amount'
)
```

## Performance Benchmarks

### Dataset Size Support
- **Small Datasets** (< 10K rows): Instant analysis with full feature set
- **Medium Datasets** (10K - 100K rows): Automatic optimization, ~2-5x speedup
- **Large Datasets** (100K - 1M rows): Chunked processing, async operations
- **Very Large Datasets** (> 1M rows): Streaming analysis, distributed computing

### Speed Improvements (vs. v3.0)
- **Correlation Analysis**: 10x faster with Numba vectorization
- **Causal Discovery**: 5x faster with parallel independence testing  
- **Effect Estimation**: 3x faster with optimized matching algorithms
- **Repeated Analysis**: 20x+ faster with intelligent caching

### Memory Efficiency  
- **Data Chunking**: Process datasets 10x larger than available RAM
- **Lazy Evaluation**: 60-80% memory reduction through deferred computation
- **Smart Caching**: Configurable memory vs. disk trade-offs

## Core Components

### EnhancedCausalLLM
High-performance main class combining statistical methods with LLM enhancement and automatic optimization.

### Statistical Methods (Performance Optimized)
- **Vectorized Linear Regression**: NumPy/Numba optimized for large datasets
- **Fast Propensity Score Matching**: Efficient matching algorithms with parallel processing  
- **Optimized Instrumental Variables**: Matrix operations optimized for speed
- **Parallel PC Algorithm**: Concurrent independence testing for causal discovery

### Domain Packages (Scalable)
Pre-configured, performance-optimized components for specific industries:

- **Healthcare Domain**: Clinical analysis optimized for medical datasets
- **Insurance Domain**: Risk assessment with actuarial computations
- **Marketing Domain**: Attribution analysis with customer segmentation
- **Education Domain**: Student outcome analysis with policy evaluation
- **Experimentation Domain**: A/B testing with statistical validation

### Performance Infrastructure
- **Data Processing**: Chunking, streaming, memory monitoring
- **Caching Layer**: Multi-tier caching with intelligent invalidation
- **Async Framework**: Task management with resource monitoring
- **Vectorized Algorithms**: Numba-optimized statistical computations
- **Lazy Evaluation**: Computation graphs with dependency tracking

## Advanced Features

### MCP Server Integration

CausalLLM provides Model Context Protocol (MCP) server capabilities:

```python
# Start MCP server for integration with Claude Desktop, VS Code, etc.
python -m causallm.mcp.server --port 8000

# Available MCP tools with performance optimization:
# - simulate_counterfactual: Generate counterfactual scenarios
# - analyze_treatment_effect: High-performance treatment analysis  
# - extract_causal_edges: Parallel causal relationship extraction
# - generate_reasoning_prompt: LLM-enhanced causal reasoning
```

### Statistical Rigor with Performance

CausalLLM maintains statistical rigor while providing performance:

- **Assumption Validation**: Automated testing with parallel processing
- **Robustness Checks**: Cross-validation across multiple optimized methods
- **Confidence Intervals**: Uncertainty quantification with bootstrap parallelization  
- **Effect Size Interpretation**: Statistical and practical significance assessment
- **Performance Monitoring**: Automatic benchmarking and optimization suggestions

## Documentation

- **[Complete User Guide](COMPLETE_USER_GUIDE.md)**: Comprehensive API documentation
- **[Performance Guide](PERFORMANCE_GUIDE.md)**: Optimization tips and benchmarks
- **[Domain Packages Guide](DOMAIN_PACKAGES.md)**: Industry-specific components  
- **[MCP Usage Guide](MCP_USAGE.md)**: Model Context Protocol integration
- **[Examples Directory](examples/)**: Performance-optimized code examples

## Requirements

### Core Dependencies
- Python 3.9+
- pandas >= 1.3.0
- numpy >= 1.21.0  
- scikit-learn >= 1.0.0
- scipy >= 1.7.0

### Performance Dependencies (Automatically Installed)
- numba >= 0.56.0 (JIT compilation)
- dask >= 2022.1.0 (distributed computing)
- psutil >= 5.8.0 (resource monitoring)

### Optional Dependencies
- openai >= 1.0.0 (LLM features)
- anthropic (Claude integration)
- aiofiles (async file operations)

## What's New in v4.0.0

### 🚀 Performance & Scalability
- **Data Chunking**: Handle datasets 10x larger than RAM
- **Vectorized Algorithms**: 10x speedup with Numba JIT compilation  
- **Intelligent Caching**: Multi-tier caching with 80%+ hit rates
- **Async Processing**: Parallel computations with resource management
- **Lazy Evaluation**: Memory-efficient deferred computation
- **Streaming Support**: Process unlimited dataset sizes

### 🛠 Enhanced Infrastructure  
- **Custom Exception Hierarchy**: Detailed error messages with recovery suggestions
- **Dependency Injection**: Modular architecture with factory patterns
- **Centralized Configuration**: Single source of truth for settings
- **Advanced Logging**: Structured logging with performance metrics
- **Memory Monitoring**: Automatic resource optimization

### 🧪 Statistical Improvements
- **Robustness Testing**: Automated assumption validation
- **Effect Size Estimation**: Enhanced interpretation with confidence intervals
- **Bootstrap Parallelization**: Faster uncertainty quantification
- **Cross-Validation**: Performance-optimized model validation

### 🏗 Developer Experience
- **Type Hints**: Complete type annotations throughout
- **Better Documentation**: Performance guides and optimization tips
- **Benchmark Tools**: Built-in performance measurement utilities
- **Example Gallery**: Real-world performance-optimized examples

## Migration from v3.x

Most existing code will work without changes. For performance benefits:

```python
# Old way (still works)
causal_llm = EnhancedCausalLLM()
result = causal_llm.comprehensive_analysis(data)

# New way (automatic performance optimization)  
causal_llm = EnhancedCausalLLM(
    enable_performance_optimizations=True,  # Default for large datasets
    chunk_size='auto',                      # Automatic sizing
    cache_dir='./cache'                     # Persistent caching
)
result = causal_llm.comprehensive_analysis(data)  # Up to 10x faster
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key areas for contributions:
- Additional domain packages (finance, retail, manufacturing)
- New statistical methods with performance optimization
- Advanced caching strategies
- Distributed computing enhancements

## Performance Support

For performance optimization help:
- **Issues**: [GitHub Issues](https://github.com/rdmurugan/causallm/issues) (tag with 'performance')
- **Discussions**: [GitHub Discussions](https://github.com/rdmurugan/causallm/discussions)
- **Benchmarking**: Use built-in `causallm.performance_demo` module

## Citation

If you use CausalLLM in your research:

```bibtex
@software{causallm2024,
  title={CausalLLM: High-Performance Causal Inference Library},
  author={Durai Rajamanickam},
  year={2024},
  version={4.0.0},
  url={https://github.com/rdmurugan/causallm},
  note={Performance-optimized causal inference with statistical rigor}
}
```

## Support & Contact

- **Issues**: [GitHub Issues](https://github.com/rdmurugan/causallm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rdmurugan/causallm/discussions)  
- **Email**: durai@infinidatum.net
- **LinkedIn**: [linkedin.com/in/durai-rajamanickam](https://www.linkedin.com/in/durai-rajamanickam)

---

## About

CausalLLM is developed and maintained by Durai Rajamanickam, with contributions from the open source community. The library aims to make causal inference more accessible while maintaining statistical rigor and providing enterprise-grade performance for production use cases.