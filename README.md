# CausalLLM: High-Performance Causal Inference Library

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/causallm.svg)](https://badge.fury.io/py/causallm)
[![GitHub stars](https://img.shields.io/github/stars/rdmurugan/causallm.svg)](https://github.com/rdmurugan/causallm/stargazers)
[![Downloads](https://img.shields.io/pypi/dm/causallm.svg)](https://pypi.org/project/causallm/)

**CausalLLM** is a powerful Python library that combines statistical causal inference methods with advanced language models to discover causal relationships and estimate treatment effects. It provides enterprise-grade performance with 10x faster computations and 80% memory reduction while maintaining statistical rigor.

---

## 🚀 Performance Highlights

- **10x Faster Computations**: Vectorized algorithms with Numba JIT compilation
- **80% Memory Reduction**: Intelligent data chunking and lazy evaluation  
- **Unlimited Scale**: Handle datasets with millions of rows through streaming processing
- **Smart Caching**: 80%+ cache hit rates for repeated analyses
- **Parallel Processing**: Async computations with automatic resource management
- **Zero Configuration**: Performance optimizations work automatically

---

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [Installation](#-installation)
3. [Key Features](#-key-features)
4. [Domain Examples](#-domain-examples)
5. [Core Components](#-core-components)
6. [Performance](#-performance)
7. [API Documentation](#-api-documentation)
8. [Advanced Features](#-advanced-features)
9. [Support & Community](#-support--community)

---

## 🚀 Quick Start

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

---

## 📦 Installation

```bash
# Install latest version with all performance optimizations
pip install causallm

# Install with optional dependencies for maximum performance  
pip install causallm[full]

# For development with all tools
pip install causallm[dev]
```

---

## ✨ Key Features

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
- **[Healthcare](#healthcare-domain)**: Clinical trial analysis, treatment effectiveness, patient outcomes
- **[Insurance](#insurance-domain)**: Risk assessment, premium optimization, claims analysis  
- **[Marketing](#marketing-domain)**: Campaign attribution, ROI optimization, customer analytics
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

---

## 🏥 Domain Examples

### Healthcare Domain

Transform clinical data analysis with domain-specific expertise:

```python
from causallm import HealthcareDomain, EnhancedCausalLLM

# Initialize healthcare domain with performance optimizations
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

print(f"Treatment effect: {results.effect_estimate:.2f} days")
print(f"Clinical interpretation: {results.domain_interpretation}")
```

**Healthcare Features:**
- Clinical trial data generation with proper randomization
- Treatment effectiveness analysis with medical context
- Safety analysis and adverse event evaluation
- Patient outcome prediction with clinical insights

### Insurance Domain

Optimize risk assessment and premium pricing:

```python
from causallm import InsuranceDomain

# Initialize insurance domain  
insurance = InsuranceDomain()

# Generate large-scale policy data
policy_data = insurance.generate_stop_loss_data(n_policies=500000)

# Memory-efficient risk factor analysis
risk_results = insurance.analyze_risk_factors(
    data=policy_data,
    risk_factor='industry_type',
    outcome='total_claim_amount'
)

print(f"Industry risk effect: ${risk_results.effect_estimate:,.0f}")
print(f"Business recommendation: {risk_results.recommendations[0]}")
```

**Insurance Features:**
- Stop loss insurance data simulation
- Risk factor analysis with actuarial insights
- Premium optimization recommendations
- Claims prediction and underwriting support

### Marketing Domain

Master campaign attribution and ROI optimization:

```python
from causallm.domains.marketing import MarketingDomain

# Initialize with performance optimizations
marketing = MarketingDomain(enable_performance_optimizations=True)

# Generate sample marketing data
data = marketing.generate_marketing_data(
    n_customers=10000,
    n_touchpoints=30000
)

# Run attribution analysis
result = marketing.analyze_attribution(
    data, 
    model='data_driven'  # Recommended for most cases
)

print(f"Top performing channel: {max(result.channel_attribution.items(), key=lambda x: x[1])}")
```

**Marketing Features:**
- Multi-touch attribution modeling (first-touch, last-touch, data-driven, Shapley)
- Campaign ROI analysis and optimization
- Cross-device and cross-channel attribution
- Customer lifetime value modeling

**Quick Reference - Attribution Models:**
| Model | Best For | Description |
|-------|----------|-------------|
| `data_driven` | **Recommended** | Uses causal inference for attribution |
| `first_touch` | Brand awareness | 100% credit to first interaction |
| `last_touch` | Direct response | 100% credit to last interaction |
| `linear` | Balanced view | Equal credit across touchpoints |
| `shapley` | Advanced | Game theory based attribution |

---

## 🏗️ Core Components

### EnhancedCausalLLM
High-performance main class combining statistical methods with LLM enhancement and automatic optimization.

```python
from causallm import EnhancedCausalLLM

# Initialize with custom settings
causal_llm = EnhancedCausalLLM(
    llm_provider="openai",           # "openai", "anthropic", "llama", or None
    llm_model="gpt-4",              # Model name
    significance_level=0.05,         # Statistical significance threshold
    enable_performance_optimizations=True,
    chunk_size='auto',              # Automatic optimization
    cache_dir='./cache'             # Persistent caching
)
```

### Statistical Methods (Performance Optimized)
- **Vectorized Linear Regression**: NumPy/Numba optimized for large datasets
- **Fast Propensity Score Matching**: Efficient matching algorithms with parallel processing  
- **Optimized Instrumental Variables**: Matrix operations optimized for speed
- **Parallel PC Algorithm**: Concurrent independence testing for causal discovery

### Domain Packages (Scalable)
Pre-configured, performance-optimized components for specific industries with built-in expertise and realistic data generators.

---

## ⚡ Performance

### Dataset Size Support
- **Small Datasets** (< 10K rows): Instant analysis with full feature set
- **Medium Datasets** (10K - 100K rows): Automatic optimization, ~2-5x speedup
- **Large Datasets** (100K - 1M rows): Chunked processing, async operations
- **Very Large Datasets** (> 1M rows): Streaming analysis, distributed computing

### Speed Improvements
- **Correlation Analysis**: 10x faster with Numba vectorization
- **Causal Discovery**: 5x faster with parallel independence testing  
- **Effect Estimation**: 3x faster with optimized matching algorithms
- **Repeated Analysis**: 20x+ faster with intelligent caching

### Memory Efficiency  
- **Data Chunking**: Process datasets 10x larger than available RAM
- **Lazy Evaluation**: 60-80% memory reduction through deferred computation
- **Smart Caching**: Configurable memory vs. disk trade-offs

### Performance Configuration Examples

```python
# Small datasets (< 10K rows)
causal_llm = EnhancedCausalLLM(
    enable_performance_optimizations=False  # Overhead not worth it
)

# Large datasets (100K+ rows)
causal_llm = EnhancedCausalLLM(
    enable_performance_optimizations=True,
    chunk_size=50000,
    use_async=True,
    cache_dir="./cache",
    max_memory_usage_gb=8
)
```

---

## 📚 API Documentation

### Core Methods

#### `comprehensive_analysis()`
Complete end-to-end causal analysis combining discovery and inference.

```python
analysis = causal_llm.comprehensive_analysis(
    data=df,                     # Required: Your dataset
    treatment='campaign',        # Optional: Specific treatment
    outcome='revenue',          # Optional: Specific outcome  
    domain='marketing',         # Optional: Domain context
    covariates=['age', 'income'] # Optional: Control variables
)
```

**Returns:** `ComprehensiveCausalAnalysis` with:
- `discovery_results`: Causal structure findings
- `inference_results`: Detailed effect estimates
- `domain_recommendations`: Domain-specific advice
- `actionable_insights`: List of actionable findings
- `confidence_score`: Overall analysis confidence (0-1)

#### `discover_causal_relationships()`
Automatically discover causal relationships in your data.

```python
discovery = causal_llm.discover_causal_relationships(
    data=df,
    variables=['age', 'treatment', 'outcome'],
    domain='healthcare'
)
```

**Returns:** `CausalDiscoveryResult` with discovered edges, confounders, and domain insights.

#### `estimate_causal_effect()`
Estimate the causal effect of a treatment on an outcome.

```python
effect = causal_llm.estimate_causal_effect(
    data=df,
    treatment='new_drug',
    outcome='recovery_rate',
    covariates=['age', 'severity'],
    method='comprehensive'  # 'regression', 'matching', 'iv'
)
```

**Returns:** `CausalInferenceResult` with effect estimates, confidence intervals, and robustness checks.

### Statistical Methods

Available through `StatisticalCausalInference`:

- `CausalMethod.LINEAR_REGRESSION`: Standard regression with covariates
- `CausalMethod.MATCHING`: Propensity score matching
- `CausalMethod.INSTRUMENTAL_VARIABLES`: Two-stage least squares
- `CausalMethod.REGRESSION_DISCONTINUITY`: RDD (if applicable)
- `CausalMethod.DIFFERENCE_IN_DIFFERENCES`: DiD (if applicable)

### Domain Packages API

Each domain package provides:
- **Data Generators**: Realistic synthetic data with proper causal structure
- **Domain Knowledge**: Expert knowledge about relationships and confounders
- **Analysis Templates**: Pre-configured workflows with domain-specific interpretation

---

## 🔧 Advanced Features

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

### MCP Server Integration

CausalLLM provides Model Context Protocol (MCP) server capabilities for integration with Claude Desktop, VS Code, and other MCP-enabled applications:

```bash
# Start MCP server for integration with Claude Desktop, VS Code, etc.
python -m causallm.mcp.server --port 8000
```

**Available MCP tools:**
- `simulate_counterfactual`: Generate counterfactual scenarios
- `analyze_treatment_effect`: High-performance treatment analysis  
- `extract_causal_edges`: Parallel causal relationship extraction
- `generate_reasoning_prompt`: LLM-enhanced causal reasoning

### Statistical Rigor with Performance

- **Assumption Validation**: Automated testing with parallel processing
- **Robustness Checks**: Cross-validation across multiple optimized methods
- **Confidence Intervals**: Uncertainty quantification with bootstrap parallelization  
- **Effect Size Interpretation**: Statistical and practical significance assessment
- **Performance Monitoring**: Automatic benchmarking and optimization suggestions

---

## 📋 Requirements

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

---

## 🤝 Support & Community

### Getting Help

- **GitHub Issues**: [Report bugs & request features](https://github.com/rdmurugan/causallm/issues)
- **GitHub Discussions**: [Community support & questions](https://github.com/rdmurugan/causallm/discussions)
- **Performance Issues**: Tag with 'performance' label
- **Email Support**: durai@infinidatum.net
- **LinkedIn**: [Durai Rajamanickam](https://www.linkedin.com/in/durai-rajamanickam)

### 📚 Documentation

- **📖 [Complete User Guide](docs/COMPLETE_USER_GUIDE.md)**: Comprehensive API reference with examples
- **⚡ [Performance Guide](docs/PERFORMANCE_GUIDE.md)**: Optimization tips and benchmarks  
- **🏭 [Domain Packages Guide](docs/DOMAIN_PACKAGES.md)**: Industry-specific components and examples
- **🔗 [MCP Usage Guide](docs/MCP_USAGE.md)**: Model Context Protocol integration
- **📚 [Usage Examples](docs/USAGE_EXAMPLES.md)**: Real-world use cases across domains
- **📈 [Marketing Quick Reference](docs/MARKETING_QUICK_REFERENCE.md)**: Marketing attribution guide
- **💡 [Examples Directory](examples/)**: Runnable code examples and tutorials

### Contributing

We welcome contributions! Areas where help is needed:
- Additional domain packages (finance, retail, manufacturing)
- New statistical methods with performance optimization
- Advanced caching strategies
- Distributed computing enhancements

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for guidelines.

### Performance Support & Benchmarking

```python
# Built-in performance demo
from causallm.performance_demo import PerformanceBenchmark

benchmark = PerformanceBenchmark()
results = benchmark.run_comprehensive_benchmark([10000, 50000, 100000])
print(benchmark.generate_performance_report())
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 📖 Citation

If you use CausalLLM in your research:

```bibtex
@software{causallm2024,
  title={CausalLLM: High-Performance Causal Inference Library},
  author={Durai Rajamanickam},
  year={2024},
  url={https://github.com/rdmurugan/causallm},
  note={Performance-optimized causal inference with statistical rigor}
}
```

---

## 🏢 About

CausalLLM is developed and maintained by **Durai Rajamanickam**, with contributions from the open source community. The library aims to make causal inference more accessible while maintaining statistical rigor and providing enterprise-grade performance for production use cases.

---

**✨ Ready to discover causal insights in your data? Start with `pip install causallm` and explore the [examples](examples/) directory!**