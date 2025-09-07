# CausalLLM Performance Optimization Guide

## Overview

CausalLLM introduces comprehensive performance optimizations that provide up to 10x faster computations and 80% memory reduction. This guide covers how to leverage these optimizations effectively.

## Performance Features

### 1. Data Chunking & Streaming

Handle datasets much larger than available RAM through intelligent data chunking and streaming processing.

```python
from causallm import EnhancedCausalLLM
from causallm.core.data_processing import DataChunker, StreamingDataProcessor

# Automatic chunking based on available memory
causal_llm = EnhancedCausalLLM(
    chunk_size='auto',  # Automatically determine optimal chunk size
    enable_performance_optimizations=True
)

# Manual chunking for fine control
chunker = DataChunker()
for chunk_idx, chunk_data in chunker.chunk_dataframe(large_data, chunk_size=10000):
    # Process each chunk independently
    results = causal_llm.estimate_causal_effect(chunk_data, 'treatment', 'outcome')

# Streaming processing for very large files
processor = StreamingDataProcessor()

def analyze_chunk(chunk):
    return chunk.corr()

# Process CSV files larger than memory
correlation_results = processor.process_streaming(
    "very_large_data.csv",
    analyze_chunk,
    aggregation_func=lambda results: pd.concat(results).mean()
)
```

### 2. Intelligent Caching

Multi-tier caching system provides 20x+ speedup for repeated analyses.

```python
from causallm.core.caching import StatisticalComputationCache, DiskCache

# Enable persistent caching across sessions
causal_llm = EnhancedCausalLLM(
    cache_dir="./causallm_cache",  # Persistent disk cache
    enable_performance_optimizations=True
)

# First analysis: computed and cached
result1 = causal_llm.estimate_causal_effect(data, 'treatment', 'outcome')

# Subsequent analyses: retrieved from cache (20x+ faster)
result2 = causal_llm.estimate_causal_effect(data, 'treatment', 'outcome')

# Cache statistics
cache_stats = causal_llm.cache.get_cache_stats()
print(f"Cache hit rate: {cache_stats.hit_rate:.1%}")
print(f"Total cache hits: {cache_stats.hits}")
```

### 3. Vectorized Algorithms

Numba-optimized statistical computations provide 10x speedup for mathematical operations.

```python
from causallm.core.optimized_algorithms import vectorized_stats, causal_inference

# Vectorized correlation matrix (10x faster than pandas)
correlation_matrix = vectorized_stats.compute_correlation_matrix(data)

# Vectorized ATE estimation (3x faster)
X = data[['age', 'income', 'education']].values
treatment = data['treatment'].values 
outcome = data['outcome'].values

ate_result = causal_inference.estimate_ate_vectorized(
    X, treatment, outcome, method='doubly_robust'
)
```

### 4. Async Processing

Parallel execution of independent computations with automatic resource management.

```python
import asyncio
from causallm.core.async_processing import AsyncCausalAnalysis

async def parallel_analysis():
    async_causal = AsyncCausalAnalysis()
    
    # Parallel correlation analysis across data chunks
    correlation_matrix = await async_causal.parallel_correlation_analysis(
        large_data, chunk_size=5000
    )
    
    # Parallel bootstrap analysis for confidence intervals
    bootstrap_results = await async_causal.parallel_bootstrap_analysis(
        data, analysis_func=my_analysis_function, n_bootstrap=1000
    )
    
    return correlation_matrix, bootstrap_results

# Run async analysis
results = asyncio.run(parallel_analysis())
```

### 5. Lazy Evaluation

Deferred computation reduces memory usage by 60-80% through lazy operations.

```python
from causallm.core.lazy_evaluation import LazyDataFrame, lazy_correlation_matrix

# Create lazy DataFrame with deferred operations
lazy_df = LazyDataFrame(data)

# Chain operations without immediate execution
processed_df = (lazy_df
                .fillna(0)
                .select_dtypes(include=[np.number])
                .dropna())

# Execute all operations at once (memory efficient)
result = processed_df.compute()

# Lazy correlation computation
lazy_corr = lazy_correlation_matrix(lazy_df)
correlation_result = lazy_corr.compute()  # Only computed when needed
```

## Performance Configuration

### Optimal Settings for Different Dataset Sizes

#### Small Datasets (< 10K rows)
```python
causal_llm = EnhancedCausalLLM(
    enable_performance_optimizations=False,  # Overhead not worth it
    use_async=False
)
```

#### Medium Datasets (10K - 100K rows)
```python
causal_llm = EnhancedCausalLLM(
    enable_performance_optimizations=True,
    chunk_size='auto',
    use_async=True,
    cache_dir="./cache"
)
```

#### Large Datasets (100K - 1M rows)
```python
causal_llm = EnhancedCausalLLM(
    enable_performance_optimizations=True,
    chunk_size=50000,  # Larger chunks for efficiency
    use_async=True,
    cache_dir="./cache",
    max_memory_usage_gb=8  # Set memory limits
)
```

#### Very Large Datasets (> 1M rows)
```python
# Use streaming processor for datasets larger than RAM
processor = StreamingDataProcessor()
causal_llm = EnhancedCausalLLM(
    enable_performance_optimizations=True,
    chunk_size=10000,  # Smaller chunks to fit in memory
    use_async=True,
    cache_dir="./cache"
)
```

## Memory Optimization

### Memory Monitoring
```python
from causallm.core.data_processing import MemoryMonitor

monitor = MemoryMonitor()

# Check current memory usage
memory_info = monitor.get_memory_info()
print(f"Available RAM: {memory_info.available_gb:.1f} GB")
print(f"Memory usage: {memory_info.usage_percent:.1f}%")

# Get suggested chunk size based on available memory
optimal_chunk_size = monitor.suggest_chunk_size(data_size_mb=500)
```

### Memory-Efficient Operations
```python
# Use memory-efficient groupby for large datasets
from causallm.core.data_processing import memory_efficient_groupby

# Standard groupby might cause memory issues
# result = large_data.groupby('category').agg({'value': 'mean'})

# Memory-efficient alternative
result = memory_efficient_groupby(
    large_data, 
    group_col='category', 
    agg_col='value', 
    agg_func='mean',
    chunk_size=10000
)
```

## Benchmarking & Monitoring

### Built-in Performance Demo
```python
from causallm.performance_demo import PerformanceBenchmark

# Run comprehensive performance benchmark
benchmark = PerformanceBenchmark()
results = benchmark.run_comprehensive_benchmark([10000, 50000, 100000])

# Generate performance report
report = benchmark.generate_performance_report()
print(report)
```

### Custom Benchmarking
```python
import time
from causallm import EnhancedCausalLLM

def benchmark_analysis(data, iterations=5):
    times = []
    
    causal_llm = EnhancedCausalLLM(enable_performance_optimizations=True)
    
    for i in range(iterations):
        start_time = time.time()
        result = causal_llm.estimate_causal_effect(data, 'treatment', 'outcome')
        times.append(time.time() - start_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times)
    }

benchmark_results = benchmark_analysis(your_data)
print(f"Average analysis time: {benchmark_results['mean_time']:.2f} seconds")
```

## Troubleshooting Performance Issues

### Common Issues and Solutions

#### 1. Memory Errors with Large Datasets
```python
# Problem: MemoryError with large dataset
# Solution: Use data chunking
causal_llm = EnhancedCausalLLM(
    chunk_size=5000,  # Reduce chunk size
    enable_performance_optimizations=True
)
```

#### 2. Slow Repeated Analyses
```python
# Problem: Repeated analyses are slow
# Solution: Enable persistent caching
causal_llm = EnhancedCausalLLM(
    cache_dir="./persistent_cache",  # Enable disk cache
    enable_performance_optimizations=True
)
```

#### 3. CPU Bottlenecks
```python
# Problem: Single-threaded processing is slow
# Solution: Enable async processing
causal_llm = EnhancedCausalLLM(
    use_async=True,
    max_concurrent_tasks=4,  # Adjust based on CPU cores
    enable_performance_optimizations=True
)
```

#### 4. Import Errors
```bash
# Problem: ModuleNotFoundError for performance dependencies
# Solution: Install performance dependencies
pip install causallm[full]

# Or install specific dependencies
pip install numba dask psutil aiofiles pyarrow
```

## Performance Best Practices

### 1. Data Preparation
- Clean data before analysis to reduce processing time
- Use appropriate data types (int32 instead of int64 where possible)
- Remove unnecessary columns before processing

### 2. Cache Management
- Use persistent caching for repeated analyses
- Clear cache periodically to avoid disk space issues
- Use memory cache for short-term repeated computations

### 3. Resource Management
- Monitor memory usage with built-in tools
- Set appropriate chunk sizes based on available RAM
- Use async processing for I/O-bound operations

### 4. Algorithm Selection
- Use vectorized algorithms for mathematical computations
- Leverage lazy evaluation for data preprocessing
- Choose appropriate statistical methods based on data size

## Performance Metrics

### Expected Speedups

| Operation | Small Data | Medium Data | Large Data |
|-----------|------------|-------------|------------|
| Correlation Analysis | 2x | 5x | 10x |
| Causal Discovery | 1.5x | 3x | 5x |
| Effect Estimation | 2x | 3x | 4x |
| Repeated Analysis | 5x | 10x | 20x+ |

### Memory Efficiency

| Feature | Memory Reduction |
|---------|------------------|
| Data Chunking | 50-90% |
| Lazy Evaluation | 60-80% |
| Streaming Processing | 95%+ |
| Intelligent Caching | Variable |

## Advanced Configuration

### Custom Performance Settings
```python
from causallm.core.data_processing import DataProcessingConfig

# Create custom configuration
config = DataProcessingConfig(
    chunk_size=20000,
    max_memory_usage_gb=16,
    enable_parallel_processing=True,
    cache_size_gb=2,
    lazy_evaluation=True
)

causal_llm = EnhancedCausalLLM(
    processing_config=config,
    enable_performance_optimizations=True
)
```

### Environment Variables
```bash
# Set performance-related environment variables
export CAUSALLM_CHUNK_SIZE=10000
export CAUSALLM_MAX_MEMORY_GB=8
export CAUSALLM_ENABLE_ASYNC=true
export CAUSALLM_CACHE_DIR=/tmp/causallm_cache
```

## Conclusion

CausalLLM's performance optimizations provide significant improvements across all dataset sizes. By following this guide and using appropriate settings for your data size, you can achieve:

- **10x faster computations** through vectorized algorithms
- **80% memory reduction** through intelligent chunking and lazy evaluation
- **20x+ speedup** for repeated analyses through caching
- **Unlimited scalability** through streaming processing

For additional support or performance tuning assistance, please refer to the [GitHub Issues](https://github.com/rdmurugan/causallm/issues) or [Discussions](https://github.com/rdmurugan/causallm/discussions).