# Changelog

All notable changes to CausalLLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.2.0] - 2025-09-09

### Added - Major Feature Release: Monitoring & Observability + Extended Testing

#### 🔍 Monitoring & Observability
- **Comprehensive Metrics Collection System**
  - `MetricsCollector` with thread-safe metrics registry
  - Support for counters, gauges, and histograms
  - Automatic system metrics (CPU, memory, disk usage)
  - Built-in tracking for causal discovery, LLM requests, and data processing operations
  - JSON export capabilities for external monitoring systems
  - Global collector instance with easy configuration

- **Advanced Health Checking System**
  - `HealthChecker` with multiple health check types
  - System resource monitoring with configurable thresholds
  - Database connectivity checks
  - LLM provider API health verification
  - Component-specific health checks
  - Background monitoring with configurable intervals
  - Health status levels: Healthy, Degraded, Unhealthy, Unknown

- **Performance Profiling Infrastructure**
  - `PerformanceProfiler` with detailed performance analysis
  - Memory tracking using `tracemalloc` (peak usage and deltas)
  - Function-level profiling with decorators
  - Context manager for profiling code blocks
  - Statistical analysis (min, max, average, percentiles)
  - Export capabilities for performance data
  - Integration with Python's `cProfile` for detailed profiling

#### 🧪 Extended Testing Framework
- **Property-Based Testing with Hypothesis**
  - `CausalDataStrategy` for generating realistic causal datasets
  - `CausalGraphStrategy` for generating valid DAGs
  - Property tests for causal inference algorithms (symmetry, consistency)
  - `CausalTestRunner` for comprehensive property-based test execution
  - Custom strategies for mixed-type data and structured causal data

- **Performance Benchmarking System**
  - `BenchmarkSuite` for comprehensive algorithm benchmarking
  - `CausalBenchmarkSuite` specialized for causal inference algorithms
  - Algorithm performance comparison and ranking
  - Scaling analysis (performance vs. data size/variables)
  - Statistical analysis of benchmark results
  - JSON export for benchmark data

- **Mutation Testing Framework**
  - `MutationTestRunner` for assessing test suite quality
  - Multiple mutation operators: arithmetic, comparison, boolean, conditional, constants
  - AST-based mutations for precise code changes
  - Configurable mutation testing with timeout and file filtering
  - Detailed mutation results with survival/kill rates
  - Mutation score calculation and threshold checking

#### 📦 Package Enhancements
- **New Dependencies**
  - Added `hypothesis>=6.0.0` for property-based testing
  - Added `pytest-benchmark>=4.0.0` for performance benchmarking
  - Added `mutmut>=2.0.0` for mutation testing support
  - Added `pytest-xdist>=3.0.0` for parallel test execution
  - Added `pytest-mock>=3.0.0` for advanced mocking

- **Integration & Usability**
  - Clean import structure with fallback handling for optional dependencies
  - Global instances for easy access to monitoring components
  - Backward compatibility - all new features are optional
  - Comprehensive example script demonstrating all features
  - Integration with existing CausalLLM components

#### 🛠 API Enhancements
- **Decorator-Based Profiling**
  ```python
  from causallm.monitoring.profiler import profile
  
  @profile(name="causal_discovery")
  def my_discovery_function(data):
      # Your code here
  ```

- **Context Manager Profiling**
  ```python
  from causallm.monitoring.profiler import profile_block
  
  with profile_block("data_processing"):
      # Your code here
  ```

- **Easy Metrics Collection**
  ```python
  from causallm.monitoring.metrics import get_global_collector
  
  collector = get_global_collector()
  collector.record_causal_discovery(variables_count=5, duration=2.5, method='PC')
  ```

- **Health Monitoring**
  ```python
  from causallm.monitoring.health import get_global_health_checker
  
  health_checker = get_global_health_checker()
  results = await health_checker.run_all_health_checks()
  ```

### Changed
- Updated package metadata and dependencies in `pyproject.toml`
- Enhanced main `__init__.py` with new component imports
- Improved error handling and logging throughout the library

### Technical Details
- **Thread Safety**: All monitoring components are thread-safe
- **Memory Efficiency**: Optional memory tracking to minimize overhead
- **Async Support**: Health checks support async operations
- **Extensibility**: Plugin architecture for custom mutators and health checks
- **Performance**: Minimal overhead when monitoring is disabled

### Documentation
- Added comprehensive example script (`examples/monitoring_and_testing_demo.py`)
- Detailed docstrings for all new classes and methods
- Type hints throughout the codebase
- Integration examples for all new features

---

## [4.1.1] - 2025-09-07

### Fixed
- Package installation issues
- Documentation updates
- Minor bug fixes in core components

### Changed
- Updated dependencies to latest versions
- Improved error handling in LLM client connections

---

## [4.1.0] - 2025-08-26

### Added
- Enhanced causal discovery algorithms
- Domain-specific templates (Healthcare, Marketing, Education)
- Improved statistical inference methods
- Web interface enhancements

### Changed
- Refactored core architecture for better extensibility
- Updated LLM client implementations
- Performance optimizations

### Fixed
- Memory leaks in large dataset processing
- Edge cases in causal graph parsing
- Thread safety issues in concurrent operations

---

## [4.0.0] - 2025-08-25

### Added
- Initial stable release of CausalLLM
- Core causal inference algorithms (PC, GES, DirectLiNGAM)
- LLM integration for causal reasoning
- Statistical validation methods
- Do-calculus implementation
- Counterfactual reasoning
- Web interface
- Comprehensive test suite
- Documentation and examples

### Technical Specifications
- Python 3.9+ support
- Multi-platform compatibility (Windows, macOS, Linux)
- Async/await support for LLM operations
- Memory-efficient processing for large datasets
- Extensible plugin architecture