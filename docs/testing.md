# Extended Testing Framework

CausalLLM v4.2.0 introduces a comprehensive testing framework designed specifically for causal inference algorithms. This includes property-based testing, performance benchmarking, and mutation testing.

## Overview

The testing framework provides three main components:

- **Property-Based Testing**: Automated property verification using Hypothesis
- **Performance Benchmarks**: Algorithm comparison and scaling analysis
- **Mutation Testing**: Test suite quality assessment

## Installation

```bash
pip install "causallm[testing]"
```

This installs the additional testing dependencies:
- `hypothesis>=6.0.0` - Property-based testing
- `pytest-benchmark>=4.0.0` - Performance benchmarking
- `mutmut>=2.0.0` - Mutation testing support
- `pytest-xdist>=3.0.0` - Parallel test execution
- `pytest-mock>=3.0.0` - Advanced mocking

## Property-Based Testing

### Overview

Property-based testing automatically generates test cases and verifies that mathematical properties hold across diverse inputs. This is particularly powerful for causal inference where algorithms must satisfy statistical properties.

### Quick Start

```python
from causallm.testing import CausalDataStrategy, causal_hypothesis_test
from hypothesis import given, settings
import pandas as pd

class TestCausalInference:
    @given(CausalDataStrategy.numeric_data(['X', 'Y', 'Z'], min_rows=100))
    @settings(max_examples=50, deadline=60000)
    def test_independence_symmetry(self, data):
        """Test that CI(X,Y|Z) equals CI(Y,X|Z)"""
        result1 = my_ci_test(data, 'X', 'Y', ['Z'])
        result2 = my_ci_test(data, 'Y', 'X', ['Z']) 
        
        # Allow small numerical differences
        assert abs(result1[1] - result2[1]) < 0.01
```

### Data Generation Strategies

#### CausalDataStrategy

Generate realistic datasets for testing:

```python
from causallm.testing.property_based import CausalDataStrategy

# Generate numeric data
@given(CausalDataStrategy.numeric_data(
    variables=['Treatment', 'Outcome', 'Confounder'], 
    min_rows=200, 
    max_rows=1000
))
def test_with_numeric_data(data):
    # Your test here
    pass

# Generate mixed-type data (numeric + categorical)
@given(CausalDataStrategy.mixed_data(
    variables=['Age', 'Treatment', 'Gender', 'Outcome'],
    min_rows=100
))
def test_with_mixed_data(data):
    # Your test here
    pass

# Generate data with known causal structure
edges = [('X1', 'X2'), ('X2', 'Y'), ('X3', 'Y')]
@given(CausalDataStrategy.causal_data_with_structure(
    variables=['X1', 'X2', 'X3', 'Y'],
    true_edges=edges,
    min_rows=200,
    noise_level=0.1
))
def test_with_structured_data(data):
    # Test discovery on data with known structure
    discovered_edges = my_discovery_algorithm(data)
    
    # Property: Should find at least some true edges
    true_edge_set = set(edges)
    discovered_edge_set = set(discovered_edges)
    assert len(true_edge_set & discovered_edge_set) > 0
```

#### CausalGraphStrategy

Generate valid causal graphs:

```python
from causallm.testing.property_based import CausalGraphStrategy

# Generate DAG edges
@given(
    variables=CausalDataStrategy.variable_names(min_size=4, max_size=8),
    edges=CausalGraphStrategy.dag_edges(variables, density=0.3)
)
def test_graph_properties(variables, edges):
    graph = nx.DiGraph(edges)
    
    # Property: Generated graph should be a DAG
    assert nx.is_directed_acyclic_graph(graph)
    
    # Property: All variables should be in the graph
    graph_nodes = set(graph.nodes())
    variable_set = set(variables)
    assert variable_set.issubset(graph_nodes)
```

### Built-in Property Tests

#### Conditional Independence Properties

```python
from causallm.testing.property_based import CausalInferenceProperties

# Test CI symmetry property
symmetry_test = CausalInferenceProperties.test_independence_symmetry(my_ci_test)
symmetry_test()  # Runs property test

# Test discovery consistency
consistency_test = CausalInferenceProperties.test_causal_discovery_consistency(my_discovery_algorithm)
consistency_test()

# Test do-calculus properties  
do_test = CausalInferenceProperties.test_do_calculus_properties(my_do_operator)
do_test()
```

### Custom Property Tests

Create domain-specific property tests:

```python
from causallm.testing.property_based import PropertyBasedTestCase

class CausalDiscoveryProperties(PropertyBasedTestCase):
    def __init__(self, discovery_algorithm):
        super().__init__(max_examples=100, deadline=120000)
        self.discovery_algorithm = discovery_algorithm
    
    def test_monotonicity_property(self, data):
        """Test that adding more data doesn't decrease edge confidence."""
        # Test with subset of data
        subset_data = data.sample(frac=0.5)
        subset_result = self.discovery_algorithm(subset_data)
        
        # Test with full data
        full_result = self.discovery_algorithm(data)
        
        # Property: Full data should have higher/equal confidence
        for edge in subset_result.edges:
            if edge in full_result.edges:
                subset_conf = subset_result.get_edge_confidence(edge)
                full_conf = full_result.get_edge_confidence(edge)
                assert full_conf >= subset_conf - 0.1  # Allow small decrease
        
        return True

# Usage
properties = CausalDiscoveryProperties(my_algorithm)
test_func = properties.run_property_test(
    CausalDataStrategy.numeric_data(['X', 'Y', 'Z'], min_rows=100),
    properties.test_monotonicity_property
)
test_func()  # Run the property test
```

### Test Runner

Use the comprehensive test runner:

```python
from causallm.testing.property_based import CausalTestRunner

runner = CausalTestRunner(max_examples=20, deadline=120000)

# Run all standard causal inference property tests
ci_results = runner.run_independence_tests(my_ci_test)
discovery_results = runner.run_discovery_tests(my_discovery_algorithm)  
do_results = runner.run_do_calculus_tests(my_do_operator)

# Get summary
summary = runner.get_test_summary()
print(f"Pass rate: {summary['pass_rate']:.1%}")
```

## Performance Benchmarking

### Overview

Performance benchmarking compares algorithms across different data sizes and complexities, providing detailed statistical analysis of performance characteristics.

### Quick Start

```python
from causallm.testing import PerformanceBenchmark, CausalBenchmarkSuite

# Simple benchmark
benchmark = PerformanceBenchmark("my_algorithm", benchmark_iterations=10)
result = benchmark.run(my_function, data, variables)

print(f"Average duration: {result.duration_seconds:.4f}s")
print(f"Memory peak: {result.memory_peak_mb:.2f}MB")
```

### CausalBenchmarkSuite

Comprehensive benchmarking for causal algorithms:

```python
from causallm.testing.benchmarks import CausalBenchmarkSuite

# Configure benchmark suite
suite = CausalBenchmarkSuite(
    data_sizes=[100, 500, 1000, 5000, 10000],
    variable_counts=[3, 5, 10, 15, 20]
)

# Benchmark causal discovery algorithms
algorithms = {
    'pc_algorithm': my_pc_implementation,
    'ges_algorithm': my_ges_implementation, 
    'direct_lingam': my_lingam_implementation
}

results = {}
for name, algorithm in algorithms.items():
    print(f"Benchmarking {name}...")
    results[name] = suite.benchmark_causal_discovery(algorithm, name)
```

### Algorithm Comparison

```python
# Compare algorithm performance
comparison = suite.compare_algorithms({
    name: suite.results[name] for name in algorithms.keys()
})

print(f"Fastest algorithm: {comparison['_summary']['fastest_algorithm']}")
print(f"Most memory efficient: {comparison['_summary']['most_memory_efficient']}")

# Detailed comparison
for alg_name, stats in comparison.items():
    if alg_name != '_summary':
        print(f"\n{alg_name}:")
        print(f"  Avg duration: {stats['avg_duration_seconds']:.4f}s")
        print(f"  Avg memory: {stats['avg_memory_peak_mb']:.2f}MB")
        print(f"  Total benchmarks: {stats['total_benchmarks']}")
```

### Scaling Analysis

Analyze how performance scales with problem size:

```python
# Get scaling analysis
for name, result in results.items():
    if 'scaling_analysis' in result:
        scaling = result['scaling_analysis']
        
        print(f"\n{name} Scaling:")
        if 'row_scaling' in scaling:
            row_scaling = scaling['row_scaling']
            print(f"  Data size scaling: {row_scaling['data_points']} data points")
            print(f"  Min time: {row_scaling['min_time_at_size']['time']:.4f}s at {row_scaling['min_time_at_size']['size']} rows")
            print(f"  Max time: {row_scaling['max_time_at_size']['time']:.4f}s at {row_scaling['max_time_at_size']['size']} rows")
        
        if 'variable_scaling' in scaling:
            var_scaling = scaling['variable_scaling']  
            print(f"  Variable scaling: {var_scaling['data_points']} data points")
            print(f"  Min time: {var_scaling['min_time_at_size']['time']:.4f}s with {var_scaling['min_time_at_size']['size']} variables")
            print(f"  Max time: {var_scaling['max_time_at_size']['time']:.4f}s with {var_scaling['max_time_at_size']['size']} variables")
```

### Custom Benchmarks

Create specialized benchmarks:

```python
class CustomCausalBenchmark(PerformanceBenchmark):
    def __init__(self, name, special_config):
        super().__init__(name, warmup_iterations=5, benchmark_iterations=20)
        self.special_config = special_config
    
    def run_with_config(self, func, data):
        """Run benchmark with special configuration."""
        # Configure environment
        self._setup_special_environment()
        
        # Run benchmark
        result = self.run(func, data, **self.special_config)
        
        # Add custom metadata
        result.metadata.update({
            'config': self.special_config,
            'environment': self._get_environment_info()
        })
        
        return result

# Usage
custom_benchmark = CustomCausalBenchmark("optimized_pc", {'alpha': 0.05, 'parallel': True})
result = custom_benchmark.run_with_config(my_optimized_pc, large_dataset)
```

### BenchmarkSuite Integration

Use the general benchmark suite:

```python
from causallm.testing.benchmarks import BenchmarkSuite

suite = BenchmarkSuite()

# Add individual benchmarks
result1 = suite.add_benchmark("algorithm_1", my_algo1, data, variables)
result2 = suite.add_benchmark("algorithm_2", my_algo2, data, variables)

# Run comprehensive causal benchmarks
causal_algorithms = {
    'pc': my_pc,
    'ges': my_ges,
    'lingam': my_lingam
}

comprehensive_results = suite.run_causal_benchmarks(causal_algorithms)
```

### Export Results

```python
# Export benchmark results
suite.causal_suite.export_results('benchmark_results.json')

# Load and analyze results later
import json
with open('benchmark_results.json', 'r') as f:
    saved_results = json.load(f)

print(f"Benchmarks run on: {saved_results['export_timestamp']}")
print(f"Data sizes tested: {saved_results['data_sizes_tested']}")
```

## Mutation Testing

### Overview

Mutation testing assesses test suite quality by introducing systematic code changes (mutations) and verifying that tests detect them. High-quality test suites will "kill" most mutations.

### Quick Start

```python
from causallm.testing import MutationTestRunner, MutationTestConfig

# Configure mutation testing
config = MutationTestConfig(
    target_files=['causallm/core/causal_discovery.py'],
    test_command='pytest tests/test_causal_discovery.py -v',
    mutation_score_threshold=0.8,
    max_mutations_per_file=50
)

# Run mutation tests
runner = MutationTestRunner(config)
results = runner.run_mutation_tests()

print(f"Mutation Score: {results['mutation_score']:.2%}")
print(f"Test Quality: {'Good' if results['passed_threshold'] else 'Needs Improvement'}")
```

### Mutation Operators

The framework includes several mutation operators:

#### ArithmeticMutator
```python
# Original: x + y
# Mutations: x - y, x * y, x / y

# Original: a * b  
# Mutations: a + b, a - b, a / b
```

#### ComparisonMutator
```python
# Original: x > y
# Mutations: x >= y, x < y, x <= y

# Original: a == b
# Mutations: a != b, a < b, a > b
```

#### BooleanMutator
```python
# Original: x and y
# Mutations: x or y

# Original: not condition
# Mutations: condition

# Original: True
# Mutations: False
```

#### ConditionalMutator
```python
# Original: if condition:
# Mutations: if not condition:, if True:, if False:
```

#### ConstantMutator
```python
# Original: threshold = 0.05
# Mutations: threshold = 0.06, threshold = 0.04, threshold = 0, threshold = 1
```

### Detailed Analysis

```python
# Analyze results by file
for file_path, stats in results['results_by_file'].items():
    print(f"\n{file_path}:")
    print(f"  Mutations: {stats['total_mutations']}")
    print(f"  Killed: {stats['killed_mutations']}")
    print(f"  Score: {stats['mutation_score']:.2%}")

# Analyze results by mutator type
for mutator_name, stats in results['results_by_mutator'].items():
    print(f"\n{mutator_name}:")
    print(f"  Mutations: {stats['total_mutations']}")
    print(f"  Killed: {stats['killed_mutations']}")
    print(f"  Score: {stats['mutation_score']:.2%}")
```

### Custom Mutators

Create specialized mutators:

```python
from causallm.testing.mutation import BaseMutator
import ast

class CausalParameterMutator(BaseMutator):
    """Mutator for causal inference parameters."""
    
    def __init__(self):
        super().__init__("CausalParameterMutator")
    
    def can_mutate(self, node):
        # Check if this is a causal parameter
        return (isinstance(node, ast.Constant) and 
                isinstance(node.value, float) and
                0.0 <= node.value <= 1.0)  # Likely a significance level or threshold
    
    def mutate(self, node):
        mutations = []
        original_value = node.value
        
        # Common significance levels and thresholds
        causal_values = [0.01, 0.05, 0.10, 0.2, 0.3, 0.5]
        
        for value in causal_values:
            if value != original_value:
                mutated_node = copy.deepcopy(node)
                mutated_node.value = value
                mutations.append(mutated_node)
        
        return mutations

# Add custom mutator to runner
runner = MutationTestRunner(config)
runner.mutators.append(CausalParameterMutator())
```

### Test Quality Analysis

```python
from causallm.testing.mutation import analyze_mutation_results

# Run analysis on saved results
analysis = analyze_mutation_results('mutation_results.json')

print(f"Overall Quality: {analysis['overall_quality']}")

if analysis['weakest_files']:
    print("\nFiles needing better tests:")
    for file_info in analysis['weakest_files']:
        print(f"  {file_info['file']}: {file_info['score']:.2%}")

if analysis['weakest_mutators']:
    print("\nMutation types poorly covered:")
    for mutator_info in analysis['weakest_mutators']:
        print(f"  {mutator_info['mutator']}: {mutator_info['score']:.2%}")

print("\nRecommendations:")
for recommendation in analysis['recommendations']:
    print(f"  - {recommendation}")
```

### Advanced Configuration

```python
config = MutationTestConfig(
    target_files=['causallm/core/*.py'],
    test_command='pytest tests/ -x --tb=short',
    timeout=300,  # 5 minute timeout per mutation
    skip_patterns=['__init__.py', 'test_', '_test.py'],
    mutation_score_threshold=0.85,
    max_mutations_per_file=100,
    random_seed=42  # Reproducible results
)
```

## Integration with CI/CD

### GitHub Actions

```yaml
name: Extended Testing
on: [push, pull_request]

jobs:
  property-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -e .[testing]
      - name: Run property-based tests
        run: |
          python -m pytest tests/property_tests.py --hypothesis-show-statistics

  benchmarks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
      - name: Install dependencies  
        run: |
          pip install -e .[testing]
      - name: Run benchmarks
        run: |
          python scripts/run_benchmarks.py
      - name: Upload benchmark results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: benchmark_results.json

  mutation-testing:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: |
          pip install -e .[testing]
      - name: Run mutation testing
        run: |
          python scripts/run_mutation_tests.py
      - name: Check mutation score
        run: |
          python -c "
          import json
          with open('mutation_results.json') as f:
              results = json.load(f)
          score = results['mutation_test_results']['summary']['mutation_score']
          if score < 0.8:
              exit(1)
          print(f'Mutation score: {score:.2%}')
          "
```

### Test Scripts

Create automated test scripts:

```python
#!/usr/bin/env python3
"""Run comprehensive testing suite."""

import asyncio
import sys
from causallm.testing import CausalTestRunner, CausalBenchmarkSuite, MutationTestRunner, MutationTestConfig

async def run_comprehensive_tests():
    """Run all extended tests."""
    results = {}
    
    # Property-based tests
    print("🧪 Running property-based tests...")
    property_runner = CausalTestRunner(max_examples=50)
    
    try:
        ci_results = property_runner.run_independence_tests(my_ci_test)
        discovery_results = property_runner.run_discovery_tests(my_discovery_algorithm)
        
        results['property_tests'] = property_runner.get_test_summary()
        print(f"✅ Property tests: {results['property_tests']['pass_rate']:.1%} pass rate")
    except Exception as e:
        print(f"❌ Property tests failed: {e}")
        results['property_tests'] = {'error': str(e)}
    
    # Performance benchmarks
    print("🏁 Running performance benchmarks...")
    try:
        benchmark_suite = CausalBenchmarkSuite()
        algorithms = {'pc': my_pc_algorithm, 'ges': my_ges_algorithm}
        
        benchmark_results = {}
        for name, algo in algorithms.items():
            benchmark_results[name] = benchmark_suite.benchmark_causal_discovery(algo, name)
        
        comparison = benchmark_suite.compare_algorithms(
            {name: benchmark_suite.results[name] for name in algorithms.keys()}
        )
        
        results['benchmarks'] = comparison
        print(f"✅ Benchmarks completed: {len(algorithms)} algorithms tested")
    except Exception as e:
        print(f"❌ Benchmarks failed: {e}")
        results['benchmarks'] = {'error': str(e)}
    
    # Mutation testing
    print("🧬 Running mutation testing...")
    try:
        config = MutationTestConfig(
            target_files=['causallm/core/causal_discovery.py'],
            test_command='pytest tests/test_causal_discovery.py --tb=short',
            mutation_score_threshold=0.8
        )
        
        mutation_runner = MutationTestRunner(config)
        mutation_results = mutation_runner.run_mutation_tests()
        
        results['mutation_testing'] = mutation_results
        score = mutation_results['mutation_score']
        quality = 'Good' if score >= 0.8 else 'Needs Improvement'
        print(f"✅ Mutation testing: {score:.1%} score ({quality})")
    except Exception as e:
        print(f"❌ Mutation testing failed: {e}")
        results['mutation_testing'] = {'error': str(e)}
    
    # Save comprehensive results
    import json
    with open('comprehensive_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Determine overall success
    success = all(
        'error' not in result for result in results.values() if isinstance(result, dict)
    )
    
    if success:
        print("\n✅ All extended tests completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed. Check results for details.")
        return 1

if __name__ == '__main__':
    exit_code = asyncio.run(run_comprehensive_tests())
    sys.exit(exit_code)
```

## Best Practices

### Property-Based Testing
1. **Start Simple**: Begin with basic properties before complex ones
2. **Use Realistic Data**: Generate data that matches your domain
3. **Test Edge Cases**: Include boundary conditions in your strategies
4. **Verify Assumptions**: Test the assumptions your algorithms make

### Performance Benchmarking
1. **Multiple Sizes**: Test across different data sizes and complexities
2. **Warmup Runs**: Include warmup iterations for accurate timing
3. **Statistical Significance**: Run enough iterations for reliable results
4. **Environment Consistency**: Control for system load and hardware differences

### Mutation Testing
1. **Good Test Coverage First**: Ensure line/branch coverage before mutation testing
2. **Realistic Thresholds**: Start with 70-80% mutation score, aim for 85%+
3. **Focus on Critical Code**: Prioritize mutation testing on core algorithms
4. **Regular Execution**: Run mutation tests as part of CI/CD pipeline

### General Testing Strategy
1. **Layered Approach**: Use all three testing types for comprehensive coverage
2. **Continuous Integration**: Automate tests in CI/CD pipeline
3. **Performance Monitoring**: Track test execution time and failure rates
4. **Regular Review**: Periodically review and update test strategies