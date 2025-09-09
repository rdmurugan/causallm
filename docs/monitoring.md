# Monitoring & Observability

CausalLLM v4.2.0 introduces comprehensive monitoring and observability features designed for production environments. This documentation covers metrics collection, health checks, and performance profiling.

## Overview

The monitoring system provides three main components:

- **Metrics Collection**: Track performance, usage patterns, and system health
- **Health Checks**: Monitor system components and dependencies  
- **Performance Profiling**: Detailed memory and execution analysis

## Quick Start

```python
from causallm.monitoring import configure_metrics, get_global_health_checker
from causallm.monitoring.profiler import get_global_profiler, profile

# Enable monitoring
collector = configure_metrics(enabled=True, collection_interval=30)
health_checker = get_global_health_checker()
profiler = get_global_profiler()

# Use decorators for automatic profiling
@profile(name="my_causal_function")
def analyze_data(data):
    # Your causal inference code
    pass
```

## Metrics Collection

### MetricsCollector

The `MetricsCollector` provides thread-safe metrics collection with multiple metric types:

```python
from causallm.monitoring.metrics import MetricsCollector

collector = MetricsCollector(enabled=True, collection_interval=5.0)

# Record different metric types
collector.record_counter('requests.total', 1.0)
collector.set_gauge('memory.usage', 512.5)
collector.record_histogram('response.time', 0.125)
```

### Built-in Metrics

CausalLLM automatically tracks several metric categories:

#### System Metrics
- `system.cpu.percent` - CPU usage percentage
- `system.memory.percent` - Memory usage percentage  
- `system.disk.percent` - Disk usage percentage

#### Causal Discovery Metrics
- `causal_discovery.requests` - Number of discovery requests
- `causal_discovery.duration` - Discovery execution time
- `causal_discovery.variables_count` - Number of variables analyzed
- `causal_discovery.success/failure` - Success/failure counters

#### LLM Request Metrics
- `llm.requests` - LLM API requests
- `llm.duration` - LLM response time
- `llm.tokens_used` - Token consumption
- `llm.cost` - API costs (when available)

### Custom Metrics

Record custom metrics for your specific use cases:

```python
# Track business metrics
collector.record_counter('analysis.completed', 1.0, {'domain': 'healthcare'})
collector.record_histogram('data.processing.time', processing_time, {'dataset_size': 'large'})
collector.set_gauge('active.sessions', session_count)

# Export metrics
collector.export_metrics_json('metrics.json')
summary = collector.get_metrics_summary()
```

### Automatic Operation Tracking

Use the `track_operation` decorator for automatic metrics:

```python
from causallm.monitoring.metrics import track_operation, get_global_collector

collector = get_global_collector()

@track_operation(collector, "causal_analysis", domain="healthcare")
def run_analysis(data):
    # Automatically tracks duration, success/failure
    return analyze_causal_relationships(data)
```

## Health Checks

### HealthChecker

Monitor system health with configurable checks:

```python
from causallm.monitoring.health import HealthChecker, HealthStatus

health_checker = HealthChecker(enabled=True)

# Run all health checks
results = await health_checker.run_all_health_checks()

# Get overall health status
overall_health = health_checker.get_overall_health()
print(f"Status: {overall_health['status']}")  # healthy, degraded, unhealthy, unknown
```

### Built-in Health Checks

#### System Resources Check
Monitors CPU, memory, and disk usage:

```python
from causallm.monitoring.health import SystemResourcesHealthCheck

resource_check = SystemResourcesHealthCheck(
    cpu_threshold=90.0,      # Alert if CPU > 90%
    memory_threshold=85.0,   # Alert if memory > 85%
    disk_threshold=95.0      # Alert if disk > 95%
)

health_checker.add_health_check(resource_check)
```

#### Database Connectivity Check
Tests database connections:

```python
from causallm.monitoring.health import DatabaseHealthCheck

db_check = DatabaseHealthCheck(
    connection_string="postgresql://user:pass@localhost/db"
)
health_checker.add_health_check(db_check)
```

#### LLM Provider Check  
Verifies LLM API connectivity:

```python
from causallm.monitoring.health import LLMProviderHealthCheck

llm_check = LLMProviderHealthCheck(
    provider="openai",
    api_key="your-api-key"
)
health_checker.add_health_check(llm_check)
```

### Custom Health Checks

Create custom health checks for your components:

```python
from causallm.monitoring.health import HealthCheck, HealthStatus

class CustomComponentCheck(HealthCheck):
    def __init__(self, component):
        super().__init__("my_component", timeout=30.0)
        self.component = component
    
    async def _perform_check(self):
        try:
            if self.component.is_healthy():
                return HealthStatus.HEALTHY, "Component is healthy", None
            else:
                return HealthStatus.DEGRADED, "Component has issues", None
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Component failed: {e}", None

# Add to health checker
health_checker.add_health_check(CustomComponentCheck(my_component))
```

### Background Monitoring

Enable continuous health monitoring:

```python
# Start background monitoring (checks every 60 seconds)
await health_checker.start_background_monitoring(interval=60.0)

# Monitor continuously
while True:
    overall_health = health_checker.get_overall_health()
    if overall_health['status'] != 'healthy':
        send_alert(overall_health)
    await asyncio.sleep(60)

# Stop monitoring
await health_checker.stop_background_monitoring()
```

## Performance Profiling

### PerformanceProfiler

Detailed performance analysis with memory tracking:

```python
from causallm.monitoring.profiler import PerformanceProfiler

profiler = PerformanceProfiler(
    enabled=True, 
    detailed_profiling=True  # Enable memory tracking
)

# Profile with context manager
with profiler.profile_context("data_processing"):
    process_large_dataset(data)

# Get profiling results
summary = profiler.get_performance_summary()
print(f"Total calls: {summary['summary']['total_calls']}")
print(f"Avg duration: {summary['summary']['avg_duration_ms']:.2f}ms")
```

### Function Profiling

Use decorators for automatic function profiling:

```python
from causallm.monitoring.profiler import profile

@profile(name="causal_discovery", track_memory=True, track_cpu=True)
def discover_relationships(data, variables):
    # Function is automatically profiled
    return run_pc_algorithm(data, variables)

# Or use global profiler
from causallm.monitoring.profiler import get_global_profiler

profiler = get_global_profiler()

@profiler.profile_function("custom_analysis")
async def async_analysis(data):
    return await analyze_async(data)
```

### Memory Profiling

Track memory usage patterns:

```python
from causallm.monitoring.profiler import MemoryProfiler

# Get current memory usage
memory_stats = MemoryProfiler.get_memory_usage()
print(f"RSS: {memory_stats['rss_mb']:.2f}MB")
print(f"Available: {memory_stats['available_mb']:.2f}MB")

# Track memory differences
with MemoryProfiler.track_memory_diff() as diff:
    # Memory-intensive operation
    large_computation()

print(f"Memory increased by: {diff['memory_diff_mb']:.2f}MB")
```

### Performance Analysis

Analyze performance patterns:

```python
# Get function statistics
function_stats = profiler.get_function_stats("causal_discovery")
print(f"Total calls: {function_stats['total_calls']}")
print(f"Average time: {function_stats['avg_time']:.4f}s")
print(f"Min time: {function_stats['min_time']:.4f}s")
print(f"Max time: {function_stats['max_time']:.4f}s")

# Find slow operations
slowest = profiler.get_slowest_operations(top_n=5)
memory_intensive = profiler.get_memory_intensive_operations(top_n=5)

# Export profiling data
profiler.export_profile_data('profile_results.json')
```

## Integration Examples

### Production Monitoring Setup

```python
import asyncio
from causallm.monitoring import MetricsCollector, HealthChecker, PerformanceProfiler

class ProductionMonitor:
    def __init__(self):
        # Configure monitoring components
        self.metrics = MetricsCollector(enabled=True, collection_interval=30)
        self.health_checker = HealthChecker(enabled=True)
        self.profiler = PerformanceProfiler(enabled=True, detailed_profiling=True)
        
        # Add health checks
        self._setup_health_checks()
    
    def _setup_health_checks(self):
        from causallm.monitoring.health import (
            SystemResourcesHealthCheck, 
            LLMProviderHealthCheck
        )
        
        # System resources
        self.health_checker.add_health_check(
            SystemResourcesHealthCheck(cpu_threshold=85, memory_threshold=80)
        )
        
        # LLM providers
        self.health_checker.add_health_check(
            LLMProviderHealthCheck("openai", api_key=os.getenv("OPENAI_API_KEY"))
        )
    
    async def get_monitoring_dashboard(self):
        """Get comprehensive monitoring data."""
        # Health status
        health_results = await self.health_checker.run_all_health_checks()
        overall_health = self.health_checker.get_overall_health()
        
        # Metrics summary
        metrics_summary = self.metrics.get_metrics_summary()
        
        # Performance data
        performance_summary = self.profiler.get_performance_summary()
        slowest_ops = self.profiler.get_slowest_operations(top_n=10)
        
        return {
            'health': {
                'overall': overall_health,
                'checks': {name: result.to_dict() for name, result in health_results.items()}
            },
            'metrics': metrics_summary,
            'performance': {
                'summary': performance_summary,
                'slowest_operations': slowest_ops
            },
            'timestamp': datetime.now().isoformat()
        }
    
    async def start_monitoring(self):
        """Start continuous monitoring."""
        await self.health_checker.start_background_monitoring(interval=60)
        print("✅ Production monitoring started")

# Usage
monitor = ProductionMonitor()
dashboard_data = await monitor.get_monitoring_dashboard()
await monitor.start_monitoring()
```

### Alerting Integration

```python
class AlertManager:
    def __init__(self, monitor: ProductionMonitor):
        self.monitor = monitor
    
    async def check_and_alert(self):
        """Check system status and send alerts."""
        dashboard = await self.monitor.get_monitoring_dashboard()
        
        # Health alerts
        if dashboard['health']['overall']['status'] != 'healthy':
            await self.send_health_alert(dashboard['health'])
        
        # Performance alerts
        if dashboard['performance']['summary']['summary']['max_duration_ms'] > 10000:
            await self.send_performance_alert(dashboard['performance'])
        
        # Metrics alerts
        metrics = dashboard['metrics']['metrics']
        if 'system.cpu.percent' in metrics['gauges']:
            if metrics['gauges']['system.cpu.percent'] > 90:
                await self.send_resource_alert('CPU', metrics['gauges']['system.cpu.percent'])
    
    async def send_health_alert(self, health_data):
        print(f"🚨 Health Alert: {health_data['overall']['message']}")
        # Integrate with your alerting system (Slack, PagerDuty, etc.)
    
    async def send_performance_alert(self, perf_data):
        print(f"⚠️ Performance Alert: Slow operations detected")
        # Send performance alert
    
    async def send_resource_alert(self, resource, value):
        print(f"💾 Resource Alert: {resource} usage at {value}%")
        # Send resource alert
```

## Best Practices

### 1. Monitoring Configuration

```python
# Development
configure_metrics(enabled=True, collection_interval=60)

# Production  
configure_metrics(enabled=True, collection_interval=30)

# High-performance production
configure_metrics(enabled=True, collection_interval=10)
```

### 2. Health Check Thresholds

```python
# Conservative thresholds for critical systems
SystemResourcesHealthCheck(
    cpu_threshold=70.0,
    memory_threshold=75.0,
    disk_threshold=80.0
)

# Standard thresholds
SystemResourcesHealthCheck(
    cpu_threshold=85.0,
    memory_threshold=85.0,
    disk_threshold=90.0
)
```

### 3. Performance Profiling

```python
# Enable detailed profiling for development
profiler = PerformanceProfiler(enabled=True, detailed_profiling=True)

# Lightweight profiling for production
profiler = PerformanceProfiler(enabled=True, detailed_profiling=False)

# Profile only critical functions in production
@profile(name="critical_analysis", track_memory=False)
def critical_function():
    pass
```

### 4. Memory Management

```python
# Clear profiling results periodically in long-running applications
if profiler.get_performance_summary()['summary']['total_calls'] > 10000:
    profiler.clear_results()

# Export and clear metrics periodically
if datetime.now().hour == 0:  # Daily export
    collector.export_metrics_json(f'metrics_{date.today()}.json')
    collector.registry.clear()
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**: Disable detailed profiling or clear results more frequently
2. **Performance Overhead**: Increase collection intervals or disable in performance-critical sections  
3. **Health Check Failures**: Adjust thresholds based on your environment
4. **Missing Metrics**: Ensure collectors are properly configured and enabled

### Debug Logging

```python
import logging
logging.getLogger('causallm.monitoring').setLevel(logging.DEBUG)
```

### Performance Impact

The monitoring system is designed for minimal overhead:

- **Metrics Collection**: ~0.1ms per metric
- **Health Checks**: Run async, don't block main thread
- **Profiling**: <1% overhead when enabled
- **Memory Tracking**: ~5MB baseline, scales with usage

Disable monitoring in performance-critical sections if needed:

```python
# Temporarily disable monitoring
collector.enabled = False
profiler.enabled = False

# Critical performance section
critical_computation()

# Re-enable monitoring
collector.enabled = True  
profiler.enabled = True
```