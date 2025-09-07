# CausalLLM API Reference

Complete API documentation for all classes, methods, and functions in the CausalLLM library.

---

## Table of Contents

1. [Core Classes](#core-classes)
   - [EnhancedCausalLLM](#enhancedcausalllm)
   - [CausalLLM](#causalllm)
2. [Statistical Inference](#statistical-inference)
   - [StatisticalCausalInference](#statisticalcausalinference)
   - [EnhancedCausalDiscovery](#enhancedcausaldiscovery)
3. [Domain Packages](#domain-packages)
   - [HealthcareDomain](#healthcaredomain)
   - [InsuranceDomain](#insurancedomain)
   - [MarketingDomain](#marketingdomain)
4. [Performance & Processing](#performance--processing)
   - [AsyncCausalAnalysis](#asynccausalanalysis)
   - [DataChunker](#datachunker)
   - [StreamingDataProcessor](#streamingdataprocessor)
5. [Data Models](#data-models)
   - [CausalDiscoveryResult](#causaldiscoveryresult)
   - [CausalInferenceResult](#causalinferenceresult)
   - [ComprehensiveCausalAnalysis](#comprehensivecausalanalysis)

---

## Core Classes

### EnhancedCausalLLM

**Main high-performance class for causal inference combining statistical methods with optional LLM enhancement.**

```python
from causallm import EnhancedCausalLLM
```

#### `__init__(self, llm_provider=None, llm_model=None, significance_level=0.05, enable_performance_optimizations=True, use_async=False, chunk_size='auto', cache_dir=None, max_memory_usage_gb=None)`

Initialize the EnhancedCausalLLM instance.

**Parameters:**
- `llm_provider` (str, optional): LLM provider to use. Options: 'openai', 'anthropic', 'llama', 'mcp', or None for statistical-only mode.
- `llm_model` (str, optional): Specific model name (e.g., 'gpt-4', 'claude-3-opus'). Defaults to provider's recommended model.
- `significance_level` (float, default=0.05): Statistical significance threshold for hypothesis testing.
- `enable_performance_optimizations` (bool, default=True): Enable vectorized algorithms, caching, and chunking for large datasets.
- `use_async` (bool, default=False): Enable asynchronous processing for parallel computations.
- `chunk_size` (int or str, default='auto'): Data chunk size for memory management. 'auto' determines optimal size.
- `cache_dir` (str, optional): Directory for persistent caching. None disables disk caching.
- `max_memory_usage_gb` (float, optional): Maximum memory usage limit in GB for automatic optimization.

**Example:**
```python
# Basic usage with statistical methods only
causal_llm = EnhancedCausalLLM()

# High-performance setup with LLM integration
causal_llm = EnhancedCausalLLM(
    llm_provider='openai',
    llm_model='gpt-4',
    enable_performance_optimizations=True,
    use_async=True,
    cache_dir='./cache',
    max_memory_usage_gb=8
)
```

---

#### `comprehensive_analysis(self, data, treatment=None, outcome=None, variables=None, domain=None, covariates=None)`

Perform complete end-to-end causal analysis combining discovery and inference.

**Parameters:**
- `data` (pd.DataFrame): Input dataset for analysis.
- `treatment` (str, optional): Name of treatment/intervention column. If None, discovers relationships automatically.
- `outcome` (str, optional): Name of outcome variable column. If None, discovers relationships automatically.
- `variables` (list of str, optional): Subset of variables to analyze. If None, uses all columns.
- `domain` (str, optional): Domain context for specialized analysis ('healthcare', 'marketing', 'finance', 'education').
- `covariates` (list of str, optional): Control variables for adjustment in detailed analyses.

**Returns:**
- `ComprehensiveCausalAnalysis`: Object containing discovery results, inference results, domain recommendations, actionable insights, and confidence scores.

**Raises:**
- `ValueError`: If data is empty or contains invalid column names.
- `MemoryError`: If dataset is too large and performance optimizations are disabled.

**Example:**
```python
# Comprehensive analysis with specific treatment and outcome
analysis = causal_llm.comprehensive_analysis(
    data=df,
    treatment='campaign_spend',
    outcome='revenue',
    domain='marketing',
    covariates=['customer_segment', 'seasonality']
)

print(f"Analysis confidence: {analysis.confidence_score:.2f}")
print(f"Key insights: {analysis.actionable_insights[:3]}")

# Exploratory analysis without specifying variables
analysis = causal_llm.comprehensive_analysis(
    data=df,
    domain='healthcare'
)
```

---

#### `discover_causal_relationships(self, data, variables=None, domain=None, max_conditioning_set_size=2)`

Automatically discover causal relationships in data using statistical methods and optional domain knowledge.

**Parameters:**
- `data` (pd.DataFrame): Input dataset for causal discovery.
- `variables` (list of str, optional): Variables to include in discovery. If None, uses all numeric columns.
- `domain` (str, optional): Domain context for enhanced discovery with domain expertise.
- `max_conditioning_set_size` (int, default=2): Maximum size of conditioning sets in PC algorithm. Higher values are more thorough but slower.

**Returns:**
- `CausalDiscoveryResult`: Object containing discovered edges, suggested confounders, domain insights, and statistical summaries.

**Raises:**
- `ValueError`: If insufficient data for reliable discovery (< 50 observations recommended).
- `NumericError`: If data contains non-numeric variables that cannot be converted.

**Example:**
```python
# Basic causal discovery
discovery = causal_llm.discover_causal_relationships(
    data=clinical_data,
    variables=['age', 'treatment', 'outcome', 'comorbidities']
)

# Enhanced discovery with domain expertise
discovery = causal_llm.discover_causal_relationships(
    data=marketing_data,
    domain='marketing'
)

print(f"Discovered {len(discovery.discovered_edges)} relationships:")
for edge in discovery.discovered_edges:
    print(f"  {edge.cause} → {edge.effect} (confidence: {edge.confidence:.3f})")
```

---

#### `estimate_causal_effect(self, data, treatment, outcome, covariates=None, method='comprehensive', instrument=None)`

Estimate the causal effect of a treatment on an outcome using various statistical methods.

**Parameters:**
- `data` (pd.DataFrame): Dataset containing treatment, outcome, and covariate variables.
- `treatment` (str): Name of treatment/intervention column.
- `outcome` (str): Name of outcome variable column.
- `covariates` (list of str, optional): Control variables for confounding adjustment.
- `method` (str, default='comprehensive'): Estimation method. Options:
  - `'comprehensive'`: Multiple methods with robustness checks
  - `'regression'`: Linear regression with covariates
  - `'matching'`: Propensity score matching
  - `'iv'`: Instrumental variables (requires `instrument` parameter)
  - `'doubly_robust'`: Doubly robust estimation
- `instrument` (str, optional): Instrumental variable column name (required for IV method).

**Returns:**
- `CausalInferenceResult`: Object containing effect estimates, confidence intervals, p-values, robustness checks, and interpretation.

**Raises:**
- `ValueError`: If treatment or outcome columns not found, or invalid method specified.
- `StatisticalError`: If assumptions are severely violated or estimation fails.

**Example:**
```python
# Comprehensive causal effect estimation
effect = causal_llm.estimate_causal_effect(
    data=df,
    treatment='new_medication',
    outcome='recovery_time',
    covariates=['age', 'severity', 'comorbidities'],
    method='comprehensive'
)

print(f"Treatment effect: {effect.primary_effect.effect_estimate:.4f}")
print(f"95% CI: [{effect.primary_effect.confidence_interval[0]:.4f}, "
      f"{effect.primary_effect.confidence_interval[1]:.4f}]")
print(f"P-value: {effect.primary_effect.p_value:.6f}")

# Instrumental variables estimation
iv_effect = causal_llm.estimate_causal_effect(
    data=economic_data,
    treatment='education_years',
    outcome='income',
    instrument='college_proximity',
    method='iv'
)
```

---

#### `generate_intervention_recommendations(self, analysis, target_outcome, budget_constraint=None)`

Generate actionable intervention recommendations based on comprehensive analysis results.

**Parameters:**
- `analysis` (ComprehensiveCausalAnalysis): Results from comprehensive_analysis() method.
- `target_outcome` (str): Name of outcome variable to optimize.
- `budget_constraint` (float, optional): Maximum budget for interventions.

**Returns:**
- `dict`: Dictionary containing:
  - `'primary_interventions'`: High-impact interventions
  - `'secondary_interventions'`: Lower-priority options
  - `'expected_impacts'`: Predicted effect sizes
  - `'implementation_priority'`: Ranked intervention order
  - `'cost_effectiveness'`: ROI estimates if budget provided

**Example:**
```python
analysis = causal_llm.comprehensive_analysis(data, domain='marketing')

recommendations = causal_llm.generate_intervention_recommendations(
    analysis=analysis,
    target_outcome='customer_ltv',
    budget_constraint=100000
)

for intervention in recommendations['primary_interventions']:
    print(f"Intervention: {intervention['target_variable']}")
    print(f"Expected impact: {intervention['expected_outcome_change']:.2f}")
    print(f"Cost-effectiveness: {intervention['roi_estimate']:.1f}x")
```

---

### CausalLLM

**Basic async causal inference class with LLM integration.**

```python
from causallm import CausalLLM
```

#### `__init__(self, llm_client=None, method='hybrid', enable_logging=True, log_level='INFO')`

Initialize the basic CausalLLM instance.

**Parameters:**
- `llm_client` (object, optional): Custom LLM client instance.
- `method` (str, default='hybrid'): Analysis method ('statistical', 'llm', 'hybrid').
- `enable_logging` (bool, default=True): Enable detailed logging.
- `log_level` (str, default='INFO'): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').

#### `async discover_causal_relationships(self, data, variables, domain_context="")`

Asynchronously discover causal relationships with LLM enhancement.

**Parameters:**
- `data` (pd.DataFrame): Input data for discovery.
- `variables` (list): Variables to analyze.
- `domain_context` (str): Domain-specific context for LLM.

**Returns:**
- `CausalDiscoveryResult`: Discovered relationships and insights.

**Example:**
```python
import asyncio

async def main():
    causal_llm = CausalLLM()
    
    results = await causal_llm.discover_causal_relationships(
        data=df,
        variables=['treatment', 'outcome', 'confounders'],
        domain_context="Clinical trial for diabetes treatment"
    )
    
    return results

results = asyncio.run(main())
```

---

## Statistical Inference

### StatisticalCausalInference

**Core statistical methods for causal effect estimation.**

```python
from causallm.core.statistical_inference import StatisticalCausalInference, CausalMethod
```

#### `__init__(self, significance_level=0.05, enable_performance_optimizations=True)`

Initialize statistical inference engine.

**Parameters:**
- `significance_level` (float, default=0.05): Statistical significance threshold.
- `enable_performance_optimizations` (bool, default=True): Enable vectorized computations.

#### `estimate_causal_effect(self, data, treatment, outcome, covariates=None, method=CausalMethod.LINEAR_REGRESSION, instrument=None)`

Estimate causal effects using specific statistical methods.

**Parameters:**
- `data` (pd.DataFrame): Dataset for analysis.
- `treatment` (str): Treatment variable name.
- `outcome` (str): Outcome variable name.
- `covariates` (list, optional): Control variables.
- `method` (CausalMethod): Statistical method enum:
  - `CausalMethod.LINEAR_REGRESSION`
  - `CausalMethod.MATCHING`
  - `CausalMethod.INSTRUMENTAL_VARIABLES`
  - `CausalMethod.REGRESSION_DISCONTINUITY`
  - `CausalMethod.DIFFERENCE_IN_DIFFERENCES`
- `instrument` (str, optional): Instrumental variable for IV method.

**Returns:**
- `CausalEffect`: Statistical results with effect size, confidence intervals, and diagnostics.

**Example:**
```python
from causallm.core.statistical_inference import CausalMethod

inference = StatisticalCausalInference(significance_level=0.01)

# Linear regression estimation
effect = inference.estimate_causal_effect(
    data=df,
    treatment='policy_change',
    outcome='unemployment_rate',
    covariates=['gdp_growth', 'inflation'],
    method=CausalMethod.LINEAR_REGRESSION
)

# Propensity score matching
matching_effect = inference.estimate_causal_effect(
    data=df,
    treatment='training_program',
    outcome='job_placement',
    covariates=['age', 'education', 'experience'],
    method=CausalMethod.MATCHING
)
```

#### `comprehensive_causal_analysis(self, data, treatment, outcome, covariates=None, instrument=None)`

Run multiple estimation methods for robustness checking.

**Returns:**
- `CausalInferenceResult`: Comprehensive results with multiple estimates and robustness metrics.

---

### EnhancedCausalDiscovery

**Advanced causal structure discovery with performance optimization.**

```python
from causallm.core.enhanced_causal_discovery import EnhancedCausalDiscovery
```

#### `__init__(self, llm_client=None, significance_level=0.05, enable_performance_optimizations=True)`

Initialize enhanced causal discovery engine.

#### `discover_causal_structure(self, data, variables=None, domain=None, max_conditioning_set_size=2, use_parallel_processing=True)`

Discover causal structure using PC algorithm with enhancements.

**Parameters:**
- `data` (pd.DataFrame): Input dataset.
- `variables` (list, optional): Variables to include.
- `domain` (str, optional): Domain context.
- `max_conditioning_set_size` (int, default=2): PC algorithm parameter.
- `use_parallel_processing` (bool, default=True): Enable parallel independence testing.

**Returns:**
- `CausalDiscoveryResult`: Discovered structure and relationships.

**Example:**
```python
discovery = EnhancedCausalDiscovery(
    significance_level=0.01,
    enable_performance_optimizations=True
)

structure = discovery.discover_causal_structure(
    data=large_dataset,
    max_conditioning_set_size=3,
    use_parallel_processing=True
)

print(f"Discovered {len(structure.edges)} edges")
print(f"Graph density: {structure.graph_density:.3f}")
```

---

## Domain Packages

### HealthcareDomain

**Healthcare-specific causal analysis with medical domain expertise.**

```python
from causallm.domains.healthcare import HealthcareDomain
```

#### `__init__(self, enable_performance_optimizations=True)`

Initialize healthcare domain package.

**Parameters:**
- `enable_performance_optimizations` (bool, default=True): Enable performance features for large clinical datasets.

#### `generate_clinical_trial_data(self, n_patients, treatment_arms=None, randomization_ratio=None, include_demographics=True, include_comorbidities=True)`

Generate realistic clinical trial data with proper causal structure.

**Parameters:**
- `n_patients` (int): Number of patients to generate.
- `treatment_arms` (list, optional): Treatment group names. Default: ['control', 'treatment'].
- `randomization_ratio` (list, optional): Allocation ratios for treatments. Default: equal allocation.
- `include_demographics` (bool, default=True): Include age, gender, etc.
- `include_comorbidities` (bool, default=True): Include comorbidity indicators.

**Returns:**
- `pd.DataFrame`: Synthetic clinical trial dataset with realistic relationships.

**Example:**
```python
healthcare = HealthcareDomain()

# Generate large clinical trial
trial_data = healthcare.generate_clinical_trial_data(
    n_patients=10000,
    treatment_arms=['placebo', 'low_dose', 'high_dose'],
    randomization_ratio=[0.33, 0.33, 0.34],
    include_demographics=True,
    include_comorbidities=True
)

print(f"Generated trial with {len(trial_data)} patients")
print(f"Treatment distribution:\n{trial_data['treatment'].value_counts()}")
```

#### `analyze_treatment_effectiveness(self, data, treatment, outcome, covariates=None, safety_outcomes=None)`

Analyze treatment effectiveness with clinical interpretation.

**Parameters:**
- `data` (pd.DataFrame): Clinical trial or observational data.
- `treatment` (str): Treatment variable name.
- `outcome` (str): Primary endpoint variable.
- `covariates` (list, optional): Clinical covariates for adjustment.
- `safety_outcomes` (list, optional): Safety endpoint variables.

**Returns:**
- `HealthcareAnalysisResult`: Clinical results with medical interpretation and safety analysis.

**Example:**
```python
results = healthcare.analyze_treatment_effectiveness(
    data=trial_data,
    treatment='drug_dose',
    outcome='recovery_time',
    covariates=['age', 'baseline_severity', 'comorbidity_score'],
    safety_outcomes=['adverse_events', 'serious_adverse_events']
)

print(f"Treatment effect: {results.primary_effect.effect_estimate:.2f} days")
print(f"Clinical significance: {results.clinical_significance}")
print(f"Number needed to treat: {results.number_needed_to_treat}")
```

### InsuranceDomain

**Insurance and risk assessment domain package.**

```python
from causallm.domains.insurance import InsuranceDomain
```

#### `__init__(self, enable_performance_optimizations=True)`

Initialize insurance domain package.

#### `generate_stop_loss_data(self, n_policies, policy_types=None, industry_mix=None, include_financials=True)`

Generate realistic stop loss insurance policy data.

**Parameters:**
- `n_policies` (int): Number of policies to generate.
- `policy_types` (list, optional): Types of stop loss coverage.
- `industry_mix` (dict, optional): Industry distribution for policies.
- `include_financials` (bool, default=True): Include premium and claims data.

**Returns:**
- `pd.DataFrame`: Synthetic insurance dataset with realistic risk factors.

#### `analyze_risk_factors(self, data, risk_factor, outcome, control_variables=None)`

Analyze factors affecting insurance risk and claims.

**Parameters:**
- `data` (pd.DataFrame): Insurance policy and claims data.
- `risk_factor` (str): Risk factor to analyze (e.g., 'industry_type', 'company_size').
- `outcome` (str): Risk outcome (e.g., 'claim_amount', 'claim_frequency').
- `control_variables` (list, optional): Variables to control for in analysis.

**Returns:**
- `InsuranceRiskResult`: Risk analysis with actuarial insights and pricing recommendations.

**Example:**
```python
insurance = InsuranceDomain()

# Generate policy data
policy_data = insurance.generate_stop_loss_data(
    n_policies=50000,
    include_financials=True
)

# Analyze industry risk
risk_analysis = insurance.analyze_risk_factors(
    data=policy_data,
    risk_factor='industry_category',
    outcome='total_claim_amount',
    control_variables=['company_size', 'policy_limit', 'geographic_region']
)

print(f"High-risk industries: {risk_analysis.high_risk_categories}")
print(f"Premium adjustment recommendation: {risk_analysis.pricing_adjustments}")
```

### MarketingDomain

**Marketing attribution and campaign analysis domain package.**

```python
from causallm.domains.marketing import MarketingDomain
```

#### `__init__(self, enable_performance_optimizations=True)`

Initialize marketing domain package.

#### `generate_marketing_data(self, n_customers, n_touchpoints, attribution_window_days=30, channels=None)`

Generate realistic marketing touchpoint and conversion data.

**Parameters:**
- `n_customers` (int): Number of customers to simulate.
- `n_touchpoints` (int): Total number of marketing touchpoints.
- `attribution_window_days` (int, default=30): Attribution window for conversions.
- `channels` (list, optional): Marketing channels to include.

**Returns:**
- `pd.DataFrame`: Marketing dataset with customer journeys and conversions.

#### `analyze_attribution(self, data, model='data_driven', attribution_window=30)`

Perform marketing attribution analysis using various models.

**Parameters:**
- `data` (pd.DataFrame): Marketing touchpoint and conversion data.
- `model` (str, default='data_driven'): Attribution model:
  - `'first_touch'`: First interaction attribution
  - `'last_touch'`: Last interaction attribution
  - `'linear'`: Equal credit distribution
  - `'time_decay'`: Time-decayed attribution
  - `'position_based'`: U-shaped attribution
  - `'data_driven'`: Causal inference-based attribution
  - `'shapley'`: Shapley value attribution
- `attribution_window` (int, default=30): Days to attribute conversions.

**Returns:**
- `MarketingAttributionResult`: Attribution results with channel effectiveness and recommendations.

**Example:**
```python
marketing = MarketingDomain(enable_performance_optimizations=True)

# Generate marketing data
marketing_data = marketing.generate_marketing_data(
    n_customers=100000,
    n_touchpoints=500000,
    attribution_window_days=30
)

# Analyze attribution with data-driven model
attribution = marketing.analyze_attribution(
    data=marketing_data,
    model='data_driven'
)

print("Channel attribution:")
for channel, attribution_value in attribution.channel_attribution.items():
    print(f"  {channel}: {attribution_value:.1%}")

print(f"Model confidence: {attribution.model_confidence:.2f}")
```

---

## Performance & Processing

### AsyncCausalAnalysis

**Asynchronous processing for high-performance causal analysis.**

```python
from causallm.core.async_processing import AsyncCausalAnalysis
```

#### `__init__(self, max_concurrent_tasks=None, enable_progress_tracking=True)`

Initialize async analysis engine.

**Parameters:**
- `max_concurrent_tasks` (int, optional): Maximum parallel tasks. Defaults to CPU count.
- `enable_progress_tracking` (bool, default=True): Show progress bars for long operations.

#### `async parallel_correlation_analysis(self, data, chunk_size=5000, method='pearson')`

Compute correlation matrix using parallel processing.

**Parameters:**
- `data` (pd.DataFrame): Input data for correlation analysis.
- `chunk_size` (int, default=5000): Size of data chunks for parallel processing.
- `method` (str, default='pearson'): Correlation method ('pearson', 'spearman', 'kendall').

**Returns:**
- `pd.DataFrame`: Correlation matrix computed in parallel.

#### `async parallel_bootstrap_analysis(self, data, analysis_func, n_bootstrap=1000, confidence_level=0.95)`

Perform bootstrap analysis using parallel processing.

**Parameters:**
- `data` (pd.DataFrame): Dataset for bootstrap sampling.
- `analysis_func` (callable): Analysis function to apply to each bootstrap sample.
- `n_bootstrap` (int, default=1000): Number of bootstrap iterations.
- `confidence_level` (float, default=0.95): Confidence level for intervals.

**Returns:**
- `BootstrapResult`: Bootstrap statistics with confidence intervals.

**Example:**
```python
import asyncio
from causallm.core.async_processing import AsyncCausalAnalysis

async def run_parallel_analysis():
    async_analysis = AsyncCausalAnalysis(max_concurrent_tasks=8)
    
    # Parallel correlation analysis
    corr_matrix = await async_analysis.parallel_correlation_analysis(
        data=large_dataset,
        chunk_size=10000
    )
    
    # Parallel bootstrap for confidence intervals
    def treatment_effect(sample_data):
        return sample_data.groupby('treatment')['outcome'].mean().diff().iloc[-1]
    
    bootstrap_results = await async_analysis.parallel_bootstrap_analysis(
        data=large_dataset,
        analysis_func=treatment_effect,
        n_bootstrap=2000
    )
    
    return corr_matrix, bootstrap_results

corr, bootstrap = asyncio.run(run_parallel_analysis())
```

### DataChunker

**Memory-efficient data processing for large datasets.**

```python
from causallm.core.data_processing import DataChunker
```

#### `__init__(self, chunk_size='auto', memory_limit_gb=None, enable_progress_bar=True)`

Initialize data chunker for memory management.

**Parameters:**
- `chunk_size` (int or str, default='auto'): Chunk size or 'auto' for automatic sizing.
- `memory_limit_gb` (float, optional): Memory usage limit in GB.
- `enable_progress_bar` (bool, default=True): Show progress during chunking.

#### `chunk_dataframe(self, data, chunk_size=None)`

Split DataFrame into memory-efficient chunks.

**Parameters:**
- `data` (pd.DataFrame): Large DataFrame to chunk.
- `chunk_size` (int, optional): Override default chunk size.

**Yields:**
- `tuple`: (chunk_index, chunk_data) pairs.

**Example:**
```python
from causallm.core.data_processing import DataChunker

chunker = DataChunker(chunk_size=10000)

# Process large dataset in chunks
results = []
for chunk_idx, chunk_data in chunker.chunk_dataframe(very_large_df):
    # Process each chunk
    chunk_result = analyze_chunk(chunk_data)
    results.append(chunk_result)
    print(f"Processed chunk {chunk_idx}")

# Combine results
final_result = combine_chunk_results(results)
```

### StreamingDataProcessor

**Stream processing for datasets larger than memory.**

```python
from causallm.core.data_processing import StreamingDataProcessor
```

#### `__init__(self, buffer_size=1000, enable_parallel_processing=True)`

Initialize streaming processor.

#### `process_streaming(self, file_path, processing_func, aggregation_func=None, **kwargs)`

Process large CSV files through streaming.

**Parameters:**
- `file_path` (str): Path to large CSV file.
- `processing_func` (callable): Function to apply to each chunk.
- `aggregation_func` (callable, optional): Function to combine chunk results.
- `**kwargs`: Additional arguments for pandas.read_csv().

**Returns:**
- Results from processing and aggregation.

**Example:**
```python
from causallm.core.data_processing import StreamingDataProcessor

processor = StreamingDataProcessor()

def compute_statistics(chunk):
    return {
        'mean': chunk.mean(),
        'std': chunk.std(),
        'count': len(chunk)
    }

def combine_stats(stats_list):
    # Combine statistics from all chunks
    total_count = sum(s['count'] for s in stats_list)
    weighted_mean = sum(s['mean'] * s['count'] for s in stats_list) / total_count
    return {'overall_mean': weighted_mean, 'total_samples': total_count}

# Process huge CSV file
results = processor.process_streaming(
    "massive_dataset.csv",
    compute_statistics,
    combine_stats
)

print(f"Processed {results['total_samples']} samples")
print(f"Overall mean: {results['overall_mean']}")
```

---

## Data Models

### CausalDiscoveryResult

**Result object from causal structure discovery.**

#### Attributes:
- `discovered_edges` (list): List of CausalEdge objects representing discovered relationships.
- `suggested_confounders` (list): Variables identified as potential confounders.
- `domain_insights` (dict): Domain-specific interpretations and recommendations.
- `statistical_summary` (dict): Summary statistics from discovery process.
- `graph_density` (float): Density of the discovered causal graph.
- `confidence_scores` (dict): Confidence scores for different aspects of discovery.

#### Methods:
- `get_edges_by_confidence(self, min_confidence=0.5)`: Filter edges by confidence threshold.
- `to_networkx(self)`: Convert to NetworkX graph object.
- `plot_causal_graph(self, layout='spring')`: Visualize discovered causal structure.

### CausalInferenceResult

**Result object from causal effect estimation.**

#### Attributes:
- `primary_effect` (CausalEffect): Main effect estimate with confidence intervals.
- `robustness_checks` (list): Results from alternative estimation methods.
- `confidence_level` (str): Overall confidence assessment ('High', 'Medium', 'Low').
- `recommendations` (list): Actionable recommendations based on results.
- `assumptions_tested` (dict): Results from assumption testing.
- `sensitivity_analysis` (dict): Sensitivity analysis results.

#### Methods:
- `summary(self)`: Generate formatted summary of results.
- `plot_effects(self)`: Visualize effect estimates with confidence intervals.
- `export_results(self, format='json')`: Export results in various formats.

### ComprehensiveCausalAnalysis

**Complete analysis result combining discovery and inference.**

#### Attributes:
- `discovery_results` (CausalDiscoveryResult): Causal structure discovery findings.
- `inference_results` (dict): Detailed effect estimates for discovered relationships.
- `domain_recommendations` (list): Domain-specific advice and insights.
- `actionable_insights` (list): Prioritized list of actionable findings.
- `confidence_score` (float): Overall analysis confidence (0-1 scale).
- `performance_metrics` (dict): Performance statistics and timing information.

#### Methods:
- `get_top_relationships(self, n=5)`: Get top N causal relationships by effect size.
- `generate_report(self, format='html')`: Generate comprehensive analysis report.
- `prioritize_interventions(self, outcome_variable)`: Rank intervention opportunities.

**Example:**
```python
# Comprehensive analysis
analysis = causal_llm.comprehensive_analysis(
    data=business_data,
    domain='marketing'
)

# Access results
print(f"Analysis confidence: {analysis.confidence_score:.2f}")
print(f"Discovered {len(analysis.discovery_results.discovered_edges)} relationships")

# Get top actionable insights
top_insights = analysis.actionable_insights[:3]
for i, insight in enumerate(top_insights, 1):
    print(f"{i}. {insight}")

# Generate detailed report
report = analysis.generate_report(format='html')
with open('causal_analysis_report.html', 'w') as f:
    f.write(report)
```

---

## Usage Examples

### Complete Workflow Example

```python
from causallm import EnhancedCausalLLM
from causallm.domains.healthcare import HealthcareDomain
import pandas as pd

# 1. Initialize with performance optimizations
causal_llm = EnhancedCausalLLM(
    llm_provider='openai',
    enable_performance_optimizations=True,
    use_async=True,
    cache_dir='./analysis_cache'
)

# 2. Generate or load data
healthcare = HealthcareDomain()
clinical_data = healthcare.generate_clinical_trial_data(
    n_patients=5000,
    treatment_arms=['control', 'treatment_a', 'treatment_b']
)

# 3. Comprehensive analysis
analysis = causal_llm.comprehensive_analysis(
    data=clinical_data,
    treatment='treatment',
    outcome='recovery_time',
    domain='healthcare',
    covariates=['age', 'baseline_severity', 'comorbidities']
)

# 4. Extract insights
print(f"Treatment effect: {analysis.inference_results['treatment']['primary_effect'].effect_estimate:.2f}")
print(f"Confidence: {analysis.confidence_score:.2f}")

# 5. Generate intervention recommendations
recommendations = causal_llm.generate_intervention_recommendations(
    analysis=analysis,
    target_outcome='recovery_time'
)

for rec in recommendations['primary_interventions']:
    print(f"Recommendation: {rec['description']}")
    print(f"Expected impact: {rec['expected_impact']}")
```

### Error Handling

```python
from causallm import EnhancedCausalLLM
from causallm.exceptions import CausalLLMError, InsufficientDataError, StatisticalError

try:
    causal_llm = EnhancedCausalLLM()
    
    results = causal_llm.comprehensive_analysis(
        data=small_dataset,  # Might be too small
        treatment='intervention',
        outcome='result'
    )
    
except InsufficientDataError as e:
    print(f"Data quality issue: {e}")
    print("Recommendation: Collect more data or use simpler methods")
    
except StatisticalError as e:
    print(f"Statistical assumption violated: {e}")
    print("Recommendation: Check data distribution or try different method")
    
except CausalLLMError as e:
    print(f"General CausalLLM error: {e}")
```

---

## Performance Considerations

### Memory Management
- Use `chunk_size='auto'` for automatic memory optimization
- Enable `enable_performance_optimizations=True` for large datasets
- Set `max_memory_usage_gb` to limit memory consumption

### Parallel Processing
- Set `use_async=True` for parallel computations
- Use `AsyncCausalAnalysis` for custom parallel workflows
- Configure `max_concurrent_tasks` based on available CPU cores

### Caching
- Set `cache_dir` for persistent caching across sessions
- Use disk caching for repeated analyses on same data
- Cache automatically invalidates when data or parameters change

### Large Dataset Handling
- Use `StreamingDataProcessor` for datasets larger than memory
- Enable data chunking with appropriate `chunk_size`
- Consider domain-specific optimizations in domain packages

---

This API reference provides comprehensive documentation for all classes, methods, and functions in CausalLLM. For additional examples and tutorials, see the [Usage Examples](USAGE_EXAMPLES.md) and [Examples Directory](../examples/).