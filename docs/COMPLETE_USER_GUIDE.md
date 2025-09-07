# CausalLLM Complete User Guide

*Comprehensive documentation for new users with all call options and parameters*

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation & Setup](#installation--setup) 
3. [Core Concepts](#core-concepts)
4. [Complete API Reference](#complete-api-reference)
5. [Usage Examples](#usage-examples)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 30-Second Example
```python
from causallm import EnhancedCausalLLM
import pandas as pd

# Initialize
causal_llm = EnhancedCausalLLM()

# Load your data
data = pd.read_csv("your_data.csv")

# One-line comprehensive analysis
results = causal_llm.comprehensive_analysis(
    data=data,
    domain='healthcare'  # or 'marketing', 'finance', etc.
)

print(f"Discovered {len(results.discovery_results.discovered_edges)} relationships")
print(f"Confidence: {results.confidence_score:.2f}")
```

---

## Installation & Setup

### Basic Installation
```bash
# Latest version
pip install causallm

# With all optional dependencies
pip install causallm[full]
```

### Environment Setup
```python
import os

# Set API keys (optional - enables LLM features)
os.environ['OPENAI_API_KEY'] = 'your-key-here'
os.environ['ANTHROPIC_API_KEY'] = 'your-key-here'  # Optional

from causallm import EnhancedCausalLLM
causal_llm = EnhancedCausalLLM()
```

---

## Core Concepts

### Main Classes

1. **EnhancedCausalLLM** - Primary interface for comprehensive causal analysis
2. **CausalLLM** - Basic interface (legacy support)
3. **EnhancedCausalDiscovery** - Specialized causal structure discovery
4. **StatisticalCausalInference** - Statistical causal effect estimation

### Key Workflows

1. **Discovery First**: Explore causal structure → Analyze specific relationships
2. **Targeted Analysis**: Direct analysis of known treatment-outcome pairs
3. **Comprehensive**: Combined discovery + detailed inference in one call

---

## Complete API Reference

### EnhancedCausalLLM (Primary Class)

#### Initialization
```python
EnhancedCausalLLM(
    llm_provider: str = "openai",      # "openai", "llama", "grok", "mcp"
    llm_model: str = "gpt-4",          # Model name
    significance_level: float = 0.05   # Statistical significance level
)
```

**Parameters:**
- `llm_provider`: LLM provider to use. Falls back to statistical-only if unavailable
- `llm_model`: Specific model (e.g., "gpt-4", "gpt-3.5-turbo", "claude-3")
- `significance_level`: Threshold for statistical significance testing

#### Method: discover_causal_relationships()
```python
discover_causal_relationships(
    data: pd.DataFrame,           # Required: Input dataset
    variables: List[str] = None,  # Optional: Variables to analyze 
    domain: str = None           # Optional: Domain context
) -> CausalDiscoveryResult
```

**Purpose**: Automatically discover causal relationships in your data

**Parameters:**
- `data`: Pandas DataFrame with your dataset
- `variables`: List of column names to analyze. If None, uses all columns
- `domain`: Domain for specialized insights ('healthcare', 'marketing', 'finance', 'education', 'policy')

**Returns**: CausalDiscoveryResult containing:
- `discovered_edges`: List of found causal relationships
- `suggested_confounders`: Potential confounding variables
- `domain_insights`: Domain-specific interpretations
- `statistical_summary`: Summary statistics

**Example:**
```python
discovery = causal_llm.discover_causal_relationships(
    data=df,
    variables=['age', 'treatment', 'outcome', 'income'],
    domain='healthcare'
)

for edge in discovery.discovered_edges:
    print(f"{edge.cause} → {edge.effect} (confidence: {edge.confidence:.3f})")
```

#### Method: estimate_causal_effect()
```python
estimate_causal_effect(
    data: pd.DataFrame,                    # Required: Dataset
    treatment: str,                        # Required: Treatment variable
    outcome: str,                         # Required: Outcome variable
    covariates: List[str] = None,         # Optional: Control variables
    method: str = "comprehensive",         # Method selection
    instrument: str = None               # Optional: Instrumental variable
) -> CausalInferenceResult
```

**Purpose**: Estimate the causal effect of a treatment on an outcome

**Parameters:**
- `data`: Your dataset
- `treatment`: Name of treatment/intervention column
- `outcome`: Name of outcome variable column
- `covariates`: List of control variables to adjust for
- `method`: Statistical method to use:
  - `"comprehensive"`: Multiple methods with robustness checks
  - `"regression"`: Linear regression with covariates
  - `"matching"`: Propensity score matching
  - `"iv"`: Instrumental variables estimation
- `instrument`: Column name for instrumental variable (required for IV method)

**Returns**: CausalInferenceResult containing:
- `primary_effect`: Main effect estimate with confidence intervals
- `robustness_checks`: Results from alternative methods
- `confidence_level`: Overall confidence ("High", "Medium", "Low")
- `recommendations`: Actionable recommendations

**Example:**
```python
effect = causal_llm.estimate_causal_effect(
    data=df,
    treatment='new_drug',
    outcome='recovery_rate',
    covariates=['age', 'severity', 'comorbidities'],
    method='comprehensive'
)

print(f"Effect: {effect.primary_effect.effect_estimate:.4f}")
print(f"95% CI: {effect.primary_effect.confidence_interval}")
print(f"P-value: {effect.primary_effect.p_value:.6f}")
```

#### Method: comprehensive_analysis()
```python
comprehensive_analysis(
    data: pd.DataFrame,              # Required: Dataset
    treatment: str = None,           # Optional: Specific treatment
    outcome: str = None,            # Optional: Specific outcome
    variables: List[str] = None,     # Optional: Variables to include
    domain: str = None,             # Optional: Domain context
    covariates: List[str] = None    # Optional: Control variables
) -> ComprehensiveCausalAnalysis
```

**Purpose**: Complete end-to-end causal analysis combining discovery and inference

**Parameters:**
- `data`: Your dataset
- `treatment`: Specific treatment to analyze (if None, analyzes top discovered relationships)
- `outcome`: Specific outcome to analyze (if None, analyzes top discovered relationships) 
- `variables`: Subset of variables to focus on (if None, uses all)
- `domain`: Domain for specialized insights
- `covariates`: Control variables for detailed analyses

**Returns**: ComprehensiveCausalAnalysis containing:
- `discovery_results`: Causal structure findings
- `inference_results`: Detailed effect estimates
- `domain_recommendations`: Domain-specific advice
- `actionable_insights`: List of actionable findings
- `confidence_score`: Overall analysis confidence (0-1)

**Example:**
```python
analysis = causal_llm.comprehensive_analysis(
    data=marketing_data,
    treatment='campaign_intensity',
    outcome='customer_ltv',
    domain='marketing',
    covariates=['age', 'income', 'previous_purchases']
)

print(f"Overall confidence: {analysis.confidence_score:.2f}")
for insight in analysis.actionable_insights[:3]:
    print(f"• {insight}")
```

#### Method: generate_intervention_recommendations()
```python
generate_intervention_recommendations(
    analysis: ComprehensiveCausalAnalysis,  # Required: Analysis results
    target_outcome: str,                    # Required: Desired outcome
    budget_constraint: float = None         # Optional: Budget limit
) -> Dict[str, Any]
```

**Purpose**: Generate specific intervention recommendations based on analysis

**Parameters:**
- `analysis`: Results from comprehensive_analysis()
- `target_outcome`: The outcome variable you want to improve
- `budget_constraint`: Maximum budget for interventions (optional)

**Returns**: Dictionary with:
- `primary_interventions`: Top-priority interventions
- `secondary_interventions`: Lower-priority options
- `expected_impacts`: Predicted effect sizes
- `implementation_priority`: Ranked order of interventions

**Example:**
```python
interventions = causal_llm.generate_intervention_recommendations(
    analysis=analysis,
    target_outcome='customer_ltv',
    budget_constraint=50000.0
)

for intervention in interventions['primary_interventions']:
    print(f"Target: {intervention['target_variable']}")
    print(f"Expected impact: {intervention['expected_outcome_change']}")
```

---

### CausalLLM (Basic Class)

#### Initialization
```python
CausalLLM(
    llm_client=None,           # Optional: Custom LLM client
    method="hybrid",           # Method selection
    enable_logging=True,       # Enable logging
    log_level="INFO"          # Logging level
)
```

#### Methods
All methods are async and require `await`:

```python
# Causal discovery
results = await causal_llm.discover_causal_relationships(
    data, variables, domain_context=""
)

# Effect estimation  
effect = await causal_llm.estimate_causal_effect(
    data, treatment, outcome
)

# Counterfactuals
counterfactuals = await causal_llm.generate_counterfactuals(
    data, intervention
)
```

---

### EnhancedCausalDiscovery (Specialized Discovery)

#### Initialization
```python
from causallm.core.enhanced_causal_discovery import EnhancedCausalDiscovery

discovery = EnhancedCausalDiscovery(
    llm_client=None,                  # Optional: LLM client
    significance_level: float = 0.05  # Statistical significance
)
```

#### Method: discover_causal_structure()
```python
discover_causal_structure(
    data: pd.DataFrame,                        # Required: Dataset
    variables: List[str] = None,              # Optional: Variables to analyze
    domain: str = None,                       # Optional: Domain context
    max_conditioning_set_size: int = 2        # Optional: PC algorithm parameter
) -> CausalDiscoveryResult
```

**Advanced Parameters:**
- `max_conditioning_set_size`: Maximum number of variables in conditioning sets for PC algorithm. Higher values are more thorough but slower.

---

### StatisticalCausalInference (Specialized Inference)

#### Initialization
```python
from causallm.core.statistical_inference import StatisticalCausalInference

inference = StatisticalCausalInference(
    significance_level: float = 0.05
)
```

#### Method: estimate_causal_effect()
```python
from causallm.core.statistical_inference import CausalMethod

estimate_causal_effect(
    data: pd.DataFrame,                           # Required: Dataset
    treatment: str,                               # Required: Treatment variable
    outcome: str,                                # Required: Outcome variable
    covariates: List[str] = None,                # Optional: Control variables
    method: CausalMethod = CausalMethod.LINEAR_REGRESSION,  # Method selection
    instrument: str = None                       # Optional: Instrumental variable
) -> CausalEffect
```

**Available Methods:**
- `CausalMethod.LINEAR_REGRESSION`: Standard regression with covariates
- `CausalMethod.MATCHING`: Propensity score matching
- `CausalMethod.INSTRUMENTAL_VARIABLES`: Two-stage least squares
- `CausalMethod.REGRESSION_DISCONTINUITY`: RDD (if applicable)
- `CausalMethod.DIFFERENCE_IN_DIFFERENCES`: DiD (if applicable)

#### Method: comprehensive_causal_analysis()
```python
comprehensive_causal_analysis(
    data: pd.DataFrame,          # Required: Dataset  
    treatment: str,             # Required: Treatment variable
    outcome: str,              # Required: Outcome variable
    covariates: List[str] = None, # Optional: Control variables
    instrument: str = None      # Optional: Instrumental variable
) -> CausalInferenceResult
```

---

## Usage Examples

### Example 1: Marketing Attribution Analysis

```python
import pandas as pd
from causallm import EnhancedCausalLLM

# Load marketing data
data = pd.read_csv('marketing_data.csv')

# Initialize CausalLLM
causal_llm = EnhancedCausalLLM(domain_expertise='marketing')

# Comprehensive marketing analysis
analysis = causal_llm.comprehensive_analysis(
    data=data,
    treatment='email_campaign',
    outcome='customer_ltv',
    domain='marketing',
    covariates=['customer_segment', 'seasonality']
)

# Display key findings
print(f"Email campaign effect: ${analysis.primary_effect.effect_estimate:.2f}")
print(f"ROI confidence: {analysis.confidence_score:.2f}")

# Get intervention recommendations
interventions = causal_llm.generate_intervention_recommendations(
    analysis, target_outcome='customer_ltv'
)
```

### Example 2: Healthcare Treatment Analysis

```python
# Clinical trial data analysis
causal_llm = EnhancedCausalLLM(domain_expertise='healthcare')

# Analyze treatment effectiveness
effect = causal_llm.estimate_causal_effect(
    data=clinical_data,
    treatment='new_drug',
    outcome='recovery_rate', 
    covariates=['age', 'severity', 'comorbidities'],
    method='comprehensive'
)

print(f"Treatment effect: {effect.primary_effect.effect_estimate:.4f}")
print(f"95% CI: [{effect.primary_effect.confidence_interval[0]:.4f}, "
      f"{effect.primary_effect.confidence_interval[1]:.4f}]")
print(f"Clinical significance: {effect.primary_effect.interpretation}")
```

### Example 3: Financial Risk Analysis

```python
# Financial data causal analysis
causal_llm = EnhancedCausalLLM(domain_expertise='finance')

# Discover risk factors
discovery = causal_llm.discover_causal_relationships(
    data=financial_data,
    variables=['market_volatility', 'interest_rates', 'portfolio_returns'],
    domain='finance'
)

# Analyze specific relationships
for edge in discovery.discovered_edges:
    if edge.confidence > 0.8:
        effect = causal_llm.estimate_causal_effect(
            data=financial_data,
            treatment=edge.cause,
            outcome=edge.effect,
            method='comprehensive'
        )
        print(f"{edge.cause} → {edge.effect}: {effect.primary_effect.effect_estimate:.4f}")
```

### Example 4: A/B Test Enhancement

```python
# Transform simple A/B test into causal analysis
ab_analysis = causal_llm.comprehensive_analysis(
    data=ab_test_data,
    treatment='variant',
    outcome='conversion_rate',
    domain='marketing',
    covariates=['user_segment', 'device_type', 'traffic_source']
)

print(f"Conversion lift: {ab_analysis.primary_effect.effect_estimate:.1%}")
print(f"Statistical power: {ab_analysis.confidence_score:.2f}")
```

---

## Advanced Features

### Custom LLM Configuration

```python
# Using different LLM providers
causal_llm_openai = EnhancedCausalLLM(
    llm_provider="openai", 
    llm_model="gpt-4"
)

causal_llm_anthropic = EnhancedCausalLLM(
    llm_provider="anthropic",
    llm_model="claude-3-opus"
)

# Statistical-only mode (no LLM)
causal_llm_stats = EnhancedCausalLLM(
    llm_provider=None  # Falls back to statistical methods only
)
```

### Domain-Specific Analysis

```python
# Healthcare analysis with medical expertise
healthcare_analysis = causal_llm.comprehensive_analysis(
    data=patient_data,
    domain='healthcare'  # Activates medical knowledge
)

# Marketing analysis with business expertise  
marketing_analysis = causal_llm.comprehensive_analysis(
    data=customer_data,
    domain='marketing'  # Activates marketing knowledge
)

# Finance analysis with economic expertise
finance_analysis = causal_llm.comprehensive_analysis(
    data=market_data, 
    domain='finance'  # Activates financial knowledge
)
```

### Batch Analysis

```python
# Analyze multiple treatment-outcome pairs
treatments = ['email', 'social_media', 'search_ads']
outcome = 'conversion_rate'

results = {}
for treatment in treatments:
    results[treatment] = causal_llm.estimate_causal_effect(
        data=marketing_data,
        treatment=treatment,
        outcome=outcome,
        method='comprehensive'
    )

# Compare effect sizes
for treatment, effect in results.items():
    print(f"{treatment}: {effect.primary_effect.effect_estimate:.4f}")
```

### Sensitivity Analysis

```python
# Comprehensive analysis includes sensitivity testing
analysis = causal_llm.comprehensive_analysis(data=data)

# Check robustness across methods
for relationship, inference in analysis.inference_results.items():
    print(f"\n{relationship}:")
    print(f"Primary: {inference.primary_effect.effect_estimate:.4f}")
    
    for i, check in enumerate(inference.robustness_checks):
        print(f"Method {i+1}: {check.effect_estimate:.4f}")
```

---

## Best Practices

### Data Preparation

```python
# 1. Clean your data
data = data.dropna()  # Handle missing values
data = data[data['outcome'] > 0]  # Remove invalid values

# 2. Check data types
print(data.dtypes)
data['treatment'] = data['treatment'].astype('category')

# 3. Validate sample size
print(f"Sample size: {len(data):,}")
# Minimum recommended: 100+ observations
```

### Variable Selection

```python
# Include relevant variables only
relevant_vars = [
    'treatment_var',      # Your intervention
    'outcome_var',       # Your outcome of interest  
    'confounder1',       # Important control variables
    'confounder2',       # Demographics, pre-treatment characteristics
    # Avoid post-treatment variables (colliders)
]

analysis = causal_llm.comprehensive_analysis(
    data=data,
    variables=relevant_vars
)
```

### Interpretation Guidelines

```python
# 1. Check confidence levels
if analysis.confidence_score > 0.8:
    print("High confidence - results are reliable")
elif analysis.confidence_score > 0.6:
    print("Medium confidence - interpret with caution")
else:
    print("Low confidence - need more data or different approach")

# 2. Examine effect sizes
effect = analysis.primary_effect.effect_estimate
if abs(effect) > 0.5:
    print("Large effect size - practically significant")
elif abs(effect) > 0.2:
    print("Medium effect size")
else:
    print("Small effect size - may not be practically significant")

# 3. Check assumptions
if analysis.primary_effect.assumptions_violated:
    print("Warning: Key assumptions violated:")
    for violation in analysis.primary_effect.assumptions_violated:
        print(f"- {violation}")
```

### Domain-Specific Best Practices

#### Healthcare
```python
# Include clinical variables
healthcare_covariates = [
    'age', 'gender', 'comorbidities', 
    'severity_score', 'previous_treatments'
]

# Check for clinical significance
if effect.primary_effect.effect_estimate > 0.1:  # Example threshold
    print("Clinically significant improvement")
```

#### Marketing  
```python
# Include customer characteristics
marketing_covariates = [
    'customer_segment', 'lifetime_value', 
    'acquisition_channel', 'seasonality'
]

# Calculate ROI
effect_size = analysis.primary_effect.effect_estimate
roi = (effect_size * customer_base) / campaign_cost
print(f"Estimated ROI: {roi:.1%}")
```

#### Finance
```python
# Include market factors
finance_covariates = [
    'market_volatility', 'interest_rates',
    'sector', 'market_cap'
]

# Check for regime changes
if 'regime_change' in analysis.discovery_results.assumptions_violated:
    print("Warning: Model may not be stable across time periods")
```

---

## Troubleshooting

### Common Errors and Solutions

#### 1. "Small sample size warning"
```python
# Error: Sample size < 50
# Solution: Collect more data or use simpler methods
if len(data) < 100:
    # Use basic regression instead of complex methods
    effect = causal_llm.estimate_causal_effect(
        data=data,
        treatment=treatment,
        outcome=outcome, 
        method='regression'  # Simpler method
    )
```

#### 2. "Variable not found in data"
```python
# Error: KeyError for variable names
# Solution: Check column names
print("Available columns:", data.columns.tolist())

# Standardize column names
data.columns = data.columns.str.lower().str.replace(' ', '_')
```

#### 3. "No causal relationships discovered"
```python
# Solution: Adjust discovery parameters
discovery = causal_llm.discover_causal_relationships(
    data=data,
    variables=key_variables,  # Focus on key variables only
)

# Or use lower significance threshold
causal_llm = EnhancedCausalLLM(significance_level=0.10)
```

#### 4. "LLM client initialization failed"
```python
# Solution: Works without LLM (statistical methods only)
causal_llm = EnhancedCausalLLM()  # Falls back gracefully
# Or set API key
import os
os.environ['OPENAI_API_KEY'] = 'your-key-here'
```

### Performance Optimization

```python
# For large datasets (>10,000 rows)
# 1. Sample your data first
sample_data = data.sample(n=5000, random_state=42)

# 2. Focus on key variables
key_vars = ['treatment', 'outcome', 'top_confounders']

# 3. Use simpler methods for initial exploration
quick_analysis = causal_llm.estimate_causal_effect(
    data=sample_data,
    treatment='treatment',
    outcome='outcome',
    method='regression'  # Faster than 'comprehensive'
)
```

### Validation Checklist

Before trusting your results:

```python
# ✓ Data quality
assert len(data) >= 100, "Need larger sample size"
assert not data.isnull().any().any(), "Handle missing values"

# ✓ Variable relationships
correlation_matrix = data.corr()
print("High correlations (>0.8):")
print(correlation_matrix[correlation_matrix > 0.8])

# ✓ Effect size interpretation
effect = analysis.primary_effect.effect_estimate
assert analysis.confidence_score > 0.6, "Low confidence results"
assert analysis.primary_effect.p_value < 0.05, "Not statistically significant"

# ✓ Robustness
if len(analysis.robustness_checks) > 0:
    effects = [analysis.primary_effect.effect_estimate] + \
              [check.effect_estimate for check in analysis.robustness_checks]
    effect_consistency = np.std(effects) / np.mean(effects)
    assert effect_consistency < 0.3, "Results not consistent across methods"
```

---

## Support and Resources

### Getting Help

```python
# Check package info
import causallm
print(f"Version: {causallm.__version__}")
print(f"Author: {causallm.__author__}")

# Enterprise support available
enterprise_info = causallm.CausalLLM().get_enterprise_info()
print(enterprise_info)
```

### Community Resources

- **GitHub Issues**: [Report bugs](https://github.com/rdmurugan/causallm/issues)
- **Documentation**: [Latest docs](https://causallm.readthedocs.io)
- **Email Support**: durai@infinidatum.net
- **LinkedIn**: [Durai Rajamanickam](https://www.linkedin.com/in/durai-rajamanickam)

### Citation

If you use CausalLLM in research:

```bibtex
@software{causallm2024,
  title={CausalLLM: Enhanced Causal Inference with Large Language Models},
  author={Durai Rajamanickam},
  year={2024},
  version={latest},
  url={https://github.com/rdmurugan/causallm}
}
```

---

*This guide covers all major features and call options. For the latest updates, check the [GitHub repository](https://github.com/rdmurugan/causallm).*