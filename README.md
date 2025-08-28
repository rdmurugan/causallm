# CausalLLM: Open Source Causal Inference Library

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/causallm.svg)](https://badge.fury.io/py/causallm)
[![GitHub stars](https://img.shields.io/github/stars/rdmurugan/causallm.svg)](https://github.com/rdmurugan/causallm/stargazers)
[![Downloads](https://img.shields.io/pypi/dm/causallm.svg)](https://pypi.org/project/causallm/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://github.com/rdmurugan/causallm/blob/main/COMPLETE_USER_GUIDE.md)

CausalLLM is a Python library for causal inference and analysis that combines statistical methods with domain-specific knowledge. It provides tools for discovering causal relationships in data and estimating treatment effects using multiple statistical approaches.

## Key Features

- **Statistical Causal Inference**: Linear regression, propensity score matching, and instrumental variables
- **Causal Structure Discovery**: PC algorithm implementation for discovering relationships from data
- **Domain-Specific Packages**: Pre-built components for healthcare and insurance analysis
- **Multiple LLM Integrations**: Optional integration with OpenAI, Anthropic, and other providers
- **Comprehensive Analysis**: Combined discovery and inference workflows

## Installation

```bash
# Install from PyPI
pip install causallm

# Install with all dependencies
pip install causallm[full]
```

## Quick Start

### Basic Usage

```python
from causallm import EnhancedCausalLLM
import pandas as pd

# Initialize (works without API keys using statistical methods)
causal_llm = EnhancedCausalLLM()

# Load your data
data = pd.read_csv("your_data.csv")

# Perform causal analysis
results = causal_llm.comprehensive_analysis(
    data=data,
    treatment='treatment_variable',
    outcome='outcome_variable',
    covariates=['age', 'gender', 'other_variables']
)

print(f"Effect estimate: {results.inference_results}")
```

### Domain-Specific Analysis

#### Healthcare Example

```python
from causallm import HealthcareDomain, EnhancedCausalLLM

# Use healthcare domain package
healthcare = HealthcareDomain()

# Generate sample clinical trial data
clinical_data = healthcare.generate_clinical_trial_data(
    n_patients=500,
    treatment_arms=['control', 'treatment']
)

# Get medical confounders
confounders = healthcare.get_medical_confounders(
    treatment='treatment',
    outcome='recovery_time',
    available_vars=list(clinical_data.columns)
)

# Run treatment effectiveness analysis
causal_llm = EnhancedCausalLLM()
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

# Generate stop loss insurance data
policy_data = insurance.generate_stop_loss_data(n_policies=1000)

# Analyze risk factors
risk_results = insurance.analyze_risk_factors(
    data=policy_data,
    risk_factor='industry',
    outcome='total_claim_amount'
)
```

## Core Components

### EnhancedCausalLLM
Main class for comprehensive causal analysis combining statistical methods with optional LLM enhancement.

### Statistical Methods
- **Linear Regression**: Covariate adjustment for causal effect estimation
- **Propensity Score Matching**: Addresses selection bias in observational data
- **Instrumental Variables**: Handles unobserved confounding
- **PC Algorithm**: Statistical method for causal structure discovery

### Domain Packages
Pre-configured components for specific industries:

- **Healthcare Domain**: Clinical trial data generation, medical confounders, treatment effectiveness analysis
- **Insurance Domain**: Risk assessment, premium optimization, claims analysis
- **Marketing Domain**: Coming soon
- **Education Domain**: Coming soon
- **Experimentation Domain**: Coming soon

## Available Examples

The `examples/` directory contains working demonstrations:

- `healthcare_analysis_openai.py` - Clinical treatment effectiveness analysis
- `stop_loss_insurance_analysis.py` - Insurance risk assessment
- `ab_testing_enhanced_demo.py` - A/B test causal analysis
- `domain_packages_demo.py` - Domain packages demonstration
- `enhanced_causallm_demo.py` - Comprehensive analysis example

## LLM Integration (Optional)

CausalLLM can work with various LLM providers for enhanced analysis:

```python
import os

# Set API keys (optional)
os.environ['OPENAI_API_KEY'] = 'your-openai-key'
os.environ['ANTHROPIC_API_KEY'] = 'your-anthropic-key'

# Initialize with LLM support
causal_llm = EnhancedCausalLLM(
    llm_provider="openai",  # or "anthropic", "llama", etc.
    llm_model="gpt-4"
)
```

**Note**: The library works fully without API keys using statistical methods only.

## Statistical Rigor

CausalLLM implements established statistical methods:

- **Assumption Testing**: Validates causal inference assumptions
- **Robustness Checks**: Cross-validates results across methods
- **Confidence Intervals**: Provides uncertainty quantification
- **P-value Calculation**: Statistical significance testing
- **Effect Size Interpretation**: Practical significance assessment

## Documentation

- **[Complete User Guide](COMPLETE_USER_GUIDE.md)**: Comprehensive documentation with API reference
- **[Domain Packages Guide](DOMAIN_PACKAGES.md)**: Industry-specific components documentation
- **[Usage Examples](USAGE_EXAMPLES.md)**: Real-world use cases and scenarios
- **[Examples Directory](examples/)**: Working code demonstrations

## Requirements

- Python 3.9+
- pandas, numpy, scipy
- scikit-learn
- Optional: openai, anthropic (for LLM features)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key areas where contributions are needed:
- Additional domain packages (marketing, finance, education)
- More statistical methods
- Performance optimizations
- Documentation improvements

## Citation

If you use CausalLLM in your research, please cite:

```bibtex
@software{causallm2024,
  title={CausalLLM: Open Source Causal Inference Library},
  author={Durai Rajamanickam},
  year={2024},
  version={3.0.0},
  url={https://github.com/rdmurugan/causallm}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/rdmurugan/causallm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rdmurugan/causallm/discussions)
- **Email**: durai@infinidatum.net

---

## About

CausalLLM is developed and maintained by Durai Rajamanickam, with contributions from the open source community. The library aims to make causal inference more accessible while maintaining statistical rigor.

**LinkedIn**: [linkedin.com/in/durai-rajamanickam](https://www.linkedin.com/in/durai-rajamanickam)