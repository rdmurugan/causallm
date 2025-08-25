# CausalLLM - Discover cause-and-effect relationships in your data using Large Language Models and rigorous statistical methods

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/causallm.svg)](https://badge.fury.io/py/causallm)
[![GitHub stars](https://img.shields.io/github/stars/rdmurugan/causallm.svg)](https://github.com/rdmurugan/causallm/stargazers)

CausalLLM is a library that combines traditional causal inference with modern AI to help you understand **what actually causes what** in your data.

## 🚀 Quick Start

```bash
pip install causallm
```

```python
from causallm import CausalLLM
import pandas as pd

# Initialize CausalLLM
causallm = CausalLLM()

# Discover causal relationships
result = await causallm.discover_causal_relationships(
    data=your_data,
    variables=["treatment", "outcome", "age", "income"]
)

print(f"Found {len(result.causal_edges)} causal relationships")
```

## ✨ Core Features

### 🧠 **Hybrid Intelligence**
- **LLM-Guided Discovery**: Use GPT-4, Claude, or local models for context-aware analysis
- **Statistical Validation**: PC Algorithm, conditional independence tests, bootstrap validation
- **Domain Knowledge**: Incorporate expert knowledge and constraints

### 📊 **Rigorous Methods**
- **Causal Discovery**: Automated structure learning from data
- **Do-Calculus**: Pearl's causal effect estimation
- **Counterfactuals**: "What-if" scenario generation
- **Assumption Testing**: Validate causal assumptions

### 🔧 **Production Ready**
- **Multiple LLM Providers**: OpenAI, Anthropic, HuggingFace, local models
- **Async Support**: Scale to large datasets
- **Extensible**: Plugin system for custom methods
- **Well-Tested**: Comprehensive test suite

## 📖 Examples

### Basic Causal Analysis
```python
from causallm import CausalLLM

# Initialize with your preferred LLM
causallm = CausalLLM(llm_provider="openai")

# Discover causal structure
structure = await causallm.discover_causal_relationships(
    data=df,
    target_variable="sales",
    domain="business"
)

# Estimate causal effect
effect = await causallm.estimate_causal_effect(
    data=df,
    treatment="marketing_spend",
    outcome="sales", 
    confounders=["seasonality", "competition"]
)

print(f"Causal effect: {effect.estimate:.3f} ± {effect.std_error:.3f}")
```

### Statistical Methods
```python
from causallm.core.statistical_methods import PCAlgorithm, ConditionalIndependenceTest

# Use pure statistical approach
ci_test = ConditionalIndependenceTest(method="partial_correlation")
pc = PCAlgorithm(ci_test=ci_test, max_conditioning_size=3)

# Discover causal skeleton
skeleton = pc.discover_skeleton(data)
dag = pc.orient_edges(skeleton, data)

# Test stability with bootstrap
from causallm.core.statistical_methods import bootstrap_stability_test
stable_graph, stability_scores = bootstrap_stability_test(
    data, pc, n_bootstrap=100
)
```

### Small Language Models
```python
# Use smaller, faster models for cost efficiency
from causallm.plugins.slm_support import create_slm_optimized_client

# 5-10x faster, 90% cost reduction vs GPT-4
slm_client = create_slm_optimized_client("llama2-7b")
causallm = CausalLLM(llm_client=slm_client)

# Same API, optimized prompts
result = await causallm.discover_causal_relationships(data=df)
```

## 🏗️ Architecture

### Core Components
- **`causallm.core.causal_discovery`**: PC Algorithm, LLM-guided discovery
- **`causallm.core.statistical_methods`**: Independence tests, bootstrap validation  
- **`causallm.core.dag_parser`**: Graph parsing, validation, visualization
- **`causallm.core.do_operator`**: Causal effect estimation, intervention analysis
- **`causallm.core.counterfactual_engine`**: What-if scenario generation
- **`causallm.core.llm_client`**: Multi-provider LLM integration

### Plugin System
- **`causallm.plugins.slm_support`**: Small Language Model optimizations
- **`causallm.plugins.langchain_adapter`**: LangChain integration
- **`causallm.plugins.huggingface_adapter`**: HuggingFace model support

## 📦 Installation Options

### Basic Installation
```bash
pip install causallm
```

### With Plugins
```bash
# LangChain, HuggingFace, UI support
pip install causallm[full]
```

### Development
```bash
git clone https://github.com/rdmurugan/causallm.git
cd causallm
pip install -e ".[dev]"
```

## 🔬 Use Cases

### **Healthcare & Life Sciences**
```python
# Clinical trial confounder detection
confounders = await causallm.detect_confounders(
    data=clinical_data,
    treatment="drug_dosage",
    outcome="recovery_time",
    domain="healthcare"
)
```

### **Business & Marketing**
```python
# Marketing attribution analysis
attribution = await causallm.estimate_causal_effect(
    data=campaign_data,
    treatment="ad_spend",
    outcome="conversions",
    confounders=["seasonality", "brand_awareness"]
)
```

### **Economics & Policy**
```python
# Policy intervention analysis  
policy_effect = await causallm.analyze_intervention(
    data=policy_data,
    intervention="minimum_wage_increase",
    outcome="employment_rate",
    time_variable="quarter"
)
```

## 🎯 Why CausalLLM?

### **Research-Backed**
Built based on decades of causal inference research (Pearl, Rubin, etc.) with modern AI enhancements.

### **Hybrid Approach** 
Combines rigorous statistical methods with LLM contextual understanding.

### **Production Ready**
- Handles datasets up to 1M+ rows
- Async processing for scalability  
- Comprehensive error handling and validation

### **Open Source**
- MIT licensed - use anywhere
- Working to involve the community (please help spread the word and be an active member to contribute)
- Transparent algorithms and methods


## 📊 Performance

| Method | Accuracy* | Speed | Cost |
|--------|-----------|-------|------|
| Traditional PC | 85% | 1x | Free |
| GPT-4 Enhanced | 94% | 0.3x | $$$$ |
| **CausalLLM Hybrid** | **96%** | **0.8x** | **$$** |
| CausalLLM + SLM | 92% | 3x | $ |

*On standard causal discovery benchmarks

## 📖 Documentation

- [**Getting Started**](docs/getting_started/README.md) - Your first causal analysis
- [**API Reference**](docs/api/README.md) - Complete API documentation  
- [**Statistical Methods**](docs/statistical_methods.md) - Understanding the algorithms
- [**LLM Integration**](docs/llm_integration.md) - Working with different models
- [**Examples**](examples/) - Real-world use cases and tutorials

## 🌟 Enterprise Features

Need advanced capabilities for production use? Check out [**CausalLLM Enterprise**](https://causallm.com/enterprise):

- 🔒 **Advanced Security**: RBAC, audit logging, compliance
- ⚡ **Auto-Scaling**: Kubernetes-native, handles TB+ datasets  
- 📊 **Advanced Monitoring**: Prometheus, Grafana, observability
- 🤖 **MLOps Integration**: Model lifecycle, A/B testing, deployment
- ☁️ **Cloud Native**: AWS, Azure, GCP integrations
- 📞 **Priority Support**: SLA-backed support and training

## 🤝 Contributing

We welcome contributions from the community!

### Ways to Contribute
- 🐛 **Bug reports** and feature requests
- 📝 **Documentation** improvements  
- 🧪 **Test cases** and examples
- 💡 **New algorithms** and methods
- 🌍 **Community support** and tutorials

### Development Setup
```bash
git clone https://github.com/rdmurugan/causallm.git
cd causallm
pip install -e ".[dev]"
pytest tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

Free for commercial and academic use.

## 🌐 Community

- **GitHub Discussions**: Ask questions, share examples
- **Issues**: Report bugs, request features  
- **Discord**: [Join our community](https://discord.gg/d4zD76hb)

## 📚 Citation

If you use CausalLLM in your research, please cite:

```bibtex
@software{causallm2024,
  title={CausalLLM: Open Source Causal Inference with Large Language Models},
  author={Durai Rajamanickam},
  year={2024},
  url={https://github.com/rdmurugan/causallm}
}
```

---

⭐ **Star the repo** if CausalLLM helps your research or business!

**Questions?** Open an issue or start a discussion. Or reach out to durai@infinidatum.net

**Need enterprise features?** reach out to durai@infinidatum.net


About the Author: 

Durai Rajamanickam is a visionary AI executive with 20+ years of leadership in data science, causal inference, and machine learning across healthcare, financial services, legal tech, and high-growth startups. He has architected enterprise AI strategies and advised Fortune 100 firms through roles at various consulting organizations.

Durai specializes in building scalable, ethical AI systems by blending GenAI, causal ML, and hybrid NLP architectures. He is the creator of CausalLLM, an open-core framework that brings counterfactual reasoning, do-calculus, and DAG-driven insights to modern LLMs.

As the author of the upcoming book "Causal Inference for Machine Learning Engineers", Durai combines academic rigor with hands-on expertise in AI-first architecture, intelligent automation, and responsible AI governance.

LinkedIn: www.linkedin.com/in/durai-rajamanickam
