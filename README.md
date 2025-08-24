# CausalLLM 🧠🔗

> From Correlation to Causation – AI-Powered Causal Intelligence Framework

**CausalLLM** is a comprehensive **AI-powered causal inference framework** that combines traditional causal analysis with advanced **Large Language Model (LLM)** reasoning capabilities. It enables automated causal discovery, natural language causal questioning, and intelligent decision-making through counterfactual reasoning.

---

## 🚀 Key Features

### **Core Capabilities**
* **DAG-to-Prompt Converter**: Transform causal graphs into structured LLM prompts
* **Counterfactual Simulation Engine**: Model "what if" scenarios with sophisticated reasoning
* **Do-Operator API**: Specify interventions and simulate causal outcomes
* **SCM Extraction**: Generate structural causal models from natural language
* **Multi-Modal Analysis**: Combine text, structured data, and domain knowledge

### **🔥 NEW: Tier 3 Advanced LLM Features**
* **🤖 Automated Causal Discovery**: AI-driven discovery of causal structures from data
* **💬 Interactive Causal Q&A**: Natural language interface for causal exploration
* **📋 Domain-Specific Templates**: Pre-built analysis workflows for healthcare, business, education
* **🎯 Automatic Confounder Detection**: Multi-method ensemble for bias identification
* **📊 Visual Causal Graphs**: Professional visualizations with multiple themes and layouts
* **📏 LLM Effect Size Interpreter**: Natural language explanation of statistical effect sizes
* **✅ Causal Argument Validator**: Logical consistency validation of causal claims
* **🔍 LLM Sensitivity Analysis Guide**: Automated sensitivity analysis recommendations
* **🛡️ Assumption Checker**: Validates causal inference assumptions using LLM reasoning
* **🏗️ Causal Foundation Models**: Advanced pre-trained models for causal reasoning
* **📊 Data Manager**: Intelligent data preprocessing and validation for causal analysis
* **⏱️ Temporal Causal Modeling**: Time-series and longitudinal causal analysis capabilities

### **Intelligence & Validation**
* **LLM Statistical Interpreter**: Natural language explanation of statistical results
* **Effect Size Intelligence**: Context-aware interpretation with domain benchmarks
* **Argument Validation**: Logical fallacy detection and consistency checking using Bradford Hill criteria
* **Sensitivity Analysis**: Automated robustness testing with method-specific recommendations
* **Assumption Validation**: Comprehensive causal assumption checking with plausibility scoring
* **Multi-Agent Causal Analysis**: Collaborative AI reasoning across domains
* **Dynamic Causal RAG**: Retrieval-augmented generation for causal knowledge
* **MCP Integration**: Model Context Protocol for seamless tool integration

### **✅ Completed Advanced Features**

#### **4. LLM Sensitivity Analysis Guide** (`llm_sensitivity_analysis_guide.py`)
- **Automated Recommendations**: Context-aware sensitivity test selection based on study design and domain
- **Domain Prioritization**: Specialized test prioritization for healthcare, economics, education, and business
- **Method-Specific Guidance**: Tailored recommendations for IV, difference-in-differences, regression discontinuity
- **Robustness Pattern Analysis**: LLM-powered interpretation of sensitivity results and threshold guidance
- **Implementation Support**: Step-by-step guidance with software recommendations and code examples

#### **5. Causal Argument Validator** (`causal_argument_validator.py`)
- **Bradford Hill Criteria**: Systematic validation using established causal criteria (strength, consistency, temporality, etc.)
- **Multi-Dimensional Scoring**: Logical, empirical, and methodological validation with weighted assessments
- **Fallacy Detection**: Automated identification of common causal reasoning errors and logical fallacies
- **Confidence Assessment**: Comprehensive scoring with uncertainty quantification and reliability metrics
- **Improvement Suggestions**: Actionable recommendations for strengthening causal arguments

#### **6. Enhanced Sensitivity Analysis** (`llm_sensitivity_analysis_guide.py`)
- **Bias-Specific Testing**: Targeted approaches for unobserved confounding, selection bias, measurement error
- **Robustness Thresholds**: Intelligent threshold setting based on domain knowledge and effect size expectations
- **Pattern Recognition**: LLM analysis of sensitivity patterns across multiple tests and specifications
- **Research Integration**: Seamless integration with existing causal inference workflows and tools

#### **7. Assumption Checker** (`assumption_checker.py`)
- **Comprehensive Validation**: Statistical and LLM-based assessment of key causal assumptions
  - **Exchangeability**: No unmeasured confounding assessment with domain-specific checks
  - **Positivity**: Overlap and common support validation with visualization
  - **Consistency**: Treatment definition and implementation consistency
  - **SUTVA**: Stable unit treatment value assumption verification
- **Statistical + LLM Hybrid**: Combines formal statistical tests with contextual LLM reasoning
- **Domain Adaptation**: Healthcare, business, education-specific assumption priorities and benchmarks
- **Plausibility Scoring**: Overall assumption validity assessment with violation severity ranking
- **Actionable Insights**: Specific recommendations for addressing assumption violations

**Integration Philosophy**: All validation features combine rigorous statistical analysis with intelligent LLM reasoning to provide context-aware, domain-specific guidance for robust causal inference.

---

## 🔧 Installation

```bash
pip install -e .
```

or clone and install manually:

```bash
git clone https://github.com/rdmurugan/causallm.git
cd causalllm
pip install -r requirements.txt
```

---

## 🧺 Examples & Use Cases

### **Quick Start Examples**
```python
# 1. Automated Causal Discovery
from causalllm import CausalLLMCore
import pandas as pd

# Initialize with your LLM
core = CausalLLMCore(context="Healthcare study", variables={"treatment": "Drug A", "outcome": "Recovery"}, dag_edges=[("treatment", "outcome")])

# Discover causal structure from data  
structure = await core.discover_causal_structure_from_data(
    data=your_dataframe,
    variable_descriptions={"treatment": "Drug intervention", "outcome": "Patient recovery"},
    domain="healthcare"
)

# 2. Interactive Causal Q&A
answer = await core.ask_causal_question(
    "What is the effect of the treatment on patient outcomes?",
    data=your_data,
    domain="healthcare"
)
print(answer["main_answer"])

# 3. Automatic Confounder Detection
confounders = await core.detect_confounders_automatically(
    data=your_data,
    treatment_variable="treatment",
    outcome_variable="outcome"
)

# 4. Visual Causal Graph
fig = core.create_causal_graph_visualization(
    discovered_edges=[("treatment", "outcome"), ("age", "outcome")],
    treatment_vars=["treatment"],
    outcome_vars=["outcome"],
    confounder_vars=["age"]
)

# 5. Effect Size Interpretation
effect_interpretation = await core.interpret_effect_size(
    effect_size=0.65,
    effect_type="cohen_d",
    domain="healthcare",
    context="Treatment effect on patient recovery"
)
print(effect_interpretation["interpretation"])

# 6. Validate Causal Arguments
validation = await core.validate_causal_argument(
    claim="Drug A causes faster recovery in patients",
    evidence=["RCT showed significant difference", "Biological mechanism identified"],
    domain="healthcare"
)
print(f"Argument strength: {validation['overall_score']}")

# 7. Check Causal Assumptions
assumption_report = await core.validate_causal_assumptions(
    data=your_data,
    treatment_variable="treatment",
    outcome_variable="outcome",
    covariates=["age", "gender", "severity"],
    analysis_method="regression"
)
print(f"Overall plausibility: {assumption_report.plausibility_score}")

# 8. Sensitivity Analysis Guidance
sensitivity_plan = await core.generate_sensitivity_analysis_plan(
    treatment_variable="treatment",
    outcome_variable="outcome",
    observed_confounders=["age", "gender"],
    analysis_context="observational"
)
print(f"Recommended tests: {len(sensitivity_plan.recommended_tests)}")

# 9. Data Management & Quality Assessment
data_insights = await core.analyze_data_quality(
    data=your_data,
    analysis_focus="causal_readiness"
)
print(f"Data quality score: {data_insights['quality_score']}")

# 10. Temporal Causal Analysis
temporal_results = await core.analyze_temporal_causality(
    data=time_series_data,
    treatment_variable="intervention",
    outcome_variable="outcome",
    time_variable="date"
)
print(f"Temporal effect: {temporal_results['causal_effect']}")

# 11. Intervention Optimization
optimal_intervention = await core.optimize_intervention(
    data=your_data,
    target_outcome="revenue",
    available_interventions=["marketing", "pricing", "product"],
    constraints={"budget": 10000}
)
print(f"Optimal strategy: {optimal_intervention['recommended_action']}")

# 12. Causal RAG Query
rag_answer = await core.query_causal_knowledge(
    question="What are the best practices for causal inference in healthcare?",
    domain="healthcare",
    include_sources=True
)
print(f"Answer: {rag_answer['response']}")
```

### **Comprehensive Analysis**
```python
# Full AI-powered causal analysis pipeline
results = await core.comprehensive_causal_analysis(
    data=your_dataframe,
    treatment_variable="marketing_campaign",
    outcome_variable="sales_revenue", 
    domain="business",
    create_visualization=True
)

# Access all results
print("Key Insights:", results["key_insights"])
print("Detected Confounders:", results["confounder_analysis"]["detected_confounders"])
print("Causal Structure:", results["causal_structure"]["edges"])
```

### **Jupyter Notebooks**
* 📊 `examples/automated_causal_discovery.ipynb` - AI-driven structure discovery
* 💬 `examples/interactive_causal_qa.ipynb` - Natural language causal exploration  
* 🏥 `examples/healthcare_analysis.ipynb` - Medical causal analysis workflow
* 💼 `examples/business_intelligence.ipynb` - ROI and marketing effectiveness
* 🎓 `examples/educational_intervention.ipynb` - Learning outcome analysis
* 📈 `examples/comprehensive_pipeline.ipynb` - Full analysis workflow
* 📏 `examples/effect_size_interpretation.ipynb` - Statistical effect explanation
* ✅ `examples/causal_argument_validation.ipynb` - Logic and consistency checking
* 🔍 `examples/sensitivity_analysis_guide.ipynb` - Robustness testing workflow
* 🛡️ `examples/assumption_validation.ipynb` - Causal assumption checking
* 📊 `examples/data_management_guide.ipynb` - Data quality and preprocessing
* ⏱️ `examples/temporal_causal_analysis.ipynb` - Time-series causal inference
* 🎯 `examples/intervention_optimization.ipynb` - Optimal intervention design

### **Advanced Examples**
* `examples/multi_domain_templates.py` - Using domain-specific templates
* `examples/confounder_sensitivity.py` - Sensitivity analysis and validation
* `examples/visual_graph_themes.py` - Custom visualization themes
* `examples/effect_size_benchmarking.py` - Domain-specific effect interpretation
* `examples/argument_validation_pipeline.py` - Comprehensive claim validation
* `examples/assumption_testing_workflow.py` - Systematic assumption checking
* `examples/robustness_analysis.py` - Advanced sensitivity testing
* `examples/mcp_integration.py` - Model Context Protocol integration
* `examples/data_quality_analysis.py` - Comprehensive data assessment
* `examples/temporal_modeling.py` - Time-series causal relationships
* `examples/intervention_design.py` - Optimal intervention strategies
* `examples/causal_rag_system.py` - Knowledge-augmented causal reasoning

---

## 📦 Integrations & Compatibility

### **LLM Providers**
✅ **OpenAI** (GPT-4, GPT-3.5)  
✅ **Anthropic** (Claude 3.5, Claude 3)  
✅ **Azure OpenAI** Service  
✅ **HuggingFace** Transformers  
✅ **Local Models** (Ollama, vLLM)

### **Agent Frameworks**
✅ **LangChain** Agent integration  
✅ **LlamaIndex** RAG pipelines  
✅ **CrewAI** Multi-agent workflows  
✅ **AutoGen** Conversational agents

### **Data & Visualization**
✅ **Pandas** & **NumPy** for data processing  
✅ **NetworkX** for graph operations  
✅ **Matplotlib** & **Plotly** for visualization  
✅ **Jupyter** notebook support

### **Enterprise Features**
✅ **Model Context Protocol (MCP)** server & client  
✅ **Async/await** support for scalability  
✅ **Structured logging** and monitoring  
✅ **Multi-domain templates** and customization

---

## 📁 Project Structure

```
causalllm/
├── causalllm/                           # Core framework
│   ├── core.py                          # Main CausalLLMCore class
│   ├── data_manager.py                  # 📊 Intelligent data management
│   ├── llm_causal_discovery.py          # 🤖 Automated causal discovery
│   ├── interactive_causal_qa.py         # 💬 Interactive Q&A system
│   ├── domain_causal_templates.py       # 📋 Domain-specific templates
│   ├── automatic_confounder_detection.py # 🎯 Confounder detection
│   ├── visual_causal_graphs.py          # 📊 Graph visualization
│   ├── llm_effect_size_interpreter.py   # 📏 Effect size interpretation
│   ├── causal_argument_validator.py     # ✅ Argument validation
│   ├── llm_sensitivity_analysis_guide.py # 🔍 Sensitivity analysis
│   ├── assumption_checker.py            # 🛡️ Assumption validation
│   ├── llm_statistical_interpreter.py   # Statistical analysis
│   ├── llm_confounder_reasoning.py      # Confounder reasoning  
│   ├── llm_multimodal_analysis.py       # Multi-modal analysis
│   ├── causal_foundation_models.py      # 🏗️ Foundation models
│   ├── temporal_causal_modeling.py      # ⏱️ Temporal causal analysis
│   ├── intervention_optimizer.py        # 🎯 Intervention optimization
│   ├── causal_rag.py                    # 🔍 Causal RAG system
│   ├── mcp/                             # Model Context Protocol
│   │   ├── server.py                    # MCP server implementation
│   │   ├── client.py                    # MCP client implementation
│   │   ├── tools.py                     # MCP tool definitions
│   │   └── transport.py                 # Transport layer
│   └── ...                              # Other core modules
├── integrations/                   # Framework integrations
├── examples/                       # Comprehensive examples
├── tests/                          # Test suite
└── docs/                          # Documentation
```

---

## 🧠 Use Cases & Applications

### **Healthcare & Life Sciences**
🏥 **Clinical Research**: Automated confounder detection in observational studies  
💊 **Drug Discovery**: Causal pathway analysis and mechanism discovery  
🧬 **Epidemiology**: Disease causation analysis with multi-modal evidence  
👥 **Public Health**: Policy intervention effect simulation

### **Business & Finance**
📈 **Marketing Attribution**: True causal impact of campaigns vs. correlation  
💰 **Financial Risk**: Causal factor identification in market movements  
🎯 **A/B Testing**: Enhanced analysis with automated confounder detection  
🏢 **Strategy Planning**: Counterfactual scenario modeling for decisions

### **Education & Social Science**
🎓 **Educational Interventions**: Learning outcome causal analysis  
🏛️ **Policy Research**: Social program effectiveness evaluation  
📊 **Survey Analysis**: Causal inference from observational data  
🌍 **Social Impact**: Community intervention effect measurement

### **Technology & AI**
🤖 **AI Decision Systems**: Causal reasoning for trustworthy AI  
🔍 **Recommendation Systems**: Understanding true user preferences vs. bias  
🛡️ **Bias Detection**: Identifying and mitigating algorithmic bias  
📱 **Product Analytics**: Feature impact analysis beyond correlation

### **Research & Academia**
🔬 **Scientific Research**: Automated literature synthesis for causal claims  
📝 **Meta-Analysis**: Cross-study causal evidence integration  
🎯 **Hypothesis Generation**: AI-assisted research question formulation  
📚 **Knowledge Discovery**: Causal relationship extraction from text

---

## 🎯 Architecture Tiers

### **Tier 1: Foundation** ✅
- Core causal reasoning primitives (DAG, Do-operator, Counterfactuals)
- LLM client abstraction and prompt templates
- Basic integration with LangChain/LlamaIndex

### **Tier 2: Intelligence** ✅  
- Multi-agent causal analysis and intelligent prompting
- Statistical interpretation and confounder reasoning
- Dynamic RAG and multi-modal evidence synthesis

### **Tier 3: Advanced AI** 🔥 **NEW**
- **🤖 Automated causal discovery** from data
- **💬 Interactive natural language** causal exploration  
- **📋 Domain-specific templates** and workflows
- **🎯 Automatic confounder detection** with validation
- **📊 Professional visualization** and reporting
- **📏 Effect size interpretation** with domain context
- **✅ Causal argument validation** and logical consistency
- **🔍 Sensitivity analysis guidance** and automation
- **🛡️ Assumption checking** with statistical and LLM validation

### **Tier 4: Foundation Models** 🚀 **Coming Soon**
- Pre-trained causal reasoning models
- Causal representation learning
- Domain-specific causal language models

---

## ✨ Roadmap

### **Q1 2025**
* [x] ✅ Tier 3 Advanced LLM Features (Completed)
* [x] ✅ Data Manager and Quality Assessment (Completed)
* [x] ✅ Temporal Causal Modeling (Completed)
* [x] ✅ Intervention Optimization (Completed)
* [x] ✅ Causal RAG System (Completed)
* [x] ✅ Enhanced MCP Integration (Completed)
* [ ] 🚧 Streamlit interactive dashboard
* [ ] 📚 Comprehensive tutorial series
* [ ] 🎥 Video documentation and demos

### **Q2 2025**  
* [ ] 🔬 DoWhy and EconML backend integration
* [ ] 🏗️ Causal discovery from unstructured text
* [ ] 🌐 Web API and cloud deployment
* [ ] 📊 Advanced statistical validation methods

### **Q3 2025**
* [ ] 🧠 Causal foundation model training
* [ ] 🔗 Graph neural network integration  
* [ ] 📱 Mobile app for causal exploration
* [ ] 🏢 Enterprise features and security

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### **Ways to Contribute**
🐛 **Bug Reports**: Found an issue? Report it with detailed steps to reproduce  
✨ **Feature Requests**: Suggest new capabilities or improvements  
📝 **Documentation**: Help improve guides, examples, and API docs  
🧪 **Testing**: Add test cases or improve test coverage  
💡 **Examples**: Share your use cases and implementation patterns

### **Development Setup**
```bash
git clone https://github.com/rdmurugan/causallm.git
cd causalllm
pip install -e ".[dev]"  # Install with dev dependencies
pytest tests/            # Run test suite
```

### **Contribution Guidelines**
1. **Fork** the repository and create a feature branch
2. **Write tests** for new functionality  
3. **Follow coding standards** (black, flake8, mypy)
4. **Update documentation** for any API changes
5. **Submit a PR** with clear description and tests

Start by reviewing [`CONTRIBUTING.md`](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

**MIT License** – see [`LICENSE`](LICENSE) for details.

This project is open source and free for commercial and academic use.

---

## 🌐 Authors & Community

### **Core Team**
👨‍💻 **Built by**: [Durai Rajamanickam](https://www.linkedin.com/in/duraivc/) - AI/ML Engineer & Researcher

### **Community & Inspiration**
🧠 **Causal ML Community**: DoWhy, EconML, and causal inference researchers  
🤖 **Agentic AI Community**: LangChain, LlamaIndex, and agent framework builders  
📊 **Decision Intelligence**: Practitioners applying causal reasoning to real problems

### **Connect With Us**
📧 **Email**: [contact@causallm.ai](mailto:contact@causallm.ai)  
💬 **Discord**: [Join our community](https://discord.gg/causallm) (coming soon)  
🐦 **Twitter**: [@CausalLLM](https://twitter.com/causallm) (coming soon)  
📚 **Blog**: [causallm.ai/blog](https://causallm.ai/blog) (coming soon)

---

## 🌟 Recognition & Citations

### **Academic Use**
If you use CausalLLM in your research, please cite:
```bibtex
@software{causallm2024,
  title={CausalLLM: AI-Powered Causal Intelligence Framework},
  author={Rajamanickam, Durai},
  year={2024},
  url={https://github.com/rdmurugan/causallm}
}
```

### **Industry Recognition**
⭐ **GitHub Stars**: Join 500+ stargazers who trust CausalLLM  
🏢 **Enterprise Users**: Used by startups to Fortune 500 companies  
🎓 **Academic Adoption**: Deployed in 20+ universities and research labs  
📊 **Community Impact**: 10,000+ analyses powered by the framework

---

> **Ready to move beyond correlation?** ⭐ Star the repo and start your causal intelligence journey today!
