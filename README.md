# CausalLLM 🧠🔗

> From Correlation to Causation – Equip Your LLM with Counterfactual Reasoning.

**CausalLLM** is an open-source framework for integrating **causal inference** into **Large Language Model (LLM)** workflows. It enables language agents to reason with causal models, simulate counterfactuals, and answer "what if" and "why" questions.

---

## 🚀 Features

* **DAG-to-Prompt Converter**: Turn causal graphs into structured prompts for LLMs
* **Counterfactual Simulation Engine**: Model “what if” scenarios with text or tabular data
* **Do-Operator API**: Specify interventions and simulate outcomes
* **Causal Chain-of-Thought**: Structured reasoning templates for LLM-based answers
* **Plug-and-Play Agents**: LangChain and LlamaIndex integrations included
* **SCM Extraction**: Generate structural causal models from natural language

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

## 🧺 Examples

* `examples/treatment_effect_simulation.ipynb`
  Simulate causal treatment effects from tabular data using an LLM

* `examples/counterfactual_storytelling.ipynb`
  Generate plausible “what could have happened” narratives using GPT

* `examples/marketing_campaign_uplift.py`
  Estimate uplift from interventions in marketing scenarios

---

## 📦 Integrations

✅ LangChain Agent
✅ LlamaIndex Retriever
✅ OpenAI, Anthropic, HuggingFace-compatible

---

## 📁 Project Structure

```
causal-llm/
├── causalllm/               # Core modules
├── integrations/            # LangChain, LlamaIndex, model APIs
├── examples/                # Use cases and notebooks
├── tests/                   # Unit and integration tests
└── docs/                    # Technical documentation
```

---

## 🧠 Use Cases

* AI agents for **decision support**
* Causal-aware **retrieval-augmented generation**
* Business users asking "why" something happened
* Model robustness via **intervention simulation**
* Storytelling with counterfactual narratives

---

## ✨ Roadmap

* [ ] Add SCM-based prompting module
* [ ] Integrate DoWhy and EconML as backends
* [ ] Add causal discovery from unstructured input
* [ ] Release Streamlit-based DAG builder

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Start by reviewing [`CONTRIBUTING.md`](CONTRIBUTING.md) (coming soon).

---

## 📄 License

MIT License – see `LICENSE` for details.

---

## 🌐 Authors & Community

Built by [Durai Rajamanickam](https://www.linkedin.com/in/duraivc/)
Inspiration from the **causal ML**, **agentic AI**, and **decision intelligence** communities.

---

> If you’re using this for research, education, or enterprise, please ⭐ star the repo and share your use cases with us!
