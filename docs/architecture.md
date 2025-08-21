# CausalLLM Architecture Overview

## 🧠 Objective

CausalLLM integrates causal inference methods with large language models (LLMs) to enable AI systems that can reason about *why* something happened, simulate *what if* scenarios, and support decision-making with *do-calculus*-inspired logic.

---

## 🧱 Core Components

### 1. `dag_parser.py`

* Converts user-defined causal DAGs (Directed Acyclic Graphs) into topological sequences
* Generates stepwise causal reasoning prompts

### 2. `counterfactual_engine.py`

* Uses an LLM to simulate counterfactuals by comparing factual and hypothetical narratives
* Takes in context, intervention, and provides counterfactual output

### 3. `prompt_templates.py`

* Houses reusable prompt structures for treatment effect estimation, counterfactuals, and chain-of-thought reasoning

### 4. `do_operator.py`

* Applies intervention logic (`do(X=x)`) to base contexts
* Modifies scenarios to reflect direct causal manipulation

### 5. `scm_explainer.py`

* Uses LLM to extract causal relationships from text
* Outputs edges (A → B) that can populate the DAGParser

### 6. `utils.py`

* Utility methods for JSON/YAML config loading, saving, etc.

---

## 🤖 Integration Layer

### `integrations/langchain_agent.py`

* Wraps the toolkit as LangChain-compatible agent
* Exposes tools for counterfactual and do-operator reasoning via zero-shot prompting

### `integrations/llamaindex_plugin.py`

* Placeholder for future integration with LlamaIndex graph RAG systems

### `integrations/openai_adapter.py`

* Provides a unified API to query OpenAI-compatible LLMs

---

## 📁 Project Structure

```
causal-llm/
├── causalllm/              # Core logic
├── integrations/           # LangChain / LLM adapters
├── examples/               # Demonstrations and notebooks
├── tests/                  # Unit tests
├── docs/                   # Architecture and specs
```

---

## 🔄 Data & Workflow Flow

```
1. Natural Language Scenario
        ↓
2. SCM Extraction (edges) ←─────┐
        ↓                       │
3. DAGParser → Reasoning Order │
        ↓                       │
4a. PromptTemplate              │
        ↓                       │
5a. Query LLM for explanation  │
                               ↓
4b. DoOperator or CounterfactualEngine
        ↓
5b. Generate Interventional Prompt → Query LLM
```

---

## 🛠️ Extensibility

* Add new prompt templates to `prompt_templates.py`
* Add domain-specific DAGs or SCM extractors
* Replace or enhance LLM backends with Anthropic, Mistral, Claude, etc.
* Wrap Streamlit or CLI interface for exploration

---

## 📍 Notes

* LLM reliability for causal reasoning is still experimental; use alongside statistical tools.
* Counterfactuals should include disclaimers when used for decision-making or clinical insights.
* Modular design enables integration with other AI/ML pipelines.

---

For deeper theory, refer to "Causal Inference for Machine Learning Engineers" by Durai Rajamanickam.
