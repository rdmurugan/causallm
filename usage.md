# Getting Started with CausalLLM

This guide shows how to use **CausalLLM** to build causal prompts, simulate counterfactuals, and integrate with agents.

---

## 🔧 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/causal-llm.git
cd causal-llm
pip install -r requirements.txt
```

---

## 🧠 Simulate a Counterfactual

```python
from causalllm.counterfactual_engine import CounterfactualEngine

engine = CounterfactualEngine()

response = engine.simulate_counterfactual(
    context="A patient with diabetes was not taking medication.",
    factual="The patient skipped doses and had high blood sugar levels.",
    intervention="The patient followed a strict medication routine."
)

print(response)
```

---

## ⚙️ Run a Do-Operator Intervention

```python
from causalllm.do_operator import DoOperatorSimulator

context = "The user interface was basic and conversion was low."
variables = {"UI": "basic", "Conversion": "low"}

sim = DoOperatorSimulator(context, variables)

prompt = sim.generate_do_prompt({"UI": "enhanced"})
print(prompt)
```

---

## 📚 Use a Prompt Template

```python
from causalllm.prompt_templates import PromptTemplates

prompt = PromptTemplates.treatment_effect_estimation(
    context="We launched a promotional banner.",
    treatment="Showing promo banner",
    outcome="Click-through rate"
)

print(prompt)
```

---

## 🤖 Run as LangChain Agent

```python
from integrations.langchain_agent import CausalAgent

agent = CausalAgent()
agent.run("Simulate a counterfactual: What if the customer was offered a 20% discount?")
```

---

## 📓 Explore Notebooks

Navigate to the `examples/` folder and try:

* `treatment_effect_simulation.ipynb`
* `counterfactual_storytelling.ipynb`

---

## ✅ Test the Code

```bash
python -m unittest discover tests/
```

---

For advanced usage, see `docs/architecture.md` or join the discussion on GitHub.
