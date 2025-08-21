# demo.py

from causalllm.core import CausalLLMCore
from causalllm.llm_client import get_llm_client

# === CONFIGURE ===
context = """
A hospital is studying the effect of a new drug on patient recovery. 
Patients are assigned either the new drug or a placebo.
"""

variables = {
    "Drug": "Patients receiving the new drug",
    "Placebo": "Patients receiving a placebo",
    "Recovery": "Observed patient recovery outcome"
}

dag_edges = [
    ("Drug", "Recovery"),
    ("Placebo", "Recovery")
]

llm_provider = "openai"  # Options: "openai", "llama", "groq"
llm_model = "gpt-4"      # Or other model like "llama2", "mistral" depending on provider
llm_client = get_llm_client(provider=llm_provider, model=llm_model)

# === INITIALIZE CAUSAL ENGINE ===
causal_engine = CausalLLMCore(
    context=context,
    variables=variables,
    dag_edges=dag_edges,
    llm_client=llm_client,
)

# === RUN DO-CALCULUS ===
print("\n=== Do-Intervention Prompt ===")
do_prompt = causal_engine.simulate_do({"Drug": "NoDrug"}, question="How would recovery change?")
print(do_prompt)

# === RUN COUNTERFACTUAL ===
print("\n=== Counterfactual Reasoning ===")
cf_response = causal_engine.simulate_counterfactual(
    factual="The patient received the drug and recovered.",
    intervention="The patient had not received the drug.",
    instruction="Explain the most likely outcome."
)
print(cf_response)

# === SCM REASONING ===
print("\n=== SCM Reasoning Prompt ===")
scm_prompt = causal_engine.generate_reasoning_prompt(task="Explain the sequence of causal influences.")
print(scm_prompt)
