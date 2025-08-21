from causalllm.core import CausalLLMCore
from causalllm.llm_client import BaseLLMClient

# Mocking a real-like LLM response
class MockLLMClient(BaseLLMClient):
    def chat(self, prompt: str, model: str = "gpt-4", temperature: float = 0.7) -> str:
        if "counterfactual" in prompt:
            return "If A had been true, then B would have changed."
        elif "why" in prompt.lower():
            return "Because B is caused by A."
        return "Simulated reasoning output."

def test_causal_llm_real_use_case():
    # Define a simple DAG and context
    context = "We are studying a causal system involving exercise (A), sleep (B), and productivity (C)."
    variables = {
        "A": "exercise",
        "B": "sleep",
        "C": "productivity"
    }
    dag_edges = [("A", "B"), ("B", "C")]
    llm_client = MockLLMClient()

    # Initialize CausalLLMCore
    engine = CausalLLMCore(context, variables, dag_edges, llm_client)

    # Step 1: Simulate do
    do_output = engine.simulate_do({"A": "yes"}, "What if I start exercising?")
    print("\nDO Output:\n", do_output)

    # Step 2: Simulate counterfactual
    cf_output = engine.simulate_counterfactual(
        factual="I didn't exercise and felt tired.",
        intervention="I did exercise.",
        instruction="How might my sleep have changed?"
    )
    print("\nCounterfactual Output:\n", cf_output)

    # Step 3: Generate reasoning prompt
    reasoning_output = engine.generate_reasoning_prompt("Explain the impact of sleep on productivity.")
    print("\nReasoning Prompt:\n", reasoning_output)

    # Basic validations
    assert isinstance(do_output, str)
    assert isinstance(cf_output, str)
    assert isinstance(reasoning_output, str)
