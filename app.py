import streamlit as st
from causalllm.core import CausalLLMCore
from causalllm.llm_client import OpenAIClient

# Initialize LLM client
llm_client = OpenAIClient()

# Streamlit UI
st.title("CausalLLM Demo")
task = st.text_input("Enter a causal reasoning task (e.g., 'Explain why variable X affects Y'):") 

if task:
    # Minimal example with empty context and dummy variables/edges
    engine = CausalLLMCore(context="", variables={}, dag_edges=[], llm_client=llm_client)
    
    # Generate and display prompt
    prompt = engine.generate_reasoning_prompt(task)
    
    st.subheader("Generated Prompt:")
    st.code(prompt)

    # Optional: Send to LLM and show response
    if st.button("Get LLM Response"):
        response = llm_client.chat(prompt)
        st.subheader("LLM Response:")
        st.write(response)
