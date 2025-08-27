import pandas as pd
import numpy as np
from causallm import CausalLLM
import asyncio
import os

# Set dummy credentials to avoid API errors
os.environ['OPENAI_API_KEY'] ="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"
os.environ['OPENAI_PROJECT_ID'] ="proj_wnehjGDtcLnMQrfSvI2CBNBJ"

# Create synthetic sample data
np.random.seed(42)

N = 100
age = np.random.randint(20, 70, size=N)
income = np.random.normal(50000, 15000, size=N)
treatment = (age > 40).astype(int)  # simple age-based treatment assignment
outcome = treatment * 5 + income * 0.0003 + np.random.normal(0, 1, size=N)

# Create DataFrame
your_data = pd.DataFrame({
    "age": age,
    "income": income,
    "treatment": treatment,
    "outcome": outcome
})

# Initialize CausalLLM
causallm = CausalLLM()

# Async causal discovery
async def main():
    result = await causallm.discover_causal_relationships(
        data=your_data,
        variables=["treatment", "outcome", "age", "income"]
    )
    print (result)
    print("Discovered Edges:")
    for edge in result.discovered_edges:
        print(f"{edge.cause} --> {edge.effect}  (confidence: {edge.confidence:.2f})")

# Run the analysis
asyncio.run(main())
