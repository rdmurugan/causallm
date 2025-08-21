# integrations/openai_adapter.py

import openai

def query_openai(prompt: str, model: str = "gpt-4", temperature: float = 0.7) -> str:
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message["content"].strip()
