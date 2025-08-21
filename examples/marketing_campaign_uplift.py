# examples/marketing_campaign_uplift.py

from causalllm.prompt_templates import PromptTemplates
from integrations.openai_adapter import query_openai

context = "An e-commerce platform wants to know the effect of using pop-up discounts."
treatment = "Pop-up discount shown on homepage"
outcome = "Increase in sales conversion"

prompt = PromptTemplates.treatment_effect_estimation(context, treatment, outcome)
response = query_openai(prompt)

print("Treatment Effect Response:")
print(response)
