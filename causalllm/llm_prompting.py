"""
Advanced LLM Prompting System for Causal Analysis.

This module provides intelligent prompt engineering capabilities including:
- Few-shot learning with curated examples
- Chain-of-thought reasoning templates  
- Self-consistency through multiple reasoning paths
- Dynamic prompt optimization and A/B testing
- Domain-specific prompt adaptation
"""

import json
import random
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
from difflib import SequenceMatcher
import hashlib
from causalllm.logging import get_logger, get_structured_logger


@dataclass
class CausalExample:
    """Structure for few-shot learning examples."""
    context: str
    factual: str
    intervention: str
    analysis: str
    reasoning_steps: List[str]
    domain: str
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        pass  # metadata now initialized by default_factory


@dataclass
class PromptTemplate:
    """Advanced prompt template with reasoning structure."""
    name: str
    description: str
    template: str
    reasoning_steps: List[str]
    domain: str = "general"
    task_type: str = "counterfactual"
    variables: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        pass  # variables now initialized by default_factory


class CausalPromptEngine:
    """
    Advanced prompt engineering system for causal analysis.
    
    Provides few-shot learning, chain-of-thought reasoning,
    self-consistency checks, and domain adaptation.
    """
    
    def __init__(self, examples_db_path: Optional[str] = None):
        """
        Initialize prompt engineering system.
        
        Args:
            examples_db_path: Optional path to examples database file
        """
        self.logger = get_logger("causalllm.llm_prompting")
        self.struct_logger = get_structured_logger("llm_prompting")
        
        self.logger.info("Initializing CausalPromptEngine")
        
        # Example databases
        self.examples_db: Dict[str, List[CausalExample]] = {}
        self.templates: Dict[str, PromptTemplate] = {}
        
        # Performance tracking
        self.prompt_performance: Dict[str, Dict[str, Any]] = {}
        
        # Load examples and templates
        self._initialize_builtin_examples()
        self._initialize_builtin_templates()
        
        if examples_db_path and Path(examples_db_path).exists():
            self.load_examples_db(examples_db_path)
        
        self.logger.info(f"CausalPromptEngine initialized with {len(self.examples_db)} example sets and {len(self.templates)} templates")
    
    def _initialize_builtin_examples(self) -> None:
        """Initialize built-in high-quality examples for few-shot learning."""
        
        # Healthcare examples
        healthcare_examples = [
            CausalExample(
                context="Clinical trial comparing new diabetes medication vs standard treatment in 500 patients over 6 months",
                factual="Patient received standard metformin treatment, HbA1c level was 8.5% at 6 months",
                intervention="Patient received new GLP-1 agonist instead of metformin",
                analysis="Based on clinical trial data, the new GLP-1 agonist shows an average 1.2% greater reduction in HbA1c compared to metformin (p<0.001). Given the patient's baseline characteristics, we would expect their HbA1c to be approximately 7.3% instead of 8.5%. However, this assumes similar adherence rates and no differential side effects that might affect compliance.",
                reasoning_steps=[
                    "Identify the causal mechanism: GLP-1 agonists improve insulin sensitivity and reduce glucose production",
                    "Check for confounders: Patient baseline characteristics, adherence, comorbidities",
                    "Apply treatment effect: 1.2% average additional HbA1c reduction based on RCT data",
                    "Consider individual variation: Effect may vary ±0.3% based on patient characteristics",
                    "Assess assumptions: Assumes similar side effect profile and adherence"
                ],
                domain="healthcare",
                quality_score=0.95
            ),
            CausalExample(
                context="Analysis of surgical vs medical treatment for coronary artery disease using observational data from 10,000 patients",
                factual="High-risk patient received medical treatment only, survived 5 years post-diagnosis",
                intervention="Same patient received coronary bypass surgery",
                analysis="For high-risk patients with multi-vessel disease, bypass surgery shows 15-20% improved 10-year survival compared to medical treatment alone. However, this patient survived 5 years with medical treatment, suggesting better-than-average response. With surgery, we'd expect extended survival (likely 8-12 years total) but with immediate surgical risk (2-3% mortality). The counterfactual depends critically on precise risk stratification and assumes surgical candidacy.",
                reasoning_steps=[
                    "Assess baseline risk: Multi-vessel disease typically carries high mortality risk",
                    "Identify treatment mechanism: Bypass improves blood flow, reduces cardiac events", 
                    "Account for selection bias: Surgical patients may be healthier/different",
                    "Apply survival benefit: 15-20% relative risk reduction for high-risk patients",
                    "Consider surgical risks: Immediate 2-3% operative mortality risk"
                ],
                domain="healthcare",
                quality_score=0.92
            )
        ]
        
        # Marketing examples
        marketing_examples = [
            CausalExample(
                context="E-commerce company A/B testing personalized product recommendations vs generic recommendations across 50,000 customers",
                factual="Customer saw generic recommendations, purchased $120 worth of products in 30 days",
                intervention="Customer saw personalized recommendations based on browsing history and demographics",
                analysis="Personalized recommendations show 18% higher conversion rates and 25% higher average order value in our A/B tests. For this customer's purchase behavior ($120 baseline), personalized recommendations would likely increase their spending to approximately $150 (25% uplift). This assumes similar engagement with personalized content and no adverse reaction to perceived invasiveness of personalization.",
                reasoning_steps=[
                    "Establish baseline behavior: $120 in 30 days represents moderate engagement",
                    "Identify mechanism: Personalization increases relevance and purchase intent",
                    "Apply measured effect sizes: 18% conversion uplift, 25% AOV increase",
                    "Consider customer heterogeneity: Effect varies by customer segment and preferences",
                    "Check assumptions: Assumes positive reception of personalized experience"
                ],
                domain="marketing",
                quality_score=0.90
            )
        ]
        
        # Finance examples  
        finance_examples = [
            CausalExample(
                context="Credit risk analysis using machine learning model trained on 100,000 loan applications with 3-year follow-up",
                factual="Applicant with 680 credit score, $50k income, 15% debt-to-income ratio was approved and defaulted within 2 years",
                intervention="Same applicant was rejected for the loan",
                analysis="This applicant had borderline risk characteristics - our model shows 22% default probability for this profile. By rejecting the loan, we avoid the default loss but also forgo potential profit. The counterfactual analysis shows the bank would save the loss amount (estimated $12,000 average loss) but lose potential interest income (approximately $3,200 over 2 years). Net impact: avoiding $8,800 in expected loss per similar rejection.",
                reasoning_steps=[
                    "Calculate default probability: 22% based on credit score, income, DTI ratio",
                    "Estimate loss given default: Historical average $12,000 per defaulted loan", 
                    "Calculate foregone profit: Interest income over loan term ($3,200)",
                    "Compute expected value: 0.22 × $12,000 - 0.78 × $3,200 = net expected loss",
                    "Consider portfolio effects: Impact on overall risk and profitability"
                ],
                domain="finance",
                quality_score=0.88
            )
        ]
        
        self.examples_db = {
            "healthcare": healthcare_examples,
            "marketing": marketing_examples,
            "finance": finance_examples,
            "general": healthcare_examples[:1] + marketing_examples[:1]  # Best examples for general use
        }
        
        self.logger.info(f"Loaded {sum(len(examples) for examples in self.examples_db.values())} built-in examples")
    
    def _initialize_builtin_templates(self) -> None:
        """Initialize built-in prompt templates with chain-of-thought reasoning."""
        
        counterfactual_template = PromptTemplate(
            name="counterfactual_cot",
            description="Chain-of-thought counterfactual analysis template",
            template="""You are an expert in causal inference. Analyze this counterfactual scenario using systematic step-by-step reasoning.

{few_shot_examples}

Now analyze this new scenario:

CONTEXT: {context}
FACTUAL SCENARIO: {factual}
COUNTERFACTUAL INTERVENTION: {intervention}

Use this structured reasoning approach:

STEP 1 - IDENTIFY CAUSAL MECHANISM:
- What is the primary causal pathway from the intervention to the outcome?
- Are there intermediate variables or mediators?
- What is the theoretical basis for this causal relationship?

STEP 2 - ASSESS BASELINE CONDITIONS:
- What were the key characteristics of the factual scenario?
- What factors contributed to the original outcome?
- Are there any special circumstances to consider?

STEP 3 - APPLY THE INTERVENTION:
- How would the intervention change the causal mechanism?
- What direct effects would we expect?
- What is the magnitude of the expected change?

STEP 4 - TRACE DOWNSTREAM EFFECTS:
- What indirect or cascading effects might occur?
- Are there unintended consequences to consider?
- How might other variables be affected?

STEP 5 - QUANTIFY UNCERTAINTY:
- What are the main sources of uncertainty in this analysis?
- What assumptions are critical to the conclusion?
- How confident should we be in this prediction?

STEP 6 - VALIDATE ASSUMPTIONS:
- Are there potential confounders or alternative explanations?
- What would need to be true for this analysis to be valid?
- What additional data would strengthen the conclusion?

ANALYSIS:""",
            reasoning_steps=[
                "Identify causal mechanism",
                "Assess baseline conditions", 
                "Apply the intervention",
                "Trace downstream effects",
                "Quantify uncertainty",
                "Validate assumptions"
            ],
            task_type="counterfactual",
            variables=["context", "factual", "intervention", "few_shot_examples"]
        )
        
        treatment_effect_template = PromptTemplate(
            name="treatment_effect_cot",
            description="Treatment effect estimation with systematic reasoning",
            template="""You are a causal inference specialist. Estimate the treatment effect using rigorous step-by-step analysis.

{few_shot_examples}

RESEARCH QUESTION: What is the causal effect of {treatment} on {outcome}?
CONTEXT: {context}

Follow this systematic approach:

STEP 1 - DEFINE THE CAUSAL ESTIMAND:
- What exactly are we trying to measure?
- Who is the target population?
- What is the specific intervention and comparison?

STEP 2 - IDENTIFY POTENTIAL CONFOUNDERS:
- What variables might affect both treatment assignment and outcomes?
- Are there common causes we need to control for?
- What about unmeasured confounders?

STEP 3 - ASSESS IDENTIFICATION STRATEGY:
- Can we identify the causal effect from available data?
- What assumptions are required (exchangeability, positivity, consistency)?
- Are there instrumental variables or natural experiments?

STEP 4 - ESTIMATE THE EFFECT:
- What is the most appropriate estimation method?
- What is the expected magnitude and direction of the effect?
- Are there effect modifiers or heterogeneous effects?

STEP 5 - EVALUATE ROBUSTNESS:
- How sensitive are results to key assumptions?
- What would change if unmeasured confounding exists?
- Are there alternative explanations for the findings?

STEP 6 - INTERPRET RESULTS:
- What is the practical significance of the estimated effect?
- How generalizable are these findings?
- What are the policy or decision implications?

ANALYSIS:""",
            reasoning_steps=[
                "Define the causal estimand",
                "Identify potential confounders",
                "Assess identification strategy",
                "Estimate the effect", 
                "Evaluate robustness",
                "Interpret results"
            ],
            task_type="treatment_effect",
            variables=["treatment", "outcome", "context", "few_shot_examples"]
        )
        
        do_calculus_template = PromptTemplate(
            name="do_calculus_cot", 
            description="Do-calculus intervention analysis template",
            template="""You are an expert in causal inference and do-calculus. Analyze this intervention scenario systematically.

{few_shot_examples}

INTERVENTION ANALYSIS:
Context: {context}
Variables: {variables}
Proposed Intervention: do({intervention})
Question: {question}

Use structured do-calculus reasoning:

STEP 1 - REPRESENT THE CAUSAL MODEL:
- What is the causal DAG structure?
- What are the key causal relationships?
- Are there any unobserved confounders?

STEP 2 - FORMALIZE THE INTERVENTION:
- What exactly does do({intervention}) mean?
- Which causal arrows are cut by this intervention?
- How does this change the causal structure?

STEP 3 - APPLY DO-CALCULUS RULES:
- Can we identify P(Y|do(X)) from observational data?
- Which backdoor paths need to be blocked?
- Are there any confounders we need to adjust for?

STEP 4 - PREDICT INTERVENTION EFFECTS:
- What is the expected change in the outcome?
- Are there ripple effects on other variables?
- What is the mechanism of the causal effect?

STEP 5 - CONSIDER PRACTICAL CONSTRAINTS:
- Is this intervention feasible to implement?
- What are the costs and benefits?
- Are there ethical considerations?

STEP 6 - QUANTIFY UNCERTAINTY:
- What are the main sources of uncertainty?
- How robust is this analysis to model misspecification?
- What additional data would reduce uncertainty?

ANALYSIS:""",
            reasoning_steps=[
                "Represent the causal model",
                "Formalize the intervention", 
                "Apply do-calculus rules",
                "Predict intervention effects",
                "Consider practical constraints",
                "Quantify uncertainty"
            ],
            task_type="do_calculus",
            variables=["context", "variables", "intervention", "question", "few_shot_examples"]
        )
        
        self.templates = {
            "counterfactual_cot": counterfactual_template,
            "treatment_effect_cot": treatment_effect_template,
            "do_calculus_cot": do_calculus_template
        }
        
        self.logger.info(f"Loaded {len(self.templates)} built-in prompt templates")
    
    def get_few_shot_examples(self, 
                             task_type: str = "counterfactual",
                             domain: str = "general", 
                             n_examples: int = 2,
                             min_quality: float = 0.8) -> str:
        """
        Get few-shot examples for prompt enhancement.
        
        Args:
            task_type: Type of causal analysis task
            domain: Domain for examples (healthcare, marketing, finance, general)
            n_examples: Number of examples to include
            min_quality: Minimum quality score for examples
            
        Returns:
            Formatted few-shot examples string
        """
        self.logger.debug(f"Getting {n_examples} few-shot examples for {domain} {task_type}")
        
        # Get examples from the specified domain
        domain_examples = self.examples_db.get(domain, self.examples_db.get("general", []))
        
        # Filter by quality and shuffle
        quality_examples = [ex for ex in domain_examples if ex.quality_score >= min_quality]
        
        if len(quality_examples) < n_examples:
            self.logger.warning(f"Only {len(quality_examples)} quality examples available for {domain}, requested {n_examples}")
            # Fall back to general examples if needed
            if domain != "general":
                general_examples = self.examples_db.get("general", [])
                quality_examples.extend([ex for ex in general_examples if ex.quality_score >= min_quality])
        
        # Select best examples
        selected_examples = sorted(quality_examples, key=lambda x: x.quality_score, reverse=True)[:n_examples]
        
        if not selected_examples:
            self.logger.warning("No examples available for few-shot learning")
            return ""
        
        # Format examples
        formatted_examples = []
        for i, example in enumerate(selected_examples, 1):
            formatted_example = f"""
EXAMPLE {i}:

Context: {example.context}
Factual Scenario: {example.factual}
Counterfactual Intervention: {example.intervention}

Analysis: {example.analysis}

Reasoning Steps:
{chr(10).join([f"- {step}" for step in example.reasoning_steps])}
"""
            formatted_examples.append(formatted_example)
        
        result = "Here are some examples of high-quality causal analysis:\n" + "\n---\n".join(formatted_examples) + "\n---\n"
        
        self.struct_logger.log_interaction(
            "few_shot_examples_generated",
            {
                "task_type": task_type,
                "domain": domain,
                "n_requested": n_examples,
                "n_returned": len(selected_examples),
                "avg_quality": sum(ex.quality_score for ex in selected_examples) / len(selected_examples)
            }
        )
        
        return result
    
    def generate_chain_of_thought_prompt(self,
                                       task_type: str,
                                       domain: str = "general",
                                       template_name: Optional[str] = None,
                                       **kwargs) -> str:
        """
        Generate a chain-of-thought prompt for causal analysis.
        
        Args:
            task_type: Type of analysis (counterfactual, treatment_effect, do_calculus)
            domain: Domain for examples and templates
            template_name: Specific template to use (if None, uses default for task_type)
            **kwargs: Variables to fill in the template
            
        Returns:
            Complete chain-of-thought prompt
        """
        self.logger.info(f"Generating chain-of-thought prompt for {task_type} in {domain} domain")
        
        # Select template
        if template_name is None:
            template_name = f"{task_type}_cot"
        
        if template_name not in self.templates:
            self.logger.error(f"Template '{template_name}' not found")
            raise ValueError(f"Template '{template_name}' not available. Available: {list(self.templates.keys())}")
        
        template = self.templates[template_name]
        
        # Get few-shot examples
        few_shot_examples = self.get_few_shot_examples(
            task_type=task_type,
            domain=domain,
            n_examples=kwargs.get('n_examples', 2)
        )
        
        # Prepare template variables
        template_vars = kwargs.copy()
        template_vars['few_shot_examples'] = few_shot_examples
        
        # Check for missing required variables
        missing_vars = [var for var in template.variables if var not in template_vars]
        if missing_vars:
            self.logger.error(f"Missing required template variables: {missing_vars}")
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        # Fill template
        try:
            prompt = template.template.format(**template_vars)
        except KeyError as e:
            self.logger.error(f"Template formatting error: {e}")
            raise ValueError(f"Template formatting error: {e}")
        
        self.struct_logger.log_interaction(
            "cot_prompt_generated",
            {
                "task_type": task_type,
                "domain": domain,
                "template_name": template_name,
                "prompt_length": len(prompt),
                "n_examples": template_vars.get('n_examples', 2)
            }
        )
        
        self.logger.debug(f"Generated chain-of-thought prompt: {len(prompt)} characters")
        return prompt
    
    def generate_self_consistency_prompts(self,
                                        task_type: str,
                                        domain: str = "general", 
                                        n_variants: int = 3,
                                        **kwargs) -> List[str]:
        """
        Generate multiple prompt variants for self-consistency checking.
        
        Args:
            task_type: Type of causal analysis
            domain: Domain for examples
            n_variants: Number of prompt variants to generate
            **kwargs: Template variables
            
        Returns:
            List of prompt variants for self-consistency
        """
        self.logger.info(f"Generating {n_variants} self-consistency prompt variants")
        
        prompts = []
        
        for i in range(n_variants):
            # Vary the examples and reasoning emphasis
            variant_kwargs = kwargs.copy()
            variant_kwargs['n_examples'] = 2 + (i % 2)  # 2 or 3 examples
            
            # Generate base prompt
            prompt = self.generate_chain_of_thought_prompt(
                task_type=task_type,
                domain=domain,
                **variant_kwargs
            )
            
            # Add variation in instruction emphasis
            if i == 0:
                emphasis = "Pay special attention to quantifying the magnitude of effects."
            elif i == 1:
                emphasis = "Focus particularly on identifying and validating key assumptions."
            else:
                emphasis = "Consider alternative explanations and competing hypotheses carefully."
            
            prompt += f"\n\nIMPORTANT: {emphasis}\n"
            
            prompts.append(prompt)
        
        self.struct_logger.log_interaction(
            "self_consistency_prompts_generated",
            {
                "task_type": task_type,
                "domain": domain,
                "n_variants": n_variants,
                "avg_prompt_length": sum(len(p) for p in prompts) / len(prompts)
            }
        )
        
        return prompts
    
    def add_example(self, example: CausalExample) -> None:
        """
        Add a new example to the database.
        
        Args:
            example: CausalExample to add
        """
        if example.domain not in self.examples_db:
            self.examples_db[example.domain] = []
        
        self.examples_db[example.domain].append(example)
        self.logger.info(f"Added new example to {example.domain} domain")
    
    def add_template(self, template: PromptTemplate) -> None:
        """
        Add a new prompt template.
        
        Args:
            template: PromptTemplate to add
        """
        self.templates[template.name] = template
        self.logger.info(f"Added new template: {template.name}")
    
    def evaluate_prompt_performance(self, 
                                  prompt_id: str,
                                  response_quality: float,
                                  response_time: float,
                                  user_feedback: Optional[str] = None) -> None:
        """
        Track prompt performance for optimization.
        
        Args:
            prompt_id: Unique identifier for the prompt
            response_quality: Quality score (0-1)
            response_time: Response time in seconds
            user_feedback: Optional user feedback
        """
        if prompt_id not in self.prompt_performance:
            self.prompt_performance[prompt_id] = {
                "quality_scores": [],
                "response_times": [],
                "feedback": [],
                "usage_count": 0
            }
        
        perf = self.prompt_performance[prompt_id]
        perf["quality_scores"].append(response_quality)
        perf["response_times"].append(response_time)
        perf["usage_count"] += 1
        
        if user_feedback:
            perf["feedback"].append(user_feedback)
        
        self.logger.debug(f"Updated performance tracking for prompt {prompt_id}")
    
    def get_optimal_template(self, task_type: str, domain: str = "general") -> str:
        """
        Get the best-performing template for a task type and domain.
        
        Args:
            task_type: Type of causal analysis
            domain: Domain context
            
        Returns:
            Name of optimal template
        """
        # Simple heuristic: return chain-of-thought version if available
        cot_template = f"{task_type}_cot"
        if cot_template in self.templates:
            return cot_template
        
        # Fallback to any template matching task type
        matching_templates = [name for name in self.templates.keys() 
                            if task_type in name]
        
        if matching_templates:
            return matching_templates[0]
        
        # Last resort: return first available template
        if self.templates:
            return list(self.templates.keys())[0]
        
        raise ValueError("No templates available")
    
    def load_examples_db(self, file_path: str) -> None:
        """
        Load examples database from JSON file.
        
        Args:
            file_path: Path to JSON file containing examples
        """
        self.logger.info(f"Loading examples database from {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for domain, examples_list in data.items():
                examples = [CausalExample(**ex) for ex in examples_list]
                self.examples_db[domain] = examples
            
            total_examples = sum(len(examples) for examples in self.examples_db.values())
            self.logger.info(f"Loaded {total_examples} examples from database")
            
        except Exception as e:
            self.logger.error(f"Failed to load examples database: {e}")
            raise RuntimeError(f"Could not load examples database: {e}")
    
    def save_examples_db(self, file_path: str) -> None:
        """
        Save examples database to JSON file.
        
        Args:
            file_path: Path to save JSON file
        """
        self.logger.info(f"Saving examples database to {file_path}")
        
        try:
            # Convert examples to dictionaries
            data = {}
            for domain, examples in self.examples_db.items():
                data[domain] = [asdict(ex) for ex in examples]
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            total_examples = sum(len(examples) for examples in self.examples_db.values())
            self.logger.info(f"Saved {total_examples} examples to database")
            
        except Exception as e:
            self.logger.error(f"Failed to save examples database: {e}")
            raise RuntimeError(f"Could not save examples database: {e}")


# Convenience functions for common use cases
def create_counterfactual_prompt(context: str, 
                               factual: str, 
                               intervention: str,
                               domain: str = "general",
                               use_few_shot: bool = True) -> str:
    """
    Create an enhanced counterfactual analysis prompt.
    
    Args:
        context: Background context
        factual: Factual scenario
        intervention: Counterfactual intervention
        domain: Domain for examples
        use_few_shot: Whether to include few-shot examples
        
    Returns:
        Enhanced prompt with chain-of-thought reasoning
    """
    engine = CausalPromptEngine()
    
    return engine.generate_chain_of_thought_prompt(
        task_type="counterfactual",
        domain=domain,
        context=context,
        factual=factual,
        intervention=intervention,
        n_examples=2 if use_few_shot else 0
    )


def create_treatment_effect_prompt(treatment: str,
                                 outcome: str, 
                                 context: str,
                                 domain: str = "general") -> str:
    """
    Create an enhanced treatment effect analysis prompt.
    
    Args:
        treatment: Treatment variable description
        outcome: Outcome variable description
        context: Study context
        domain: Domain for examples
        
    Returns:
        Enhanced prompt with systematic reasoning
    """
    engine = CausalPromptEngine()
    
    return engine.generate_chain_of_thought_prompt(
        task_type="treatment_effect",
        domain=domain,
        treatment=treatment,
        outcome=outcome,
        context=context
    )


def create_self_consistency_analysis(context: str,
                                   factual: str,
                                   intervention: str,
                                   domain: str = "general") -> List[str]:
    """
    Create multiple prompt variants for self-consistency checking.
    
    Args:
        context: Background context
        factual: Factual scenario  
        intervention: Counterfactual intervention
        domain: Domain for examples
        
    Returns:
        List of prompt variants for ensemble analysis
    """
    engine = CausalPromptEngine()
    
    return engine.generate_self_consistency_prompts(
        task_type="counterfactual",
        domain=domain,
        context=context,
        factual=factual,
        intervention=intervention,
        n_variants=3
    )