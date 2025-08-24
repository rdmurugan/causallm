"""
Domain-Specific Causal Templates

This module provides pre-built causal analysis templates and patterns for different domains
including healthcare, business, education, and other specialized areas. Templates include
common causal structures, typical confounders, and domain-specific analysis approaches.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
from pathlib import Path
import pandas as pd

from causalllm.logging import get_logger


class TemplateType(Enum):
    """Types of causal templates."""
    CAUSAL_STRUCTURE = "causal_structure"
    ANALYSIS_WORKFLOW = "analysis_workflow"
    CONFOUNDER_SET = "confounder_set"
    ASSUMPTION_CHECK = "assumption_check"
    INTERPRETATION_GUIDE = "interpretation_guide"


class DomainType(Enum):
    """Supported domain types."""
    HEALTHCARE = "healthcare"
    BUSINESS = "business"
    EDUCATION = "education"
    SOCIAL_SCIENCE = "social_science"
    TECHNOLOGY = "technology"
    ENVIRONMENTAL = "environmental"
    ECONOMICS = "economics"
    POLICY = "policy"


@dataclass
class VariableTemplate:
    """Template for a variable in a causal structure."""
    
    name: str
    role: str  # treatment, outcome, confounder, mediator, etc.
    data_type: str  # continuous, binary, categorical, ordinal
    typical_values: Optional[List[Any]] = None
    measurement_considerations: List[str] = field(default_factory=list)
    common_proxies: List[str] = field(default_factory=list)


@dataclass
class CausalEdgeTemplate:
    """Template for a causal relationship."""
    
    source: str
    target: str
    relationship_type: str  # causal, confounding, mediating
    expected_direction: str  # positive, negative, non_linear
    mechanism_description: str
    typical_effect_size: Optional[str] = None
    evidence_strength: str = "moderate"
    common_moderators: List[str] = field(default_factory=list)


@dataclass
class DomainCausalTemplate:
    """Complete causal template for a domain."""
    
    domain: DomainType
    template_name: str
    description: str
    variables: List[VariableTemplate]
    causal_edges: List[CausalEdgeTemplate]
    common_confounders: List[str]
    typical_analyses: List[str]
    key_assumptions: List[str]
    interpretation_guidelines: List[str]
    example_questions: List[str]
    references: List[str] = field(default_factory=list)


class DomainTemplateEngine:
    """Engine for managing and applying domain-specific causal templates."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.logger = get_logger("causalllm.domain_causal_templates")
        
        # Initialize built-in templates
        self.templates = self._initialize_builtin_templates()
        
        # Custom templates directory
        self.custom_templates_dir = Path("custom_templates")
        self.custom_templates_dir.mkdir(exist_ok=True)
    
    def _initialize_builtin_templates(self) -> Dict[str, List[DomainCausalTemplate]]:
        """Initialize built-in domain templates."""
        
        templates = {}
        
        # Healthcare templates
        templates[DomainType.HEALTHCARE.value] = [
            self._create_clinical_trial_template(),
            self._create_epidemiology_template(),
            self._create_health_intervention_template()
        ]
        
        # Business templates
        templates[DomainType.BUSINESS.value] = [
            self._create_marketing_roi_template()
        ]
        
        # Education templates
        templates[DomainType.EDUCATION.value] = [
            self._create_learning_intervention_template()
        ]
        
        return templates
    
    def _create_clinical_trial_template(self) -> DomainCausalTemplate:
        """Template for clinical trial analysis."""
        
        variables = [
            VariableTemplate("treatment", "treatment", "binary", [0, 1], 
                           ["Randomization", "Blinding", "Adherence"]),
            VariableTemplate("outcome", "outcome", "continuous", 
                           measurement_considerations=["Baseline measurement", "Follow-up timing"]),
            VariableTemplate("age", "confounder", "continuous", 
                           measurement_considerations=["Age at enrollment"]),
            VariableTemplate("baseline_severity", "confounder", "continuous",
                           measurement_considerations=["Pre-treatment status"]),
            VariableTemplate("comorbidities", "confounder", "categorical",
                           measurement_considerations=["Complete medical history"])
        ]
        
        edges = [
            CausalEdgeTemplate("treatment", "outcome", "causal", "positive", 
                             "Treatment effect on primary outcome"),
            CausalEdgeTemplate("age", "treatment", "confounding", "negative",
                             "Age affects treatment assignment (if not randomized)"),
            CausalEdgeTemplate("age", "outcome", "confounding", "negative",
                             "Age affects health outcomes"),
            CausalEdgeTemplate("baseline_severity", "outcome", "confounding", "negative",
                             "Sicker patients have worse outcomes")
        ]
        
        return DomainCausalTemplate(
            domain=DomainType.HEALTHCARE,
            template_name="Clinical Trial Analysis",
            description="Template for analyzing randomized controlled trials",
            variables=variables,
            causal_edges=edges,
            common_confounders=["age", "gender", "baseline_severity", "comorbidities"],
            typical_analyses=["Intent-to-treat", "Per-protocol", "Subgroup analysis"],
            key_assumptions=["Randomization successful", "No selection bias", "SUTVA"],
            interpretation_guidelines=[
                "Consider clinical vs statistical significance",
                "Report absolute risk reduction, not just p-values",
                "Assess generalizability to target population"
            ],
            example_questions=[
                "What is the treatment effect on the primary outcome?",
                "Does the effect vary by patient subgroups?",
                "Are there any safety concerns?"
            ]
        )
    
    def _create_epidemiology_template(self) -> DomainCausalTemplate:
        """Template for epidemiological studies."""
        
        variables = [
            VariableTemplate("exposure", "treatment", "binary", [0, 1],
                           ["Exposure timing", "Dose measurement"]),
            VariableTemplate("disease", "outcome", "binary", [0, 1],
                           ["Case definition", "Diagnostic criteria"]),
            VariableTemplate("age", "confounder", "continuous"),
            VariableTemplate("socioeconomic_status", "confounder", "ordinal"),
            VariableTemplate("lifestyle_factors", "confounder", "categorical")
        ]
        
        edges = [
            CausalEdgeTemplate("exposure", "disease", "causal", "positive",
                             "Exposure increases disease risk"),
            CausalEdgeTemplate("age", "exposure", "confounding", "varies",
                             "Age affects exposure patterns"),
            CausalEdgeTemplate("age", "disease", "confounding", "positive",
                             "Age increases disease risk"),
            CausalEdgeTemplate("socioeconomic_status", "exposure", "confounding", "negative",
                             "SES affects exposure likelihood"),
            CausalEdgeTemplate("socioeconomic_status", "disease", "confounding", "negative",
                             "SES affects disease risk")
        ]
        
        return DomainCausalTemplate(
            domain=DomainType.HEALTHCARE,
            template_name="Epidemiological Study",
            description="Template for observational epidemiological studies",
            variables=variables,
            causal_edges=edges,
            common_confounders=["age", "gender", "socioeconomic_status", "lifestyle_factors"],
            typical_analyses=["Logistic regression", "Propensity score matching", "Instrumental variables"],
            key_assumptions=["No unmeasured confounding", "Temporal precedence", "No selection bias"],
            interpretation_guidelines=[
                "Consider Bradford Hill criteria",
                "Assess dose-response relationship",
                "Evaluate biological plausibility"
            ],
            example_questions=[
                "Does exposure X increase risk of disease Y?",
                "What is the population attributable risk?",
                "Are there vulnerable subpopulations?"
            ]
        )
    
    def _create_health_intervention_template(self) -> DomainCausalTemplate:
        """Template for health intervention studies."""
        
        variables = [
            VariableTemplate("intervention", "treatment", "categorical",
                           measurement_considerations=["Implementation fidelity"]),
            VariableTemplate("health_outcome", "outcome", "continuous",
                           measurement_considerations=["Validated instruments"]),
            VariableTemplate("health_behavior", "mediator", "continuous",
                           measurement_considerations=["Self-report vs objective"]),
            VariableTemplate("healthcare_access", "confounder", "ordinal"),
            VariableTemplate("motivation", "confounder", "continuous")
        ]
        
        edges = [
            CausalEdgeTemplate("intervention", "health_behavior", "causal", "positive",
                             "Intervention changes behavior"),
            CausalEdgeTemplate("health_behavior", "health_outcome", "causal", "positive",
                             "Behavior change improves health"),
            CausalEdgeTemplate("healthcare_access", "intervention", "confounding", "positive",
                             "Access affects intervention participation"),
            CausalEdgeTemplate("motivation", "intervention", "confounding", "positive",
                             "Motivated individuals more likely to participate")
        ]
        
        return DomainCausalTemplate(
            domain=DomainType.HEALTHCARE,
            template_name="Health Intervention",
            description="Template for behavioral health interventions",
            variables=variables,
            causal_edges=edges,
            common_confounders=["healthcare_access", "motivation", "baseline_health"],
            typical_analyses=["Mediation analysis", "Difference-in-differences", "RCT analysis"],
            key_assumptions=["No spillover effects", "Stable unit treatment value"],
            interpretation_guidelines=[
                "Distinguish between efficacy and effectiveness",
                "Consider implementation barriers",
                "Assess sustainability of effects"
            ],
            example_questions=[
                "Does the intervention improve health outcomes?",
                "What are the mediating mechanisms?",
                "How can implementation be improved?"
            ]
        )
    
    def _create_marketing_roi_template(self) -> DomainCausalTemplate:
        """Template for marketing ROI analysis."""
        
        variables = [
            VariableTemplate("marketing_spend", "treatment", "continuous",
                           measurement_considerations=["Attribution tracking", "Budget allocation"]),
            VariableTemplate("sales_revenue", "outcome", "continuous",
                           measurement_considerations=["Revenue attribution", "Time lag"]),
            VariableTemplate("market_conditions", "confounder", "continuous"),
            VariableTemplate("seasonality", "confounder", "categorical"),
            VariableTemplate("brand_awareness", "mediator", "continuous")
        ]
        
        edges = [
            CausalEdgeTemplate("marketing_spend", "brand_awareness", "causal", "positive",
                             "Marketing increases awareness"),
            CausalEdgeTemplate("brand_awareness", "sales_revenue", "causal", "positive",
                             "Awareness drives sales"),
            CausalEdgeTemplate("market_conditions", "sales_revenue", "confounding", "positive",
                             "Market conditions affect sales"),
            CausalEdgeTemplate("seasonality", "sales_revenue", "confounding", "varies",
                             "Seasonal effects on sales")
        ]
        
        return DomainCausalTemplate(
            domain=DomainType.BUSINESS,
            template_name="Marketing ROI Analysis",
            description="Template for measuring marketing return on investment",
            variables=variables,
            causal_edges=edges,
            common_confounders=["market_conditions", "seasonality", "competitive_activity"],
            typical_analyses=["Attribution modeling", "Marketing mix modeling", "Lift testing"],
            key_assumptions=["Stable customer base", "No external shocks"],
            interpretation_guidelines=[
                "Consider both short-term and long-term effects",
                "Account for diminishing returns",
                "Separate incremental from baseline sales"
            ],
            example_questions=[
                "What is the ROI of marketing channel X?",
                "How does marketing effectiveness vary by segment?",
                "What is the optimal marketing budget allocation?"
            ]
        )
    
    def _create_learning_intervention_template(self) -> DomainCausalTemplate:
        """Template for educational intervention analysis."""
        
        variables = [
            VariableTemplate("intervention", "treatment", "binary",
                           measurement_considerations=["Implementation fidelity", "Dosage"]),
            VariableTemplate("learning_outcome", "outcome", "continuous",
                           measurement_considerations=["Assessment validity", "Test alignment"]),
            VariableTemplate("prior_achievement", "confounder", "continuous"),
            VariableTemplate("socioeconomic_background", "confounder", "ordinal"),
            VariableTemplate("teacher_quality", "confounder", "continuous"),
            VariableTemplate("engagement", "mediator", "continuous")
        ]
        
        edges = [
            CausalEdgeTemplate("intervention", "engagement", "causal", "positive",
                             "Intervention increases student engagement"),
            CausalEdgeTemplate("engagement", "learning_outcome", "causal", "positive",
                             "Engagement improves learning"),
            CausalEdgeTemplate("prior_achievement", "learning_outcome", "confounding", "positive",
                             "Previous learning affects current outcomes"),
            CausalEdgeTemplate("socioeconomic_background", "learning_outcome", "confounding", "positive",
                             "SES affects educational outcomes")
        ]
        
        return DomainCausalTemplate(
            domain=DomainType.EDUCATION,
            template_name="Learning Intervention",
            description="Template for educational intervention studies",
            variables=variables,
            causal_edges=edges,
            common_confounders=["prior_achievement", "socioeconomic_background", "teacher_quality"],
            typical_analyses=["Randomized controlled trial", "Regression discontinuity", "Difference-in-differences"],
            key_assumptions=["No spillover effects between students", "Stable teaching quality"],
            interpretation_guidelines=[
                "Consider educational significance vs statistical significance",
                "Assess generalizability across contexts",
                "Evaluate implementation feasibility"
            ],
            example_questions=[
                "Does intervention X improve learning outcomes?",
                "What are the mediating mechanisms?",
                "Does effectiveness vary by student characteristics?"
            ]
        )
    
    def get_templates_for_domain(self, domain: str) -> List[DomainCausalTemplate]:
        """Get all templates for a specific domain."""
        return self.templates.get(domain, [])
    
    def get_template_by_name(self, domain: str, template_name: str) -> Optional[DomainCausalTemplate]:
        """Get a specific template by domain and name."""
        domain_templates = self.get_templates_for_domain(domain)
        for template in domain_templates:
            if template.template_name == template_name:
                return template
        return None
    
    async def suggest_template(self, 
                             data_description: str,
                             research_question: str,
                             domain: Optional[str] = None) -> List[DomainCausalTemplate]:
        """
        Suggest appropriate templates based on data and research question.
        
        Args:
            data_description: Description of available data
            research_question: Research question or analysis goal
            domain: Optional domain specification
            
        Returns:
            List of recommended templates
        """
        self.logger.info("Suggesting templates based on context")
        
        # If domain not specified, try to infer it
        if not domain:
            domain = await self._infer_domain(data_description, research_question)
        
        # Get candidate templates
        candidate_templates = []
        if domain in self.templates:
            candidate_templates = self.templates[domain]
        
        # Use LLM to rank templates by relevance
        if candidate_templates:
            ranked_templates = await self._rank_templates_by_relevance(
                candidate_templates, data_description, research_question
            )
            return ranked_templates[:3]  # Top 3 recommendations
        
        return []
    
    async def _infer_domain(self, data_description: str, research_question: str) -> str:
        """Infer domain from data description and research question."""
        
        domain_keywords = {
            DomainType.HEALTHCARE.value: [
                "patient", "treatment", "clinical", "medical", "health", "disease", 
                "therapy", "drug", "intervention", "symptoms", "diagnosis"
            ],
            DomainType.BUSINESS.value: [
                "revenue", "sales", "marketing", "customer", "profit", "ROI", 
                "campaign", "conversion", "business", "company", "market"
            ],
            DomainType.EDUCATION.value: [
                "student", "learning", "school", "teacher", "education", "test", 
                "achievement", "curriculum", "grade", "academic"
            ],
            DomainType.SOCIAL_SCIENCE.value: [
                "behavior", "social", "psychology", "survey", "attitude", "policy",
                "intervention", "community", "demographic"
            ]
        }
        
        text = (data_description + " " + research_question).lower()
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            domain_scores[domain] = score
        
        # Return domain with highest score
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return "general"
    
    async def _rank_templates_by_relevance(self, 
                                         templates: List[DomainCausalTemplate],
                                         data_description: str, 
                                         research_question: str) -> List[DomainCausalTemplate]:
        """Rank templates by relevance using LLM."""
        
        template_summaries = []
        for i, template in enumerate(templates):
            summary = f"""
            Template {i}: {template.template_name}
            Description: {template.description}
            Key variables: {[v.name for v in template.variables[:5]]}
            Example questions: {template.example_questions[:3]}
            """
            template_summaries.append(summary)
        
        ranking_prompt = f"""
        Rank these causal analysis templates by relevance to the given context:
        
        DATA DESCRIPTION: {data_description}
        RESEARCH QUESTION: {research_question}
        
        AVAILABLE TEMPLATES:
        {chr(10).join(template_summaries)}
        
        Rank the templates from most relevant (1) to least relevant based on:
        1. Variable alignment with described data
        2. Relevance to research question
        3. Appropriateness of causal structure
        
        Return ranking as JSON array of template indices: [0, 2, 1, ...]
        """
        
        try:
            response = await self.llm_client.generate_response(ranking_prompt)
            
            # Parse ranking
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                ranking = json.loads(json_match.group())
                return [templates[i] for i in ranking if 0 <= i < len(templates)]
                
        except Exception as e:
            self.logger.error(f"Template ranking failed: {e}")
        
        # Fallback: return original order
        return templates
    
    async def customize_template(self, 
                               base_template: DomainCausalTemplate,
                               customization_request: str,
                               available_variables: List[str]) -> DomainCausalTemplate:
        """
        Customize a template based on specific requirements and available data.
        
        Args:
            base_template: Base template to customize
            customization_request: Description of needed customizations
            available_variables: Variables available in the data
            
        Returns:
            Customized template
        """
        self.logger.info(f"Customizing template: {base_template.template_name}")
        
        customization_prompt = f"""
        Customize this causal analysis template based on the specific requirements:
        
        BASE TEMPLATE: {base_template.template_name}
        DESCRIPTION: {base_template.description}
        
        CURRENT VARIABLES: {[v.name for v in base_template.variables]}
        CURRENT EDGES: {[(e.source, e.target) for e in base_template.causal_edges]}
        
        CUSTOMIZATION REQUEST: {customization_request}
        AVAILABLE VARIABLES: {available_variables}
        
        Provide customizations including:
        1. Variable mappings (template variable -> available variable)
        2. Additional variables to include
        3. Modified causal relationships
        4. Updated confounders list
        5. Revised assumptions
        
        Respond with JSON:
        {{
            "variable_mappings": {{"template_var": "available_var"}},
            "additional_variables": ["var1", "var2"],
            "modified_edges": [
                {{"source": "var1", "target": "var2", "relationship": "causal"}}
            ],
            "updated_confounders": ["conf1", "conf2"],
            "revised_assumptions": ["assumption1", "assumption2"],
            "customization_notes": "explanation of changes"
        }}
        """
        
        try:
            response = await self.llm_client.generate_response(customization_prompt)
            
            # Parse customization instructions
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                customizations = json.loads(json_match.group())
                
                # Apply customizations to create new template
                return self._apply_template_customizations(base_template, customizations)
                
        except Exception as e:
            self.logger.error(f"Template customization failed: {e}")
        
        # Return original template if customization fails
        return base_template
    
    def _apply_template_customizations(self, 
                                     base_template: DomainCausalTemplate,
                                     customizations: Dict[str, Any]) -> DomainCausalTemplate:
        """Apply customization instructions to create new template."""
        
        # Create copy of base template
        customized = DomainCausalTemplate(
            domain=base_template.domain,
            template_name=f"{base_template.template_name} (Customized)",
            description=base_template.description + f" | {customizations.get('customization_notes', '')}",
            variables=base_template.variables.copy(),
            causal_edges=base_template.causal_edges.copy(),
            common_confounders=customizations.get('updated_confounders', base_template.common_confounders),
            typical_analyses=base_template.typical_analyses,
            key_assumptions=customizations.get('revised_assumptions', base_template.key_assumptions),
            interpretation_guidelines=base_template.interpretation_guidelines,
            example_questions=base_template.example_questions
        )
        
        # Apply variable mappings
        variable_mappings = customizations.get('variable_mappings', {})
        for template_var, available_var in variable_mappings.items():
            # Update variable names in template
            for var in customized.variables:
                if var.name == template_var:
                    var.name = available_var
            
            # Update edge references
            for edge in customized.causal_edges:
                if edge.source == template_var:
                    edge.source = available_var
                if edge.target == template_var:
                    edge.target = available_var
        
        # Add additional variables
        additional_vars = customizations.get('additional_variables', [])
        for var_name in additional_vars:
            new_var = VariableTemplate(
                name=var_name,
                role="covariate",  # Default role
                data_type="continuous",  # Default type
                measurement_considerations=["Added during customization"]
            )
            customized.variables.append(new_var)
        
        # Add modified edges
        modified_edges = customizations.get('modified_edges', [])
        for edge_info in modified_edges:
            new_edge = CausalEdgeTemplate(
                source=edge_info.get('source', ''),
                target=edge_info.get('target', ''),
                relationship_type=edge_info.get('relationship', 'causal'),
                expected_direction="positive",  # Default
                mechanism_description="Added during customization"
            )
            customized.causal_edges.append(new_edge)
        
        return customized
    
    def save_custom_template(self, template: DomainCausalTemplate, filename: str):
        """Save a custom template to file."""
        
        filepath = self.custom_templates_dir / f"{filename}.json"
        
        # Convert template to dictionary
        template_dict = {
            "domain": template.domain.value,
            "template_name": template.template_name,
            "description": template.description,
            "variables": [
                {
                    "name": v.name,
                    "role": v.role,
                    "data_type": v.data_type,
                    "typical_values": v.typical_values,
                    "measurement_considerations": v.measurement_considerations,
                    "common_proxies": v.common_proxies
                }
                for v in template.variables
            ],
            "causal_edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "relationship_type": e.relationship_type,
                    "expected_direction": e.expected_direction,
                    "mechanism_description": e.mechanism_description,
                    "typical_effect_size": e.typical_effect_size,
                    "evidence_strength": e.evidence_strength,
                    "common_moderators": e.common_moderators
                }
                for e in template.causal_edges
            ],
            "common_confounders": template.common_confounders,
            "typical_analyses": template.typical_analyses,
            "key_assumptions": template.key_assumptions,
            "interpretation_guidelines": template.interpretation_guidelines,
            "example_questions": template.example_questions,
            "references": template.references
        }
        
        with open(filepath, 'w') as f:
            json.dump(template_dict, f, indent=2)
        
        self.logger.info(f"Custom template saved: {filepath}")
    
    def load_custom_template(self, filename: str) -> Optional[DomainCausalTemplate]:
        """Load a custom template from file."""
        
        filepath = self.custom_templates_dir / f"{filename}.json"
        
        try:
            with open(filepath, 'r') as f:
                template_dict = json.load(f)
            
            # Convert dictionary back to template
            variables = [
                VariableTemplate(
                    name=v["name"],
                    role=v["role"],
                    data_type=v["data_type"],
                    typical_values=v.get("typical_values"),
                    measurement_considerations=v.get("measurement_considerations", []),
                    common_proxies=v.get("common_proxies", [])
                )
                for v in template_dict["variables"]
            ]
            
            causal_edges = [
                CausalEdgeTemplate(
                    source=e["source"],
                    target=e["target"],
                    relationship_type=e["relationship_type"],
                    expected_direction=e["expected_direction"],
                    mechanism_description=e["mechanism_description"],
                    typical_effect_size=e.get("typical_effect_size"),
                    evidence_strength=e.get("evidence_strength", "moderate"),
                    common_moderators=e.get("common_moderators", [])
                )
                for e in template_dict["causal_edges"]
            ]
            
            template = DomainCausalTemplate(
                domain=DomainType(template_dict["domain"]),
                template_name=template_dict["template_name"],
                description=template_dict["description"],
                variables=variables,
                causal_edges=causal_edges,
                common_confounders=template_dict["common_confounders"],
                typical_analyses=template_dict["typical_analyses"],
                key_assumptions=template_dict["key_assumptions"],
                interpretation_guidelines=template_dict["interpretation_guidelines"],
                example_questions=template_dict["example_questions"],
                references=template_dict.get("references", [])
            )
            
            return template
            
        except Exception as e:
            self.logger.error(f"Failed to load custom template {filename}: {e}")
            return None
    
    def list_available_templates(self) -> Dict[str, List[str]]:
        """List all available templates by domain."""
        
        available = {}
        
        # Built-in templates
        for domain, templates in self.templates.items():
            available[domain] = [t.template_name for t in templates]
        
        # Custom templates
        custom_files = list(self.custom_templates_dir.glob("*.json"))
        if custom_files:
            available["custom"] = [f.stem for f in custom_files]
        
        return available


# Additional helper templates for specific use cases
def _create_additional_templates():
    """Create additional specialized templates."""
    
    # These would be additional templates for:
    # - Drug safety analysis
    # - Digital transformation ROI
    # - Employee performance
    # - Customer satisfaction
    # - Product adoption
    # - Environmental impact
    # - Policy evaluation
    # etc.
    
    pass


# Convenience functions
def create_template_engine(llm_client) -> DomainTemplateEngine:
    """Create a domain template engine."""
    return DomainTemplateEngine(llm_client)


async def get_template_for_analysis(data_description: str,
                                  research_question: str,
                                  llm_client,
                                  domain: Optional[str] = None) -> Optional[DomainCausalTemplate]:
    """Quick function to get a recommended template."""
    
    engine = create_template_engine(llm_client)
    suggestions = await engine.suggest_template(data_description, research_question, domain)
    
    return suggestions[0] if suggestions else None