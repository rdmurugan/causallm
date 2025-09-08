# 🏥 CausalLLM Domain Packages

**Transform your causal analysis with industry-specific expertise built into CausalLLM.**

Domain packages provide pre-configured components that make causal analysis easier, faster, and more accurate for specific industries. Each domain package includes realistic data generators, expert knowledge, and business-ready analysis templates.

---

## 🚀 Quick Start with Standardized Interfaces

```python
from causallm import EnhancedCausalLLM, HealthcareDomain, InsuranceDomain

# Initialize with domain-optimized configuration
causal_llm = EnhancedCausalLLM(
    config_file='healthcare_config.json',  # Domain-specific configuration
    use_async=True                         # Enable async for large datasets
)

# Healthcare analysis with standardized parameters
healthcare = HealthcareDomain()
clinical_data = healthcare.generate_clinical_trial_data(n_patients=500)

treatment_results = causal_llm.estimate_causal_effect(
    data=clinical_data,                    # Standardized: 'data'
    treatment_variable='treatment',        # Standardized: 'treatment_variable'
    outcome_variable='recovery_time',      # Standardized: 'outcome_variable'
    covariate_variables=['age', 'severity'], # Standardized: 'covariate_variables'
    domain_context='healthcare'           # Standardized: 'domain_context'
)

# Insurance analysis with standardized parameters
insurance = InsuranceDomain()
policy_data = insurance.generate_stop_loss_data(n_policies=1000)

risk_analysis = causal_llm.estimate_causal_effect(
    data=policy_data,                     # Standardized: 'data'
    treatment_variable='industry_type',   # Standardized: 'treatment_variable'
    outcome_variable='total_claim_amount', # Standardized: 'outcome_variable'
    covariate_variables=['company_size', 'region'], # Standardized: 'covariate_variables'
    domain_context='insurance'           # Standardized: 'domain_context'
)
```

### Configuration-Based Domain Setup

```python
from causallm.config import CausalLLMConfig

# Create domain-specific configurations
healthcare_config = CausalLLMConfig()
healthcare_config.llm.provider = 'openai'          # Medical insights
healthcare_config.statistical.significance_level = 0.01  # Stricter for clinical
healthcare_config.performance.chunk_size = 10000   # Patient data optimization

insurance_config = CausalLLMConfig()
insurance_config.performance.use_async = True      # Handle large policy datasets
insurance_config.performance.chunk_size = 50000   # Policy data optimization
insurance_config.statistical.bootstrap_samples = 2000  # Robust estimates

# Initialize with domain configurations
healthcare_llm = EnhancedCausalLLM(config=healthcare_config)
insurance_llm = EnhancedCausalLLM(config=insurance_config)
```

---

## 🏥 Healthcare Domain

**Focus**: Clinical analysis, treatment effectiveness, patient outcomes

### Key Features
- **Clinical Trial Data**: Randomized controlled trials with proper randomization
- **Patient Cohort Data**: Observational studies with realistic patient characteristics
- **Treatment Effectiveness Analysis**: Pre-configured analysis for clinical outcomes
- **Medical Domain Knowledge**: Built-in understanding of medical confounders and relationships
- **Clinical Interpretation**: Medical context for statistical results

### Example Use Cases
```python
from causallm.domains.healthcare import HealthcareDomain

healthcare = HealthcareDomain()

# Generate clinical trial data
trial_data = healthcare.generate_clinical_trial_data(
    n_patients=500,
    treatment_arms=['control', 'new_treatment'],
    randomization_ratio=[0.5, 0.5]
)

# Analyze treatment effectiveness
causal_engine = EnhancedCausalLLM()
results = healthcare.treatment_template.run_analysis(
    'treatment_effectiveness',
    trial_data,
    causal_engine
)

print(f"Treatment effect: {results.effect_estimate:.2f} days")
print(f"Clinical interpretation: {results.domain_interpretation}")
```

### Available Analyses
- **Treatment Effectiveness**: Analyze treatment impact on clinical outcomes
- **Safety Analysis**: Evaluate treatment safety and adverse events  
- **Mortality Analysis**: Assess treatment effects on mortality outcomes
- **Patient Satisfaction**: Analyze factors affecting patient satisfaction
- **Readmission Analysis**: Study factors contributing to hospital readmissions

---

## 💼 Insurance Domain

**Focus**: Risk assessment, premium optimization, claims analysis

### Key Features
- **Stop Loss Insurance Data**: Realistic policy and claims data
- **Risk Factor Analysis**: Industry, company size, and regional risk assessment
- **Actuarial Domain Knowledge**: Built-in understanding of insurance relationships
- **Business Impact Calculation**: ROI and premium optimization insights
- **Regulatory Compliance**: Insurance-specific constraints and requirements

### Example Use Cases
```python
from causallm.domains.insurance import InsuranceDomain

insurance = InsuranceDomain()

# Generate stop loss insurance data
policy_data = insurance.generate_stop_loss_data(n_policies=1000)

# Analyze risk factors
risk_results = insurance.risk_template.run_analysis(
    'risk_assessment',
    policy_data
)

print(f"Industry risk effect: ${risk_results.effect_estimate:,.0f}")
print(f"Business recommendation: {risk_results.recommendations[0]}")
```

### Available Analyses
- **Risk Assessment**: Analyze factors affecting claim amounts
- **Premium Optimization**: Optimize pricing based on risk factors
- **Claims Prediction**: Predict high-risk policies
- **Underwriting Analysis**: Support underwriting decisions

---

## 📊 Marketing Domain (Coming Soon)

**Focus**: Campaign attribution, ROI optimization, customer analytics

### Planned Features
- Multi-channel attribution modeling
- Customer lifetime value analysis
- Campaign effectiveness measurement
- Marketing mix optimization

---

## 🎓 Education Domain (Coming Soon) 

**Focus**: Student outcomes, intervention analysis, policy evaluation

### Planned Features
- Student achievement analysis
- Educational intervention effectiveness
- Policy impact assessment  
- Resource allocation optimization

---

## 🧪 Experimentation Domain (Coming Soon)

**Focus**: A/B testing, experimental design, causal inference

### Planned Features
- Advanced A/B test analysis
- Multi-variate testing
- Heterogeneous treatment effects
- Experimental design optimization

---

## 🏗️ Architecture

Each domain package consists of:

### 1. **Data Generators** (`domains/{domain}/generators/`)
- Realistic synthetic data with proper causal structure
- Domain-specific variables and relationships
- Multiple data scenarios (trials, cohorts, populations)

### 2. **Domain Knowledge** (`domains/{domain}/knowledge/`)
- Expert knowledge about causal relationships
- Known confounders and effect modifiers  
- Industry-specific constraints and rules
- Evidence-based priors and assumptions

### 3. **Analysis Templates** (`domains/{domain}/templates/`)
- Pre-configured analysis workflows
- Domain-specific interpretation
- Business impact calculations
- Actionable recommendations

---

## 🔧 Creating Custom Domain Packages

You can extend CausalLLM with your own domain expertise:

### 1. **Create Domain Structure**
```python
from causallm.domains.base import BaseDomainDataGenerator, BaseDomainKnowledge, BaseAnalysisTemplate

class MyDomainDataGenerator(BaseDomainDataGenerator):
    def get_causal_structure(self):
        # Define your domain's causal structure
        pass
    
    def generate_base_variables(self, n_samples):
        # Generate exogenous variables
        pass
    
    def apply_causal_mechanisms(self, data):
        # Apply causal relationships
        pass

class MyDomainKnowledge(BaseDomainKnowledge):
    def load_domain_knowledge(self):
        # Load your domain expertise
        pass
    
    def get_likely_confounders(self, treatment, outcome, available_variables):
        # Return likely confounders for your domain
        pass
```

### 2. **Package Your Domain**
```python
class MyDomain:
    def __init__(self):
        self.data_generator = MyDomainDataGenerator()
        self.domain_knowledge = MyDomainKnowledge()
        self.analysis_template = MyAnalysisTemplate()
```

---

## 📈 Benefits of Domain Packages

### **Traditional Approach**
❌ Weeks of manual analysis and hypothesis generation  
❌ Requires deep domain expertise for each analysis  
❌ Generic interpretations with limited actionability  
❌ High risk of missing important confounders  
❌ Inconsistent analysis approaches across teams  

### **Domain Package Approach**
✅ **Expert Knowledge Built-In**: Decades of domain expertise encoded  
✅ **Faster Time to Insights**: Minutes instead of weeks  
✅ **Consistent Quality**: Standardized analysis approaches  
✅ **Business-Ready Results**: Domain-specific interpretations  
✅ **Reduced Expertise Requirement**: Democratize advanced analytics  
✅ **Realistic Testing**: High-quality synthetic data for validation  

---

## 🤝 Contributing Domain Expertise

We welcome contributions to expand domain packages:

### **High-Priority Domains**
- **Marketing**: Attribution modeling, campaign optimization
- **Finance**: Risk modeling, portfolio optimization  
- **Education**: Student outcomes, intervention analysis
- **Manufacturing**: Quality control, process optimization
- **Retail**: Demand forecasting, pricing optimization

### **How to Contribute**
1. **Fork the Repository**: Start with the existing domain structure
2. **Build Your Domain**: Use base classes and follow patterns
3. **Add Documentation**: Include usage examples and validation
4. **Submit Pull Request**: We'll review and provide feedback
5. **Community Review**: Get input from domain experts

---

## 📖 Complete Example

See `examples/domain_packages_demo.py` for a comprehensive demonstration of healthcare and insurance domain packages in action.

---

## 🔗 Learn More

- **[Complete User Guide](COMPLETE_USER_GUIDE.md)**: Comprehensive documentation
- **[Examples](examples/)**: Domain-specific analysis examples
- **[Contributing](CONTRIBUTING.md)**: How to contribute domain expertise
- **[GitHub Discussions](https://github.com/rdmurugan/causallm/discussions)**: Community support

---

> **💎 "Domain packages don't just save time—they enable better science and better business decisions by encoding expert knowledge directly into the analysis workflow."**