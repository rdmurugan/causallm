# Tier 1 LLM Enhancements

## Overview

CausalLLM now includes **Tier 1 LLM Enhancements** - advanced capabilities that significantly improve the quality and sophistication of causal reasoning with Large Language Models. These enhancements provide intelligent prompt engineering, multi-agent collaborative analysis, and dynamic retrieval-augmented generation specifically designed for causal inference tasks.

## Key Features

### 🎯 Intelligent Prompt Engineering (`llm_prompting.py`)

Advanced prompt engineering system with:

- **Few-Shot Learning**: Curated database of high-quality causal inference examples
- **Chain-of-Thought Reasoning**: Structured step-by-step reasoning templates
- **Self-Consistency Checks**: Multiple reasoning paths for robust conclusions
- **Domain Adaptation**: Specialized prompts for healthcare, marketing, economics, etc.
- **Performance Tracking**: Continuous quality assessment and optimization

### 👥 Multi-Agent Causal Analysis (`llm_agents.py`)

Collaborative analysis system featuring specialized agents:

- **Domain Expert**: Provides field-specific insights and context
- **Statistician**: Focuses on methodological rigor and statistical validity
- **Skeptic**: Challenges assumptions and identifies potential biases
- **Synthesizer**: Combines perspectives into coherent conclusions

### 📚 Dynamic RAG for Causal Knowledge (`causal_rag.py`)

Retrieval-augmented generation system with:

- **Causal Knowledge Base**: Curated collection of methodologies, case studies, and research
- **Semantic Search**: Advanced embedding-based document retrieval
- **Domain-Specific Retrieval**: Specialized knowledge for different fields
- **Confidence Assessment**: Quality scoring and gap identification
- **Citation Tracking**: Proper attribution of retrieved knowledge

## Getting Started

### Basic Usage

```python
from causalllm.core import CausalLLMCore
from causalllm.llm_client import get_llm_client

# Initialize enhanced core
core = CausalLLMCore(
    context="Your causal analysis context",
    variables={"treatment": "Treatment description", "outcome": "Outcome description"},
    dag_edges=[("treatment", "outcome")],
    llm_client=get_llm_client()
)

# Enhanced counterfactual analysis
result = await core.enhanced_counterfactual_analysis(
    factual="Current scenario",
    intervention="Alternative scenario",
    domain="healthcare"  # optional domain specialization
)

# Enhanced treatment effect analysis
result = await core.enhanced_treatment_effect_analysis(
    treatment="Treatment variable",
    outcome="Outcome variable", 
    domain="marketing"
)
```

### Standalone Components

```python
# Use intelligent prompting independently
from causalllm.llm_prompting import IntelligentCausalPrompting

prompting = IntelligentCausalPrompting(llm_client)
enhanced_prompt = await prompting.generate_enhanced_counterfactual_prompt(
    context="Analysis context",
    factual="Factual scenario",
    intervention="Counterfactual intervention",
    domain="economics"
)

# Use multi-agent analysis independently
from causalllm.llm_agents import MultiAgentCausalAnalyzer

analyzer = MultiAgentCausalAnalyzer(llm_client)
collaborative_result = await analyzer.analyze_counterfactual(
    context="Analysis context",
    factual="Factual scenario", 
    intervention="Counterfactual intervention"
)

# Use RAG system independently
from causalllm.causal_rag import create_rag_system

rag_system = create_rag_system(llm_client=llm_client)
enhanced_query = await rag_system.enhance_query(
    query="What is the impact of confounding?",
    domain="healthcare",
    causal_concepts=["confounding", "bias"]
)
```

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    CausalLLMCore                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Intelligent    │  │  Multi-Agent    │  │  Dynamic RAG    │ │
│  │   Prompting     │  │    Analysis     │  │    System       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│           Original CausalLLM Components                    │
│  DAGParser | DoOperator | CounterfactualEngine | etc.     │
└─────────────────────────────────────────────────────────────┘
```

### Analysis Flow

1. **Query Input** → User provides causal analysis request
2. **RAG Enhancement** → Retrieve relevant knowledge and context
3. **Intelligent Prompting** → Generate optimized prompts with examples
4. **Multi-Agent Analysis** → Collaborative analysis by specialized agents
5. **Synthesis** → Combine insights into comprehensive conclusion

## Configuration

### Environment Setup

```bash
# Required dependencies
pip install sentence-transformers  # For embeddings (optional)
pip install jinja2                 # For template rendering

# Set up LLM client
export OPENAI_API_KEY="your-key"
# OR
export ANTHROPIC_API_KEY="your-key"
```

### Customization

#### Adding Custom Examples

```python
from causalllm.llm_prompting import CausalExample

# Add domain-specific examples
custom_example = CausalExample(
    context="Your domain context",
    factual="Factual scenario", 
    intervention="Counterfactual intervention",
    analysis="Expected analysis",
    reasoning_steps=["Step 1", "Step 2", "Step 3"],
    domain="your_domain",
    quality_score=0.95
)

prompting_system.add_example(custom_example)
```

#### Extending Knowledge Base

```python
from causalllm.causal_rag import CausalDocument

# Add custom knowledge documents
doc = CausalDocument(
    id="custom_method",
    title="Custom Causal Method",
    content="Detailed explanation of your method...",
    doc_type="methodology",
    domain="your_domain",
    causal_concepts=["custom_concept", "methodology"],
    citation="Your Citation (2024)"
)

rag_system.knowledge_base.add_document(doc)
```

## Examples

### Healthcare Analysis

```python
context = """
Clinical trial studying the effectiveness of a new diabetes medication.
1000 patients randomized to treatment vs placebo groups.
Primary outcome: HbA1c levels after 6 months.
"""

variables = {
    "age": "Patient age",
    "baseline_hba1c": "Baseline HbA1c levels",
    "treatment": "New medication vs placebo",
    "outcome_hba1c": "HbA1c after 6 months"
}

dag_edges = [
    ("age", "baseline_hba1c"),
    ("baseline_hba1c", "treatment"),
    ("treatment", "outcome_hba1c"),
    ("age", "outcome_hba1c")
]

core = CausalLLMCore(context, variables, dag_edges)

result = await core.enhanced_counterfactual_analysis(
    factual="65-year-old patient received new medication",
    intervention="Same patient received placebo instead",
    domain="healthcare"
)
```

### Marketing Campaign Analysis

```python
context = """
E-commerce personalization experiment testing impact of 
recommendation algorithms on customer purchase behavior.
"""

result = await core.enhanced_treatment_effect_analysis(
    treatment="Personalized product recommendations",
    outcome="Customer purchase amount over 30 days", 
    domain="marketing"
)
```

## API Reference

### Enhanced Core Methods

#### `enhanced_counterfactual_analysis(factual, intervention, domain=None)`

Comprehensive counterfactual analysis using all Tier 1 enhancements.

**Parameters:**
- `factual` (str): The factual scenario
- `intervention` (str): The counterfactual intervention
- `domain` (str, optional): Domain for specialized analysis

**Returns:**
- Dictionary with RAG analysis, intelligent prompting results, and multi-agent insights

#### `enhanced_treatment_effect_analysis(treatment, outcome, domain=None)`

Advanced treatment effect analysis with collaborative agent review.

**Parameters:**
- `treatment` (str): Treatment variable description
- `outcome` (str): Outcome variable description  
- `domain` (str, optional): Domain specialization

**Returns:**
- Comprehensive treatment effect analysis results

### Intelligent Prompting

#### `IntelligentCausalPrompting.generate_enhanced_counterfactual_prompt(...)`

Generate optimized prompts with few-shot learning and chain-of-thought reasoning.

#### `IntelligentCausalPrompting.add_example(example)`

Add custom examples to the few-shot learning database.

### Multi-Agent Analysis

#### `MultiAgentCausalAnalyzer.analyze_counterfactual(...)`

Run collaborative counterfactual analysis with specialized agents.

#### `MultiAgentCausalAnalyzer.analyze_treatment_effect(...)`

Perform collaborative treatment effect analysis.

### Dynamic RAG

#### `DynamicCausalRAG.enhance_query(...)`

Enhance queries with retrieved causal knowledge and context.

#### `CausalKnowledgeBase.add_document(document)`

Add custom knowledge documents to the retrieval system.

## Performance Considerations

### Optimization Tips

1. **Concurrent Processing**: Tier 1 components support concurrent execution
2. **Caching**: RAG system includes intelligent caching for repeated queries
3. **Selective Enhancement**: Use specific components as needed rather than full enhancement
4. **Domain Filtering**: Specify domains to improve retrieval relevance

### Resource Requirements

- **Memory**: Additional ~100-200MB for knowledge base and embeddings
- **Processing**: Increased LLM API calls for multi-agent analysis
- **Network**: Optional sentence-transformers model download (~400MB)

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Missing dependencies
pip install sentence-transformers jinja2

# If sentence-transformers fails
# System will fallback to hash-based embeddings
```

#### LLM Client Issues
```python
# Verify client configuration
from causalllm.llm_client import get_llm_client
client = get_llm_client()
print(f"Using client: {type(client).__name__}")
```

#### Performance Issues
```python
# Use selective enhancement for faster processing
result = await core.enhanced_counterfactual_analysis(
    factual="scenario",
    intervention="intervention"
    # Don't specify domain to use general knowledge only
)

# Or use components individually
prompt_result = await prompting.generate_enhanced_counterfactual_prompt(...)
```

### Debug Mode

Enable detailed logging to troubleshoot issues:

```python
from causalllm.logging import get_logger
import logging

logger = get_logger("causalllm.tier1_debug")
logger.setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features

- **Tier 2 Enhancements**: Advanced causal discovery and graph learning
- **Custom Agent Types**: User-defined specialized agents
- **Knowledge Graph Integration**: Structured causal knowledge representation
- **Batch Processing**: Optimized batch analysis capabilities
- **Fine-tuning Interface**: Custom model adaptation for domain-specific use cases

### Contributing

To contribute to Tier 1 enhancements:

1. **Examples**: Add domain-specific examples to `llm_prompting.py`
2. **Knowledge**: Contribute causal knowledge documents to `causal_rag.py`
3. **Agents**: Propose new specialized agent types
4. **Testing**: Improve test coverage in `tests/test_tier1_enhancements.py`

## Citation

When using Tier 1 enhancements in research, please cite:

```bibtex
@software{causallm_tier1_2024,
  title={CausalLLM Tier 1 Enhancements: Advanced LLM Capabilities for Causal Inference},
  author={CausalLLM Development Team},
  year={2024},
  url={https://github.com/rdmurugan/causallm}
}
```

## License

Tier 1 enhancements are released under the same MIT License as the core CausalLLM framework.