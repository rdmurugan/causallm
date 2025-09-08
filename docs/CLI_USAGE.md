# CausalLLM Command Line Interface (CLI) Usage Guide

The CausalLLM CLI provides powerful causal inference capabilities directly from your terminal, making it easy to analyze data without writing Python code.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Commands Overview](#commands-overview)
4. [Command Reference](#command-reference)
5. [Data Formats](#data-formats)
6. [Domain Contexts](#domain-contexts)
7. [Examples](#examples)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Installation

```bash
# Basic installation (includes CLI)
pip install causallm

# Verify installation
causallm --version
causallm --help
```

---

## Quick Start

### 1. Prepare Your Data

Save your data in CSV format with clear column names:

```csv
age,gender,treatment,outcome,severity
25,M,1,8,2
35,F,0,5,3
45,M,1,9,4
```

### 2. Run Your First Analysis

```bash
# Discover causal relationships
causallm discover --data your_data.csv --variables "age,treatment,outcome"

# Estimate treatment effect
causallm effect --data your_data.csv --treatment treatment --outcome outcome
```

---

## Commands Overview

| Command | Purpose | Example |
|---------|---------|---------|
| `causallm discover` | Find causal relationships in data | `causallm discover --data data.csv --variables "age,income,health"` |
| `causallm effect` | Estimate causal effects | `causallm effect --data data.csv --treatment drug --outcome recovery` |
| `causallm counterfactual` | Generate counterfactual scenarios | `causallm counterfactual --data data.csv --intervention "treatment=1"` |
| `causallm info` | Display package information | `causallm info --examples` |
| `causallm web` | Launch web interface | `causallm web --port 8080` |

---

## Command Reference

### Global Options

Available for all commands:

```bash
--version              Show version and exit
--verbose, -v          Enable verbose logging  
--config, -c CONFIG    Configuration file path
--help, -h            Show help message
```

### `causallm discover`

Discover causal relationships in your data.

**Syntax:**
```bash
causallm discover --data DATA --variables VARIABLES [OPTIONS]
```

**Required Arguments:**
- `--data, -d`: Path to data file (CSV/JSON)
- `--variables, -var`: Comma-separated variable names

**Optional Arguments:**
- `--domain`: Domain context (healthcare, marketing, education, insurance, experimentation)
- `--method`: Discovery method (hybrid, llm, statistical) [default: hybrid]
- `--output, -o`: Output file path for results
- `--llm-provider`: LLM provider (openai, anthropic) [default: openai]
- `--llm-model`: LLM model name [default: gpt-4]

**Example:**
```bash
causallm discover \
  --data healthcare_data.csv \
  --variables "age,treatment,outcome,severity" \
  --domain healthcare \
  --method hybrid \
  --output discovery_results.json
```

**Output Structure:**
```json
{
  "command": "discover",
  "data_shape": [1000, 4],
  "variables": ["age", "treatment", "outcome", "severity"],
  "domain": "healthcare",
  "method": "hybrid",
  "results": {
    "relationships": [...],
    "confidence_scores": {...}
  }
}
```

### `causallm effect`

Estimate causal effects of treatments on outcomes.

**Syntax:**
```bash
causallm effect --data DATA --treatment TREATMENT --outcome OUTCOME [OPTIONS]
```

**Required Arguments:**
- `--data, -d`: Path to data file (CSV/JSON)
- `--treatment, -t`: Treatment variable name
- `--outcome, -y`: Outcome variable name

**Optional Arguments:**
- `--confounders`: Comma-separated confounder variables
- `--method`: Estimation method (backdoor, iv, regression_discontinuity) [default: backdoor]
- `--output, -o`: Output file path for results

**Example:**
```bash
causallm effect \
  --data clinical_trial.csv \
  --treatment drug_dosage \
  --outcome recovery_time \
  --confounders "age,gender,baseline_severity" \
  --method backdoor \
  --output effect_results.json
```

**Output Structure:**
```json
{
  "command": "effect",
  "data_shape": [500, 6],
  "treatment": "drug_dosage",
  "outcome": "recovery_time", 
  "confounders": ["age", "gender", "baseline_severity"],
  "method": "backdoor",
  "results": {
    "effect_estimate": -2.3,
    "confidence_interval": [-3.1, -1.5],
    "p_value": 0.002,
    "standard_error": 0.4
  }
}
```

### `causallm counterfactual`

Generate counterfactual scenarios to understand "what-if" situations.

**Syntax:**
```bash
causallm counterfactual --data DATA --intervention INTERVENTION [OPTIONS]
```

**Required Arguments:**
- `--data, -d`: Path to data file (CSV/JSON)
- `--intervention, -i`: Intervention specification (format: "variable=value")

**Optional Arguments:**
- `--samples, -n`: Number of counterfactual samples [default: 100]
- `--output, -o`: Output file path for results

**Example:**
```bash
causallm counterfactual \
  --data patient_outcomes.csv \
  --intervention "treatment=1" \
  --samples 200 \
  --output counterfactual_scenarios.json
```

**Intervention Format:**
- Numeric: `"dosage=50"` or `"age=35"`
- Categorical: `"treatment=active"` or `"group=control"`

**Output Structure:**
```json
{
  "command": "counterfactual",
  "data_shape": [300, 5],
  "intervention": {"treatment": 1},
  "samples": 200,
  "results": {
    "counterfactuals": [...],
    "summary_statistics": {...},
    "effect_distribution": [...]
  }
}
```

### `causallm info`

Display package information, examples, and help.

**Syntax:**
```bash
causallm info [OPTIONS]
```

**Optional Arguments:**
- `--enterprise`: Show enterprise features information
- `--domains`: List available domain contexts
- `--examples`: Show usage examples

**Examples:**
```bash
# Show basic package info
causallm info

# Show enterprise features
causallm info --enterprise

# List available domains
causallm info --domains

# Show usage examples
causallm info --examples
```

### `causallm web`

Launch the interactive web interface.

**Syntax:**
```bash
causallm web [OPTIONS]
```

**Optional Arguments:**
- `--port, -p`: Port number [default: 8080]
- `--host`: Host address [default: localhost]
- `--debug`: Enable debug mode

**Example:**
```bash
causallm web --port 3000 --host 0.0.0.0
```

---

## Data Formats

### CSV Format (Recommended)

```csv
participant_id,age,gender,treatment,outcome,baseline_score
1,25,M,1,8.5,6.2
2,35,F,0,5.1,5.8
3,45,M,1,9.2,7.1
```

**Best Practices:**
- Use clear, descriptive column names
- Avoid spaces in column names (use underscores)
- Include categorical variables as strings or numbers
- Handle missing values appropriately (empty cells or explicit NaN)

### JSON Format

```json
{
  "participant_id": [1, 2, 3],
  "age": [25, 35, 45],
  "gender": ["M", "F", "M"],
  "treatment": [1, 0, 1],
  "outcome": [8.5, 5.1, 9.2],
  "baseline_score": [6.2, 5.8, 7.1]
}
```

---

## Domain Contexts

Domain contexts provide specialized knowledge and interpretation for your analysis:

### Available Domains

| Domain | Description | Best For |
|--------|-------------|----------|
| `healthcare` | Medical and clinical analysis | Clinical trials, treatment effectiveness |
| `marketing` | Marketing and advertising | Campaign attribution, ROI analysis |
| `education` | Educational interventions | Student outcomes, policy evaluation |
| `insurance` | Risk assessment and insurance | Claims analysis, premium optimization |
| `experimentation` | A/B testing and experiments | Controlled experiments, design validation |

### Using Domain Context

```bash
# Healthcare example
causallm discover --data clinical_data.csv --domain healthcare --variables "age,treatment,recovery"

# Marketing example  
causallm effect --data campaign_data.csv --domain marketing --treatment channel --outcome conversion
```

---

## Examples

### Healthcare: Clinical Trial Analysis

```bash
# 1. Prepare data (clinical_trial.csv)
# columns: patient_id,age,gender,treatment_group,recovery_days,baseline_severity

# 2. Discover relationships
causallm discover \
  --data clinical_trial.csv \
  --variables "age,gender,treatment_group,recovery_days,baseline_severity" \
  --domain healthcare \
  --output discovery_results.json

# 3. Estimate treatment effect
causallm effect \
  --data clinical_trial.csv \
  --treatment treatment_group \
  --outcome recovery_days \
  --confounders "age,gender,baseline_severity" \
  --output treatment_effects.json

# 4. Generate counterfactuals
causallm counterfactual \
  --data clinical_trial.csv \
  --intervention "treatment_group=1" \
  --samples 500 \
  --output counterfactual_analysis.json
```

### Marketing: Campaign Attribution

```bash
# 1. Prepare data (marketing_data.csv)
# columns: customer_id,age,channel_exposure,spend,conversion,revenue

# 2. Analyze campaign effectiveness
causallm effect \
  --data marketing_data.csv \
  --treatment channel_exposure \
  --outcome revenue \
  --confounders "age,spend" \
  --domain marketing \
  --output campaign_effects.json

# 3. Counterfactual revenue analysis
causallm counterfactual \
  --data marketing_data.csv \
  --intervention "channel_exposure=1" \
  --samples 1000 \
  --output revenue_scenarios.json
```

### Educational Research

```bash
# 1. Prepare data (education_data.csv)
# columns: student_id,age,socioeconomic_status,intervention,test_score,attendance

# 2. Discover educational factors
causallm discover \
  --data education_data.csv \
  --variables "age,socioeconomic_status,intervention,test_score,attendance" \
  --domain education \
  --output educational_discovery.json

# 3. Measure intervention impact
causallm effect \
  --data education_data.csv \
  --treatment intervention \
  --outcome test_score \
  --confounders "age,socioeconomic_status,attendance" \
  --output intervention_effects.json
```

---

## Best Practices

### Data Preparation

1. **Clean Column Names**: Use descriptive, consistent naming
   ```bash
   # Good: age, treatment_status, outcome_score
   # Avoid: Age, Treatment Status, outcome
   ```

2. **Handle Missing Values**: Decide on strategy before analysis
   ```bash
   # Option 1: Remove rows with missing values
   # Option 2: Impute missing values
   # Option 3: Use analysis methods that handle missing data
   ```

3. **Appropriate Data Types**: Ensure variables have correct types
   - Numeric: ages, scores, measurements
   - Categorical: groups, categories, binary indicators

### Analysis Strategy

1. **Start with Discovery**: Understand relationships first
   ```bash
   causallm discover --data your_data.csv --variables "var1,var2,var3"
   ```

2. **Use Domain Context**: Leverage specialized knowledge
   ```bash
   --domain healthcare  # For medical data
   --domain marketing   # For business data
   ```

3. **Control for Confounders**: Include relevant control variables
   ```bash
   --confounders "age,gender,baseline_score"
   ```

4. **Generate Counterfactuals**: Explore "what-if" scenarios
   ```bash
   causallm counterfactual --data data.csv --intervention "treatment=1"
   ```

### Output Management

1. **Save Results**: Always specify output files for important analyses
   ```bash
   --output results_$(date +%Y%m%d).json
   ```

2. **Version Control**: Track different analysis versions
   ```bash
   mkdir analysis_results
   causallm discover --data data.csv --output analysis_results/discovery_v1.json
   ```

3. **Documentation**: Keep notes about analysis decisions and findings

---

## Troubleshooting

### Common Issues

#### 1. File Not Found
```bash
Error: Data file not found: data.csv
```
**Solution**: Check file path and ensure file exists
```bash
# Check current directory
ls -la
# Use absolute path if needed
causallm discover --data /full/path/to/data.csv --variables "var1,var2"
```

#### 2. Invalid Variable Names
```bash
Error: Variable 'treatment_status' not found in data
```
**Solution**: Check column names in your data
```bash
# Check your CSV header row
head -1 data.csv
# Update command with correct variable names
```

#### 3. Insufficient Data
```bash
Warning: Small sample size may affect reliability
```
**Solution**: 
- Ensure adequate sample size (>100 observations recommended)
- Consider collecting more data
- Use appropriate statistical methods for small samples

#### 4. Memory Issues with Large Files
```bash
Error: Memory error processing large dataset
```
**Solution**: Use the Python API with chunking for very large datasets, or split your file into smaller parts

#### 5. LLM API Errors
```bash
Error: OpenAI API key not found
```
**Solution**: 
- Set environment variable: `export OPENAI_API_KEY=your_key`
- Use statistical-only methods: `--method statistical`

### Getting Help

1. **Built-in Help**:
   ```bash
   causallm --help
   causallm discover --help
   causallm effect --help
   ```

2. **Examples and Documentation**:
   ```bash
   causallm info --examples
   causallm info --domains
   ```

3. **Verbose Output**:
   ```bash
   causallm --verbose discover --data data.csv --variables "var1,var2"
   ```

4. **Community Support**:
   - GitHub Issues: [Report problems](https://github.com/rdmurugan/causallm/issues)
   - Discussions: [Ask questions](https://github.com/rdmurugan/causallm/discussions)

---

## Advanced Usage

### Configuration Files

Create a configuration file for repeated analyses:

```json
{
  "llm_provider": "openai",
  "llm_model": "gpt-4",
  "default_domain": "healthcare",
  "output_format": "json",
  "verbose": true
}
```

Use with:
```bash
causallm --config my_config.json discover --data data.csv --variables "var1,var2"
```

### Environment Variables

Set common parameters via environment variables:

```bash
export CAUSALLM_DEFAULT_DOMAIN=healthcare
export CAUSALLM_OUTPUT_DIR=./results
export CAUSALLM_VERBOSE=true
export OPENAI_API_KEY=your_key_here

# Commands will use these defaults
causallm discover --data data.csv --variables "age,treatment,outcome"
```

### Batch Processing

Process multiple files:

```bash
#!/bin/bash
for file in data/*.csv; do
  basename=$(basename "$file" .csv)
  causallm effect \
    --data "$file" \
    --treatment treatment \
    --outcome outcome \
    --output "results/${basename}_effects.json"
done
```

---

**Ready to start analyzing? Try the quick start examples or explore the web interface with `causallm web`!**