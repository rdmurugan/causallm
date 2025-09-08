# CausalLLM Web Interface User Guide

The CausalLLM web interface provides an intuitive, point-and-click way to perform causal inference analysis without writing any code. Perfect for researchers, business analysts, and anyone who prefers visual interfaces.

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Getting Started](#getting-started)
3. [Interface Overview](#interface-overview)
4. [Data Input & Management](#data-input--management)
5. [Causal Discovery](#causal-discovery)
6. [Effect Estimation](#effect-estimation)
7. [Counterfactual Analysis](#counterfactual-analysis)
8. [Documentation & Examples](#documentation--examples)
9. [Export & Sharing](#export--sharing)
10. [Configuration & Settings](#configuration--settings)
11. [Troubleshooting](#troubleshooting)

---

## Installation & Setup

### Installation

```bash
# Install CausalLLM with web interface support
pip install "causallm[ui]"

# Verify installation
causallm web --help
```

### Launching the Web Interface

```bash
# Launch with default settings (localhost:8080)
causallm web

# Customize host and port
causallm web --host 0.0.0.0 --port 3000

# Enable debug mode for development
causallm web --debug
```

After launching, open your browser and navigate to the displayed URL (typically `http://localhost:8080`).

---

## Getting Started

### First Launch

When you first open the web interface, you'll see:

1. **Welcome Screen**: Overview of CausalLLM capabilities
2. **Navigation Tabs**: Data, Discovery, Effects, Counterfactuals, Documentation
3. **Sidebar**: Configuration options and settings

### Quick Start Workflow

1. **Upload Data** → Choose your CSV/JSON file or select sample data
2. **Discover Relationships** → Find causal connections in your data
3. **Estimate Effects** → Quantify impact of treatments/interventions
4. **Explore Counterfactuals** → Generate "what-if" scenarios
5. **Export Results** → Download analysis and visualizations

---

## Interface Overview

### Main Navigation

The interface is organized into five main tabs:

| Tab | Icon | Purpose |
|-----|------|---------|
| **📊 Data** | 📊 | Upload, preview, and manage datasets |
| **🔍 Discovery** | 🔍 | Discover causal relationships automatically |
| **⚡ Effects** | ⚡ | Estimate causal effects and treatment impacts |
| **🔮 Counterfactuals** | 🔮 | Generate and analyze counterfactual scenarios |
| **📖 Documentation** | 📖 | Help, examples, and API reference |

### Sidebar Configuration

The left sidebar contains:

- **LLM Settings**: Provider, model selection
- **Analysis Settings**: Methods, domain context
- **Performance Options**: Async processing, logging levels

---

## Data Input & Management

### Data Upload Methods

#### 1. File Upload
- **Supported Formats**: CSV, JSON
- **Maximum Size**: Depends on your system memory
- **Features**: Drag-and-drop or click to browse

```text
Supported CSV structure:
participant_id,age,gender,treatment,outcome
1,25,M,1,8.5
2,35,F,0,5.1
```

#### 2. Sample Datasets

Choose from built-in sample datasets:

- **Healthcare Example**: Clinical trial with treatment outcomes
- **Marketing Example**: Campaign effectiveness analysis
- **Custom Datasets**: Upload your own for repeated use

#### 3. Manual Entry

For small datasets, enter data directly:

```json
{
  "age": [25, 35, 45, 55],
  "treatment": [1, 0, 1, 1],
  "outcome": [8.5, 5.1, 9.2, 7.8]
}
```

### Data Preview & Validation

Once uploaded, you'll see:

- **Data Preview**: First 10 rows with column types
- **Data Summary**: Shape, missing values, basic statistics
- **Column Information**: Detected data types and distributions

**Data Quality Checks:**
- ✅ Missing value detection
- ✅ Column type validation
- ✅ Sample size recommendations
- ⚠️ Warnings for potential issues

---

## Causal Discovery

The Discovery tab helps you find causal relationships in your data automatically.

### Variable Selection

1. **Choose Variables**: Select columns for analysis
   - Click checkboxes next to variable names
   - Use "Select All" for comprehensive analysis
   - Minimum 3 variables recommended

2. **Domain Context**: Choose your field
   - **Healthcare**: Clinical trials, treatment analysis
   - **Marketing**: Campaign attribution, customer analytics
   - **Education**: Intervention studies, policy evaluation
   - **Insurance**: Risk assessment, claims analysis
   - **Experimentation**: A/B testing, experimental design

### Analysis Options

**Discovery Method:**
- **Hybrid** (Recommended): Combines statistical and LLM approaches
- **Statistical**: PC algorithm and independence tests only
- **LLM**: Language model-based discovery only

**Advanced Settings:**
- **Significance Level**: Statistical threshold (default: 0.05)
- **Max Conditioning Set Size**: Complexity control (default: 3)

### Running Discovery Analysis

1. Click **"Run Causal Discovery"** button
2. Watch progress indicator and status messages
3. View results in three tabs:

#### Graph Tab
- **Visual Network**: Interactive causal graph
- **Node Interactions**: Click nodes to see connections
- **Edge Information**: Hover over edges for strength/confidence

#### Relationships Tab
- **Discovered Relationships**: List of causal connections
- **Confidence Scores**: Strength of each relationship
- **Domain Insights**: Contextual interpretation

#### Statistics Tab
- **Technical Details**: P-values, test statistics
- **Method Information**: Which algorithms were used
- **Performance Metrics**: Processing time, iterations

### Interpreting Results

**Graph Elements:**
- **Nodes**: Your variables
- **Directed Edges**: Causal relationships (A → B means A causes B)
- **Edge Colors**: Relationship strength (darker = stronger)
- **Node Sizes**: Variable importance

**Relationship Types:**
- **Direct Causation**: X directly causes Y
- **Confounding**: Common cause of multiple variables
- **Mediation**: X causes Y through intermediate variable Z

---

## Effect Estimation

Quantify the causal impact of treatments or interventions on outcomes.

### Variable Setup

1. **Treatment Variable**: The intervention or exposure
   - Example: drug_dosage, campaign_exposure, training_program
   
2. **Outcome Variable**: The result you're measuring
   - Example: recovery_time, sales_revenue, test_score

3. **Confounding Variables**: Variables to control for
   - Example: age, gender, baseline_severity
   - Use Ctrl/Cmd+Click to select multiple

### Method Selection

**Estimation Methods:**

| Method | When to Use | Assumptions |
|--------|-------------|-------------|
| **Backdoor** | Default choice | No unmeasured confounders |
| **Instrumental Variables** | When treatment isn't random | Valid instrument available |
| **Regression Discontinuity** | Sharp cutoff in treatment | Discontinuity in assignment |

### Running Effect Analysis

1. Configure variables and method
2. Click **"Estimate Causal Effect"** 
3. Monitor progress and view results

### Results Interpretation

**Key Metrics:**
- **Effect Estimate**: Average treatment effect (with units)
- **Confidence Interval**: Uncertainty range (typically 95%)
- **P-value**: Statistical significance
- **Standard Error**: Precision measure

**Example Results:**
```text
Effect Estimate: -2.3 days
Confidence Interval: [-3.1, -1.5] days
P-value: 0.002
Interpretation: Treatment reduces recovery time by 2.3 days on average (highly significant)
```

**Visualizations:**
- **Effect Size Plot**: Visual representation of effect and confidence interval
- **Distribution Plot**: Treatment vs. control outcome distributions
- **Scatter Plot**: Relationship between treatment and outcome

---

## Counterfactual Analysis

Explore "what-if" scenarios to understand potential outcomes under different conditions.

### Intervention Setup

1. **Select Variable**: Choose which variable to modify
2. **Set Value**: Specify the counterfactual value
   - **Numeric Variables**: Enter specific number
   - **Categorical Variables**: Select from dropdown

**Examples:**
- `treatment = 1` (give treatment to all)
- `dosage = 50` (set dosage to 50mg)
- `channel = "social_media"` (assign to social media channel)

### Sample Size Configuration

- **Number of Samples**: How many counterfactual scenarios to generate
- **Range**: 10-1000 (larger = more stable estimates)
- **Recommendation**: 100+ for reliable results

### Running Counterfactual Analysis

1. Specify intervention and sample size
2. Click **"Generate Counterfactuals"**
3. View results in multiple formats

### Results & Visualization

**Summary Statistics:**
- **Mean Outcome**: Average under counterfactual scenario
- **Standard Deviation**: Variability in outcomes
- **Quantiles**: 25th, 50th, 75th percentile outcomes

**Comparison with Observed:**
- **Counterfactual vs. Observed**: Side-by-side comparison
- **Difference Distribution**: Distribution of individual-level effects
- **Effect Magnitude**: Average difference and confidence interval

**Interactive Visualizations:**
- **Histogram**: Distribution of counterfactual outcomes
- **Box Plot**: Quartiles and outliers
- **Scatter Plot**: Individual counterfactual vs. observed outcomes

---

## Documentation & Examples

The Documentation tab provides comprehensive help and examples.

### Getting Started Section

- **Welcome Guide**: Step-by-step introduction
- **Key Concepts**: Causal inference fundamentals
- **Workflow Overview**: Typical analysis process

### API Reference Section

- **Main Classes**: `CausalLLM`, `EnhancedCausalLLM` documentation
- **Key Methods**: Parameter descriptions and examples
- **Return Types**: Understanding analysis results

### Examples Section

**Domain-Specific Examples:**

#### Healthcare Analysis
```python
# Load patient data
data = pd.read_csv('patient_outcomes.csv')

# Initialize CausalLLM
causal_llm = CausalLLM()

# Discover relationships
results = causal_llm.discover_causal_relationships(
    data=data,
    variables=['age', 'treatment', 'outcome'],
    domain_context='healthcare'
)
```

#### Marketing Attribution
```python
# Estimate campaign effectiveness
effect = causal_llm.estimate_causal_effect(
    data=marketing_data,
    treatment='campaign_exposure',
    outcome='conversion',
    confounders=['age', 'income', 'previous_purchases']
)
```

#### Educational Research
```python
# Generate counterfactuals
scenarios = causal_llm.generate_counterfactuals(
    data=student_data,
    intervention={'tutoring': 1}
)
```

---

## Export & Sharing

### Download Options

**Analysis Results:**
- **JSON Format**: Complete results with metadata
- **CSV Format**: Tabular results for spreadsheet analysis
- **PDF Report**: Formatted analysis summary

**Visualizations:**
- **PNG Images**: High-resolution plots
- **SVG Vector**: Scalable graphics
- **Interactive HTML**: Shareable interactive plots

### Sharing & Collaboration

1. **Session URLs**: Share analysis configurations
2. **Result Links**: Direct links to specific results
3. **Report Generation**: Automated analysis summaries
4. **Export Codes**: Reproducible analysis scripts

### Integration Options

**Export to:**
- **Jupyter Notebooks**: Generate notebook with your analysis
- **R Scripts**: Convert to R code for further analysis
- **PowerPoint**: Create presentation slides
- **Dashboard Tools**: Export to Tableau, Power BI formats

---

## Configuration & Settings

### LLM Configuration

**Provider Selection:**
- **OpenAI**: GPT-4, GPT-3.5-turbo models
- **Anthropic**: Claude models
- **Local Models**: Support for local deployments

**API Setup:**
- Enter API keys in settings
- Test connection before analysis
- Fallback to statistical methods if unavailable

### Performance Settings

**Async Processing:**
- Enable for large datasets (>10,000 rows)
- Parallel computation for faster results
- Progress indicators for long-running analyses

**Cache Settings:**
- **Memory Cache**: Faster repeat analyses
- **Disk Cache**: Persistent across sessions
- **Cache Size**: Configurable storage limits

### Analysis Preferences

**Default Settings:**
- **Significance Level**: Statistical threshold
- **Confidence Level**: For confidence intervals
- **Bootstrap Iterations**: For uncertainty quantification

**Display Options:**
- **Number Format**: Decimal places, scientific notation
- **Plot Themes**: Color schemes, fonts
- **Table Pagination**: Rows per page

---

## Troubleshooting

### Common Issues

#### 1. Data Upload Problems

**Issue**: "Failed to load data file"
```
Solutions:
✅ Check file format (CSV/JSON only)
✅ Verify file size (<100MB recommended)
✅ Ensure proper encoding (UTF-8)
✅ Check for special characters in headers
```

#### 2. Analysis Errors

**Issue**: "Insufficient data for analysis"
```
Solutions:
✅ Ensure adequate sample size (>100 observations)
✅ Check for missing values in key variables
✅ Verify variable types are appropriate
✅ Use simpler models for small datasets
```

#### 3. Memory Issues

**Issue**: "Browser becomes unresponsive"
```
Solutions:
✅ Reduce dataset size or use sampling
✅ Enable async processing in settings
✅ Clear browser cache and reload
✅ Use desktop browser instead of mobile
```

#### 4. Visualization Problems

**Issue**: "Plots not displaying correctly"
```
Solutions:
✅ Update browser to latest version
✅ Enable JavaScript
✅ Check browser console for errors
✅ Try different browser (Chrome/Firefox recommended)
```

### Performance Optimization

**For Large Datasets:**
1. Enable async processing
2. Use statistical-only methods (faster than LLM)
3. Reduce number of variables
4. Consider sampling large datasets

**For Slow Analysis:**
1. Check internet connection (affects LLM calls)
2. Use cached results when available
3. Simplify analysis parameters
4. Consider using CLI for batch processing

### Getting Help

**Built-in Help:**
- **Info Tooltips**: Hover over ⓘ icons
- **Examples Tab**: Working code examples
- **Documentation Tab**: Comprehensive guides

**Community Support:**
- **GitHub Issues**: [Report bugs](https://github.com/rdmurugan/causallm/issues)
- **Discussions**: [Ask questions](https://github.com/rdmurugan/causallm/discussions)
- **Email**: durai@infinidatum.net

**Debug Information:**
- Check browser console (F12) for error messages
- Enable verbose logging in settings
- Screenshot issues for better support

---

## Advanced Features

### Custom Data Generators

Generate synthetic datasets for testing:

```python
# Available in Documentation tab
from causallm.domains import HealthcareDomain

healthcare = HealthcareDomain()
data = healthcare.generate_clinical_trial_data(n_patients=1000)
```

### Configuration Profiles

Save analysis settings:
- **Development Profile**: Fast settings for testing
- **Production Profile**: High-quality settings for final analysis
- **Custom Profiles**: Your personalized configurations

### Batch Analysis

Process multiple datasets:
1. Upload multiple files
2. Configure analysis template
3. Run batch processing
4. Download combined results

### API Integration

Access programmatic interface:
- **REST API**: HTTP endpoints for integration
- **Python SDK**: Direct library access
- **Export Code**: Generate Python scripts from web analysis

---

## Best Practices

### Data Preparation

1. **Clean Data**: Remove duplicates, handle missing values
2. **Clear Names**: Use descriptive variable names
3. **Appropriate Types**: Ensure correct data types
4. **Sample Size**: Aim for >100 observations minimum

### Analysis Strategy

1. **Start Simple**: Begin with basic discovery
2. **Use Domain Context**: Leverage specialized knowledge
3. **Validate Results**: Cross-check with multiple methods
4. **Document Decisions**: Save configuration and results

### Result Interpretation

1. **Statistical Significance**: Don't ignore p-values
2. **Practical Significance**: Consider effect size magnitude
3. **Confidence Intervals**: Understand uncertainty
4. **Domain Knowledge**: Apply subject matter expertise

---

**Ready to start your causal analysis journey? Launch the web interface with `causallm web` and follow the guided workflow!**