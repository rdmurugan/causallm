# CausalLLM Data Manager

The **CausalDataManager** is a comprehensive data handling component designed specifically for causal analysis workflows. It provides utilities for loading, cleaning, validating, and analyzing datasets to ensure they're ready for causal inference with CausalLLM.

## Overview

Working with real-world data for causal analysis involves several challenges:
- **Messy variable names** that don't match DAG node definitions
- **Multiple data formats** (CSV, Parquet, Excel, JSON)
- **Missing required variables** for causal models
- **Need for conditional analysis** and data slicing
- **Data validation** and quality checks

CausalDataManager solves these challenges by providing a unified interface for data preprocessing and analysis.

## Key Features

### 🔧 **Multi-Format Data Loading**
- Load data from CSV, Parquet, Excel, and JSON files
- Automatic format detection from file extensions
- Comprehensive error handling and validation
- Memory-efficient processing with progress tracking

### 🏷️ **Intelligent Variable Normalization**
- Clean messy column names using standard conventions
- Automatic fuzzy matching to DAG variables
- Custom mapping support for domain-specific needs
- Preserve original-to-normalized name mappings

### ✅ **Causal Variable Validation**
- Validate that required DAG variables exist in dataset
- Suggest similar column names for missing variables
- Flexible validation with strict and permissive modes
- Comprehensive validation reporting

### 🔍 **Advanced Data Filtering**
- Conditional filtering with multiple operators (AND/OR)
- Support for exact matches, ranges, and lists
- Efficient pandas-based filtering
- Maintain data lineage and provenance

### 📊 **Distribution Analysis**
- Comprehensive statistical analysis of variables
- Conditional distributions for causal analysis
- Support for numeric and categorical variables
- Missing value analysis and reporting

### 🔌 **CausalLLMCore Integration**
- Seamless integration with causal reasoning workflows
- Automatic data preparation for causal models
- Validation against DAG structure requirements
- Export capabilities for processed datasets

## Installation and Setup

The CausalDataManager is included with CausalLLM. For optional features:

```bash
# For Parquet support
pip install pyarrow

# For Excel support  
pip install openpyxl

# All optional dependencies
pip install pandas[all]
```

## Quick Start

### Basic Usage

```python
from causalllm.data_manager import CausalDataManager

# Initialize with DAG variables (optional)
dag_variables = {
    'age': 'Patient age in years',
    'treatment': 'Medical treatment received', 
    'outcome': 'Health outcome score'
}

dm = CausalDataManager(dag_variables)

# Load data
data = dm.load_data('healthcare_study.csv')
print(f"Loaded {data.shape[0]} rows, {data.shape[1]} columns")

# Normalize variable names
dm.normalize_variable_names(mapping_strategy="auto")
processed_data = dm.apply_variable_mapping()

# Validate causal variables
validation = dm.validate_causal_variables()
print(f"Validation passed: {validation['is_valid']}")
```

### One-Step Data Loading

```python
from causalllm.data_manager import load_causal_data

# Load and prepare data in one step
data_manager, data = load_causal_data(
    'study_data.csv',
    dag_variables=dag_variables,
    normalize_names=True,
    validate_variables=True
)
```

## Detailed Usage Guide

### 1. Data Loading

#### Multi-Format Support

```python
# CSV files
data = dm.load_data('data.csv')

# Parquet files  
data = dm.load_data('data.parquet')

# Excel files
data = dm.load_data('data.xlsx', sheet_name='Sheet1')

# JSON files
data = dm.load_data('data.json')

# Auto-detect format
data = dm.load_data('data_file')  # Extension determines format
```

#### Advanced Loading Options

```python
# CSV with custom options
data = dm.load_data(
    'data.csv', 
    sep=';',
    encoding='utf-8',
    na_values=['NULL', 'missing', '']
)

# Parquet with column selection
data = dm.load_data(
    'large_dataset.parquet',
    columns=['age', 'treatment', 'outcome']
)
```

### 2. Variable Name Normalization

#### Automatic Normalization Strategies

```python
# Clean strategy: Standardize names without DAG matching
mapping = dm.normalize_variable_names(mapping_strategy="clean")

# Auto strategy: Fuzzy match to DAG variables
mapping = dm.normalize_variable_names(mapping_strategy="auto") 

# Custom strategy: Use your own mapping
custom_mapping = {
    'Patient_Age_Years': 'age',
    'Treatment_Status': 'treatment',
    'Health_Score_Final': 'outcome'
}
mapping = dm.normalize_variable_names(
    mapping_strategy="custom",
    custom_mapping=custom_mapping
)
```

#### Understanding the Mapping Process

```python
# Check original column names
print("Original columns:", dm.original_columns)

# See the mapping that was created
print("Variable mapping:", dm.variable_mapping)

# Apply the mapping
processed_data = dm.apply_variable_mapping()
print("New columns:", list(processed_data.columns))

# Access reverse mapping (normalized -> original)
print("Reverse mapping:", dm.reverse_mapping)
```

### 3. Variable Validation

#### Basic Validation

```python
# Validate all DAG variables
validation = dm.validate_causal_variables()

print(f"Valid: {validation['is_valid']}")
print(f"Present: {validation['present_variables']}")
print(f"Missing: {validation['missing_variables']}")
print(f"Coverage: {validation['coverage_ratio']:.2%}")
```

#### Custom Validation

```python
# Validate specific variables
required_vars = ['age', 'treatment', 'outcome']
validation = dm.validate_causal_variables(
    required_variables=required_vars,
    strict=False  # Don't raise exception on failure
)

# Get suggestions for missing variables
if validation['suggestions']:
    print("Suggestions for missing variables:")
    for missing, suggestions in validation['suggestions'].items():
        print(f"  {missing} -> {suggestions}")
```

#### Strict Validation

```python
try:
    # This will raise ValueError if validation fails
    validation = dm.validate_causal_variables(strict=True)
    print("✅ All required variables present")
except ValueError as e:
    print(f"❌ Validation failed: {e}")
```

### 4. Conditional Data Filtering

#### Single Conditions

```python
# Exact value matching
treated_patients = dm.get_conditional_data({
    'treatment': True
})

# List of values (isin operation)
educated_patients = dm.get_conditional_data({
    'education': ['Bachelor', 'Master', 'PhD']
})

# Range conditions (inclusive)
middle_aged = dm.get_conditional_data({
    'age': (30, 60)
})
```

#### Multiple Conditions

```python
# AND conditions (all must be true)
high_risk = dm.get_conditional_data({
    'age': (65, 100),
    'treatment': False,
    'income': (0, 50000)
}, operator="and")

# OR conditions (any can be true)
priority_patients = dm.get_conditional_data({
    'age': (0, 18),    # Children
    'age': (65, 100),  # Elderly  
    'condition': ['critical']
}, operator="or")
```

#### Complex Filtering Examples

```python
# Healthcare example: High-risk untreated patients
high_risk_untreated = dm.get_conditional_data({
    'age': (60, float('inf')),
    'treatment_received': False,
    'comorbidities': ['diabetes', 'hypertension', 'heart_disease'],
    'insurance_status': ['uninsured', 'underinsured']
}, operator="and")

# Marketing example: Target demographic
target_customers = dm.get_conditional_data({
    'age': (25, 45),
    'income': (50000, 150000),
    'region': ['urban', 'suburban'],
    'previous_purchase': True
}, operator="and")
```

### 5. Distribution Analysis

#### Numeric Variables

```python
# Basic distribution
income_stats = dm.get_variable_distribution('income')

print(f"Mean: ${income_stats['mean']:,.2f}")
print(f"Median: ${income_stats['median']:,.2f}")
print(f"Std Dev: ${income_stats['std']:,.2f}")
print(f"Range: ${income_stats['min']:,.0f} - ${income_stats['max']:,.0f}")
print(f"Quartiles: Q1={income_stats['quartiles']['q1']:,.0f}, Q3={income_stats['quartiles']['q3']:,.0f}")
```

#### Categorical Variables

```python
# Categorical distribution
edu_stats = dm.get_variable_distribution('education')

print(f"Unique values: {edu_stats['unique_values']}")
print(f"Most common: {edu_stats['mode']}")
print("Value counts:")
for value, count in edu_stats['value_counts'].items():
    print(f"  {value}: {count}")
```

#### Conditional Distributions

```python
# Income distribution for treated vs untreated
treated_income = dm.get_variable_distribution(
    'income',
    conditional_on={'treatment': True}
)

untreated_income = dm.get_variable_distribution(
    'income', 
    conditional_on={'treatment': False}
)

print(f"Treated group income: ${treated_income['mean']:,.0f}")
print(f"Untreated group income: ${untreated_income['mean']:,.0f}")
print(f"Difference: ${treated_income['mean'] - untreated_income['mean']:,.0f}")
```

### 6. Integration with CausalLLMCore

#### Seamless Workflow Integration

```python
from causalllm.core import CausalLLMCore
from causalllm.llm_client import get_llm_client

# Define causal model
dag_variables = {
    'education': 'Education level achieved',
    'income': 'Annual household income', 
    'healthcare_access': 'Healthcare access score',
    'health_outcome': 'Health outcome index'
}

dag_edges = [
    ('education', 'income'),
    ('income', 'healthcare_access'),
    ('healthcare_access', 'health_outcome')
]

# Load and prepare data
data_manager, data = load_causal_data(
    'healthcare_survey.csv',
    dag_variables=dag_variables,
    normalize_names=True,
    validate_variables=True
)

# Create CausalLLMCore with validated data
llm_client = get_llm_client("openai")
core = CausalLLMCore(
    context="Healthcare access and outcomes analysis",
    variables=dag_variables,
    dag_edges=dag_edges,
    llm_client=llm_client
)

# Perform causal analysis on filtered data subsets
high_income_data = data_manager.get_conditional_data({
    'income': (80000, float('inf'))
})

# Run counterfactual analysis
result = core.simulate_counterfactual(
    factual=f"High-income patients (n={len(high_income_data)}) have current healthcare access",
    intervention="All patients receive premium healthcare access",
    instruction="Focus on health outcome improvements"
)
```

### 7. Data Export

#### Multiple Format Export

```python
# Export processed data
dm.export_processed_data('processed_data.csv')
dm.export_processed_data('processed_data.parquet')
dm.export_processed_data('processed_data.json')
dm.export_processed_data('processed_data.xlsx')

# Export with custom options
dm.export_processed_data(
    'final_dataset.csv',
    index=False,
    float_format='%.3f'
)
```

## Advanced Features

### Sample Data Generation

```python
from causalllm.data_manager import create_sample_causal_data

# Create synthetic causal dataset for testing
sample_data = create_sample_causal_data()
print(f"Generated {len(sample_data)} samples with realistic causal relationships")

# Save sample data
sample_data = create_sample_causal_data('sample_healthcare_data.csv')
```

### Data Quality Assessment

```python
# Comprehensive data summary
summary = dm.get_data_summary()

print(f"Shape: {summary['shape']}")
print(f"Data types: {summary['data_types']}")
print(f"Missing data: {summary['missing_data']['total_missing']} values")
print(f"Memory usage: {summary['memory_usage']['total_mb']:.1f} MB")
print(f"Variable mapping: {summary['variable_mapping']['has_mapping']}")

if 'dag_validation' in summary:
    print(f"DAG validation: {summary['dag_validation']['is_valid']}")
```

### Memory-Efficient Processing

```python
# For large datasets, use chunked processing
for chunk in pd.read_csv('large_dataset.csv', chunksize=10000):
    dm_chunk = CausalDataManager(dag_variables)
    dm_chunk.data = chunk
    dm_chunk._collect_data_info()
    
    # Process chunk
    dm_chunk.normalize_variable_names()
    processed_chunk = dm_chunk.apply_variable_mapping()
    
    # Analyze chunk
    validation = dm_chunk.validate_causal_variables()
    print(f"Chunk validation: {validation['coverage_ratio']:.2%}")
```

## Real-World Examples

### Healthcare Study Data Preparation

```python
# Healthcare outcomes study
healthcare_dag = {
    'age': 'Patient age in years',
    'socioeconomic_status': 'Socioeconomic status indicator',
    'insurance_type': 'Type of health insurance',
    'treatment_access': 'Access to specialized treatment',
    'health_outcome': 'Primary health outcome measure'
}

healthcare_edges = [
    ('age', 'health_outcome'),
    ('socioeconomic_status', 'insurance_type'),
    ('insurance_type', 'treatment_access'),
    ('treatment_access', 'health_outcome'),
    ('socioeconomic_status', 'treatment_access')
]

# Load messy healthcare data
dm = CausalDataManager(healthcare_dag)
data = dm.load_data('messy_healthcare_survey.csv')

# Clean and validate
dm.normalize_variable_names(mapping_strategy="auto")
validation = dm.validate_causal_variables()

if not validation['is_valid']:
    print("Missing variables:", validation['missing_variables'])
    print("Suggestions:", validation['suggestions'])

# Analyze subpopulations
elderly_patients = dm.get_conditional_data({'age': (65, float('inf'))})
uninsured_patients = dm.get_conditional_data({'insurance_type': ['uninsured']})

# Generate insights
elderly_outcomes = dm.get_variable_distribution(
    'health_outcome', 
    conditional_on={'age': (65, float('inf'))}
)
print(f"Elderly patient outcomes: mean={elderly_outcomes['mean']:.2f}")
```

### Marketing Campaign Analysis

```python
# Marketing campaign effectiveness
marketing_dag = {
    'customer_age': 'Customer age group',
    'income_level': 'Annual income bracket',  
    'previous_purchases': 'Number of previous purchases',
    'campaign_exposure': 'Marketing campaign exposure level',
    'purchase_decision': 'Made purchase after campaign'
}

marketing_edges = [
    ('customer_age', 'income_level'),
    ('income_level', 'previous_purchases'),
    ('campaign_exposure', 'purchase_decision'),
    ('previous_purchases', 'purchase_decision'),
    ('income_level', 'campaign_exposure')
]

# Load campaign data
dm = CausalDataManager(marketing_dag)
campaign_data = dm.load_data('campaign_results.xlsx', sheet_name='Results')

# Segment analysis
high_value_customers = dm.get_conditional_data({
    'income_level': ['high', 'very_high'],
    'previous_purchases': (5, float('inf'))
}, operator="and")

# Campaign effectiveness by segment
campaign_effect = dm.get_variable_distribution(
    'purchase_decision',
    conditional_on={'campaign_exposure': ['high']}
)
```

### Financial Risk Assessment

```python
# Credit risk modeling
risk_dag = {
    'credit_score': 'Credit score (FICO)',
    'debt_to_income': 'Debt-to-income ratio',
    'employment_status': 'Employment status',
    'loan_amount': 'Requested loan amount',
    'default_risk': 'Probability of default'
}

risk_edges = [
    ('credit_score', 'default_risk'),
    ('debt_to_income', 'default_risk'), 
    ('employment_status', 'debt_to_income'),
    ('loan_amount', 'default_risk'),
    ('employment_status', 'default_risk')
]

# Load financial data
dm = CausalDataManager(risk_dag)
financial_data = dm.load_data('loan_applications.parquet')

# Risk segmentation
high_risk_applications = dm.get_conditional_data({
    'credit_score': (300, 650),
    'debt_to_income': (0.4, 1.0),
    'employment_status': ['unemployed', 'part_time']
}, operator="or")

print(f"High-risk applications: {len(high_risk_applications)}")

# Risk distribution analysis
risk_by_employment = dm.get_variable_distribution(
    'default_risk',
    conditional_on={'employment_status': ['full_time']}
)
```

## Best Practices

### 1. Data Preparation Workflow

```python
# Recommended workflow
def prepare_causal_data(file_path, dag_variables, dag_edges):
    """Complete data preparation workflow."""
    
    # 1. Load data
    dm = CausalDataManager(dag_variables)
    data = dm.load_data(file_path)
    
    # 2. Get data summary
    summary = dm.get_data_summary()
    print(f"Loaded: {summary['shape']} with {summary['missing_data']['total_missing']} missing values")
    
    # 3. Normalize variables
    dm.normalize_variable_names(mapping_strategy="auto")
    data = dm.apply_variable_mapping()
    
    # 4. Validate causal variables
    validation = dm.validate_causal_variables()
    if not validation['is_valid']:
        print(f"⚠️ Missing variables: {validation['missing_variables']}")
        # Handle missing variables...
    
    # 5. Create CausalLLMCore
    from causalllm.core import CausalLLMCore
    core = CausalLLMCore(
        context="Prepared dataset for causal analysis",
        variables=dag_variables,
        dag_edges=dag_edges
    )
    
    return dm, data, core, validation
```

### 2. Handling Missing Variables

```python
# Strategy for handling missing variables
validation = dm.validate_causal_variables()

if not validation['is_valid']:
    for missing_var in validation['missing_variables']:
        if missing_var in validation['suggestions']:
            suggestions = validation['suggestions'][missing_var]
            print(f"Consider mapping '{suggestions[0]}' to '{missing_var}'")
            
            # Create custom mapping
            custom_mapping = dm.variable_mapping.copy()
            custom_mapping[suggestions[0]] = missing_var
            
            dm.normalize_variable_names(
                mapping_strategy="custom",
                custom_mapping=custom_mapping
            )
```

### 3. Large Dataset Handling

```python
# Memory-efficient processing for large datasets
def process_large_dataset(file_path, dag_variables, chunk_size=10000):
    """Process large datasets in chunks."""
    
    results = []
    
    for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
        print(f"Processing chunk {chunk_num + 1}...")
        
        dm = CausalDataManager(dag_variables)
        dm.data = chunk
        dm._collect_data_info()
        
        # Process chunk
        dm.normalize_variable_names(mapping_strategy="auto")
        processed_chunk = dm.apply_variable_mapping()
        
        # Validate and analyze
        validation = dm.validate_causal_variables()
        
        results.append({
            'chunk': chunk_num,
            'rows': len(chunk),
            'valid': validation['is_valid'],
            'coverage': validation['coverage_ratio']
        })
    
    return results
```

### 4. Quality Assurance

```python
# Data quality checks
def quality_check(dm):
    """Comprehensive data quality assessment."""
    
    summary = dm.get_data_summary()
    issues = []
    
    # Check missing data
    missing_ratio = summary['missing_data']['total_missing'] / (
        summary['shape'][0] * summary['shape'][1]
    )
    if missing_ratio > 0.1:
        issues.append(f"High missing data: {missing_ratio:.1%}")
    
    # Check variable mapping
    if not summary['variable_mapping']['has_mapping']:
        issues.append("No variable mapping applied")
    
    # Check data types
    if summary['data_types']['numeric'] == 0:
        issues.append("No numeric variables found")
    
    return issues
```

## Error Handling

### Common Issues and Solutions

```python
# Handle file not found
try:
    data = dm.load_data('missing_file.csv')
except FileNotFoundError as e:
    print(f"File not found: {e}")
    # Use sample data or alternative file

# Handle unsupported format
try:
    data = dm.load_data('data.xyz')
except ValueError as e:
    print(f"Unsupported format: {e}")
    # Convert file or use different format

# Handle validation failures
try:
    validation = dm.validate_causal_variables(strict=True)
except ValueError as e:
    print(f"Validation failed: {e}")
    # Implement fallback strategy

# Handle missing columns in filtering
try:
    filtered = dm.get_conditional_data({'nonexistent_col': 'value'})
except ValueError as e:
    print(f"Column error: {e}")
    # Use available columns or fix column names
```

## Performance Tips

1. **Use appropriate data types**: Convert strings to categories for memory efficiency
2. **Filter early**: Apply conditions before expensive operations
3. **Chunk large files**: Process large datasets in manageable chunks  
4. **Cache results**: Store processed data to avoid recomputation
5. **Use Parquet**: Faster loading than CSV for large datasets

```python
# Optimize data types
def optimize_dtypes(df):
    """Optimize DataFrame data types for memory efficiency."""
    
    for col in df.select_dtypes(include=['object']):
        if df[col].nunique() < 0.5 * len(df):
            df[col] = df[col].astype('category')
    
    for col in df.select_dtypes(include=['int64']):
        if df[col].max() < 2**31:
            df[col] = df[col].astype('int32')
    
    return df

# Cache processed data
processed_data = dm.apply_variable_mapping()
processed_data.to_parquet('processed_cache.parquet')
```

## API Reference

### CausalDataManager Class

#### Initialization
```python
CausalDataManager(dag_variables: Optional[Dict[str, str]] = None)
```

#### Core Methods

**Data Loading:**
- `load_data(file_path, file_format=None, **kwargs)` → pd.DataFrame
- `export_processed_data(output_path, file_format=None, **kwargs)` → None

**Variable Processing:**
- `normalize_variable_names(mapping_strategy="auto", custom_mapping=None)` → Dict[str, str]  
- `apply_variable_mapping()` → pd.DataFrame
- `validate_causal_variables(required_variables=None, strict=False)` → Dict[str, Any]

**Data Analysis:**
- `get_conditional_data(conditions, operator="and")` → pd.DataFrame
- `get_variable_distribution(variable, conditional_on=None)` → Dict[str, Any]
- `get_data_summary()` → Dict[str, Any]

#### Utility Functions

```python
create_sample_causal_data(output_path=None) → pd.DataFrame
load_causal_data(file_path, dag_variables=None, normalize_names=True, validate_variables=True, **kwargs) → Tuple[CausalDataManager, pd.DataFrame]
```

## Integration Examples

The CausalDataManager integrates seamlessly with other CausalLLM components and popular data science libraries.

### With Scikit-learn

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Prepare data with CausalDataManager
dm, data = load_causal_data('study.csv', dag_variables)

# Get feature subset
features = dm.get_conditional_data({'include_in_model': True})
X = features[['age', 'income', 'education_years']]
y = features['outcome']

# Standard ML workflow
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

### With Pandas Profiling

```python
import pandas_profiling

# Generate comprehensive data report
dm.load_data('dataset.csv')
processed_data = dm.apply_variable_mapping()

profile = pandas_profiling.ProfileReport(
    processed_data, 
    title="Causal Dataset Profile"
)
profile.to_file("data_report.html")
```

### With Plotly/Matplotlib

```python
import plotly.express as px
import matplotlib.pyplot as plt

# Distribution plots for different subgroups
treated = dm.get_conditional_data({'treatment': True})
untreated = dm.get_conditional_data({'treatment': False})

fig = px.histogram(
    treated, 
    x='outcome', 
    title='Outcome Distribution - Treated Group',
    marginal='box'
)
fig.show()

# Matplotlib comparison
plt.figure(figsize=(10, 6))
plt.hist(treated['outcome'], alpha=0.7, label='Treated', bins=30)
plt.hist(untreated['outcome'], alpha=0.7, label='Untreated', bins=30)
plt.xlabel('Outcome')
plt.ylabel('Frequency')
plt.legend()
plt.title('Treatment Effect on Outcomes')
plt.show()
```

## Summary

The CausalDataManager provides a comprehensive solution for data preparation in causal analysis workflows. Key benefits:

✅ **Unified Interface**: Single class handles all data operations  
✅ **Intelligent Processing**: Auto-mapping and validation reduce manual work  
✅ **Flexible Filtering**: Complex conditional analysis made simple  
✅ **Quality Assurance**: Built-in validation and error handling  
✅ **Performance Optimized**: Efficient pandas operations and memory management  
✅ **Integration Ready**: Works seamlessly with CausalLLMCore and ML libraries

The CausalDataManager bridges the gap between raw data and causal analysis, ensuring your datasets are clean, validated, and ready for rigorous causal inference.

For more examples and advanced usage, see:
- [`examples/data_manager_example.py`](../examples/data_manager_example.py)
- [`tests/test_data_manager.py`](../tests/test_data_manager.py)
- [CausalLLMCore Integration Guide](./Core_Integration.md)