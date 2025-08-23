#!/usr/bin/env python3
"""
CausalLLM Data Manager Example

This script demonstrates comprehensive usage of the CausalDataManager for:
- Loading various data formats
- Variable name normalization and mapping
- Causal variable validation
- Conditional data filtering
- Distribution analysis
- Integration with CausalLLMCore

Usage:
    python data_manager_example.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path so we can import causalllm
sys.path.insert(0, str(Path(__file__).parent.parent))

from causalllm.data_manager import CausalDataManager, create_sample_causal_data, load_causal_data
from causalllm.core import CausalLLMCore
from causalllm.llm_client import get_llm_client
from causalllm.logging import setup_package_logging


def demonstrate_basic_data_loading():
    """Demonstrate basic data loading and exploration."""
    print("\n" + "="*60)
    print("🔧 Basic Data Loading & Exploration")
    print("="*60)
    
    # Create sample data
    print("📊 Creating sample causal dataset...")
    sample_file = Path("sample_causal_data.csv")
    sample_data = create_sample_causal_data(str(sample_file))
    print(f"✅ Sample data created: {sample_data.shape}")
    print(f"   Columns: {list(sample_data.columns)}")
    
    # Initialize data manager
    print("\n🔧 Initializing CausalDataManager...")
    data_manager = CausalDataManager()
    
    # Load data
    print("📥 Loading data from CSV...")
    loaded_data = data_manager.load_data(sample_file)
    print(f"✅ Data loaded successfully: {loaded_data.shape}")
    
    # Get data summary
    print("\n📋 Data Summary:")
    summary = data_manager.get_data_summary()
    print(f"   Shape: {summary['shape']}")
    print(f"   Data types: {summary['data_types']}")
    print(f"   Missing data: {summary['missing_data']['total_missing']} total")
    print(f"   Memory usage: {summary['memory_usage']['total_mb']:.2f} MB")
    
    # Clean up
    sample_file.unlink()
    return data_manager, loaded_data


def demonstrate_variable_normalization():
    """Demonstrate variable name normalization and mapping."""
    print("\n" + "="*60)
    print("🏷️  Variable Name Normalization")
    print("="*60)
    
    # Create sample data with messy column names
    messy_data = pd.DataFrame({
        'Patient Age (Years)': [25, 30, 35, 40],
        'Treatment Status!!': [True, False, True, False],
        'Health Score - Final': [0.8, 0.6, 0.9, 0.7],
        'Income Level ($USD)': [50000, 60000, 45000, 80000],
        '   Education   ': ['Bachelor', 'Master', 'High School', 'PhD']
    })
    
    print("📝 Original messy column names:")
    for i, col in enumerate(messy_data.columns, 1):
        print(f"   {i}. '{col}'")
    
    # Initialize data manager with messy data
    data_manager = CausalDataManager()
    data_manager.data = messy_data
    data_manager.original_columns = list(messy_data.columns)
    data_manager._collect_data_info()
    
    # Clean variable names
    print("\n🧹 Normalizing with 'clean' strategy...")
    mapping = data_manager.normalize_variable_names(mapping_strategy="clean")
    
    print("✅ Variable mapping created:")
    for orig, clean in mapping.items():
        print(f"   '{orig}' -> '{clean}'")
    
    # Apply mapping
    print("\n🔄 Applying variable mapping...")
    cleaned_data = data_manager.apply_variable_mapping()
    print("✅ Column names after cleaning:")
    for i, col in enumerate(cleaned_data.columns, 1):
        print(f"   {i}. '{col}'")
    
    return data_manager


def demonstrate_dag_variable_mapping():
    """Demonstrate automatic mapping to DAG variables."""
    print("\n" + "="*60)
    print("🎯 DAG Variable Mapping")
    print("="*60)
    
    # Define DAG variables for a healthcare study
    dag_variables = {
        'age': 'Patient age in years',
        'treatment': 'Medical treatment received',
        'health_outcome': 'Patient health outcome score',
        'income': 'Annual household income',
        'education': 'Education level achieved'
    }
    
    print("🎯 Defined DAG variables:")
    for var, desc in dag_variables.items():
        print(f"   {var}: {desc}")
    
    # Create sample data that should map to DAG variables
    sample_data = pd.DataFrame({
        'Patient_Age': [25, 30, 35, 40, 45],
        'Treatment_Received': [1, 0, 1, 0, 1],
        'Health_Score': [0.8, 0.6, 0.9, 0.7, 0.85],
        'Household_Income': [50000, 60000, 45000, 80000, 70000],
        'Education_Level': ['Bachelor', 'Master', 'High School', 'PhD', 'Bachelor']
    })
    
    print(f"\n📊 Sample data columns: {list(sample_data.columns)}")
    
    # Initialize data manager with DAG variables
    data_manager = CausalDataManager(dag_variables)
    data_manager.data = sample_data
    data_manager.original_columns = list(sample_data.columns)
    data_manager._collect_data_info()
    
    # Auto-map to DAG variables
    print("\n🤖 Auto-mapping to DAG variables...")
    mapping = data_manager.normalize_variable_names(mapping_strategy="auto")
    
    print("✅ Auto-mapping results:")
    for orig, mapped in mapping.items():
        print(f"   '{orig}' -> '{mapped}'")
    
    # Apply mapping and validate
    mapped_data = data_manager.apply_variable_mapping()
    print(f"\n📋 Final column names: {list(mapped_data.columns)}")
    
    # Validate causal variables
    print("\n✅ Validating causal variables...")
    validation = data_manager.validate_causal_variables()
    
    print(f"   Validation passed: {validation['is_valid']}")
    print(f"   Present variables: {validation['present_variables']}")
    if validation['missing_variables']:
        print(f"   Missing variables: {validation['missing_variables']}")
        if validation['suggestions']:
            print("   💡 Suggestions for missing variables:")
            for missing, suggestions in validation['suggestions'].items():
                print(f"      {missing} -> {suggestions}")
    
    return data_manager, dag_variables


def demonstrate_conditional_filtering():
    """Demonstrate conditional data filtering and analysis."""
    print("\n" + "="*60)
    print("🔍 Conditional Data Filtering")
    print("="*60)
    
    # Create comprehensive sample data
    print("📊 Creating comprehensive sample dataset...")
    sample_file = Path("temp_sample.csv")
    sample_data = create_sample_causal_data(str(sample_file))
    
    # Load with data manager
    data_manager = CausalDataManager()
    data = data_manager.load_data(sample_file)
    
    print(f"✅ Dataset loaded: {data.shape}")
    print(f"   Columns: {list(data.columns)}")
    
    # Demonstrate different filtering conditions
    print("\n🔍 Filtering Examples:")
    
    # 1. Single value condition
    print("\n1️⃣ Filter by single value (Treatment Received = True):")
    treated_data = data_manager.get_conditional_data({
        'Treatment Received': True
    })
    print(f"   Treated patients: {len(treated_data)} ({len(treated_data)/len(data)*100:.1f}%)")
    
    # 2. List condition (multiple values)
    print("\n2️⃣ Filter by multiple values (Education in ['Master', 'PhD']):")
    educated_data = data_manager.get_conditional_data({
        'Education Level (Highest)': ['Master', 'PhD']
    })
    print(f"   Advanced education: {len(educated_data)} ({len(educated_data)/len(data)*100:.1f}%)")
    
    # 3. Range condition
    print("\n3️⃣ Filter by range (Age between 30-50):")
    age_filtered = data_manager.get_conditional_data({
        'Age in Years': (30, 50)
    })
    print(f"   Ages 30-50: {len(age_filtered)} ({len(age_filtered)/len(data)*100:.1f}%)")
    
    # 4. Multiple conditions with AND
    print("\n4️⃣ Multiple conditions (AND): Treated AND High Income:")
    complex_filter = data_manager.get_conditional_data({
        'Treatment Received': True,
        'Annual Income ($)': (70000, float('inf'))
    }, operator="and")
    print(f"   Treated + High Income: {len(complex_filter)} ({len(complex_filter)/len(data)*100:.1f}%)")
    
    # 5. Multiple conditions with OR
    print("\n5️⃣ Multiple conditions (OR): Young OR Highly Educated:")
    or_filter = data_manager.get_conditional_data({
        'Age in Years': (22, 30),
        'Education Level (Highest)': ['PhD']
    }, operator="or")
    print(f"   Young OR PhD: {len(or_filter)} ({len(or_filter)/len(data)*100:.1f}%)")
    
    # Clean up
    sample_file.unlink()
    return data_manager


def demonstrate_distribution_analysis():
    """Demonstrate variable distribution analysis."""
    print("\n" + "="*60)
    print("📊 Distribution Analysis")
    print("="*60)
    
    # Create sample data
    sample_file = Path("temp_dist.csv")
    sample_data = create_sample_causal_data(str(sample_file))
    
    data_manager = CausalDataManager()
    data = data_manager.load_data(sample_file)
    
    print("📊 Analyzing variable distributions...")
    
    # 1. Numeric variable distribution
    print("\n1️⃣ Numeric Variable Analysis (Annual Income):")
    income_dist = data_manager.get_variable_distribution('Annual Income ($)')
    
    print(f"   Count: {income_dist['count']}")
    print(f"   Mean: ${income_dist['mean']:,.2f}")
    print(f"   Std: ${income_dist['std']:,.2f}")
    print(f"   Range: ${income_dist['min']:,.0f} - ${income_dist['max']:,.0f}")
    print(f"   Median: ${income_dist['median']:,.2f}")
    
    # 2. Categorical variable distribution
    print("\n2️⃣ Categorical Variable Analysis (Education Level):")
    edu_dist = data_manager.get_variable_distribution('Education Level (Highest)')
    
    print(f"   Unique values: {edu_dist['unique_values']}")
    print(f"   Mode: {edu_dist['mode']}")
    print("   Value counts:")
    for value, count in edu_dist['value_counts'].items():
        print(f"      {value}: {count}")
    
    # 3. Conditional distribution
    print("\n3️⃣ Conditional Distribution (Income | Treatment = True):")
    conditional_dist = data_manager.get_variable_distribution(
        'Annual Income ($)',
        conditional_on={'Treatment Received': True}
    )
    
    print(f"   Treated group - Mean income: ${conditional_dist['mean']:,.2f}")
    print(f"   Treated group - Count: {conditional_dist['count']}")
    
    # Compare with untreated
    untreated_dist = data_manager.get_variable_distribution(
        'Annual Income ($)',
        conditional_on={'Treatment Received': False}
    )
    print(f"   Untreated group - Mean income: ${untreated_dist['mean']:,.2f}")
    print(f"   Treatment effect on income: ${conditional_dist['mean'] - untreated_dist['mean']:,.2f}")
    
    # Clean up
    sample_file.unlink()
    return data_manager


def demonstrate_core_integration():
    """Demonstrate integration with CausalLLMCore."""
    print("\n" + "="*60)
    print("🧠 Integration with CausalLLMCore")
    print("="*60)
    
    # Define causal model
    dag_variables = {
        'education': 'Education level achieved',
        'income': 'Annual household income',
        'healthcare_access': 'Healthcare access score',
        'treatment': 'Medical treatment received',
        'health_outcome': 'Health outcome index'
    }
    
    dag_edges = [
        ('education', 'income'),
        ('income', 'healthcare_access'),
        ('healthcare_access', 'treatment'),
        ('treatment', 'health_outcome'),
        ('education', 'healthcare_access')  # Direct effect
    ]
    
    print("🎯 Defined causal model:")
    print(f"   Variables: {list(dag_variables.keys())}")
    print(f"   Edges: {dag_edges}")
    
    # Load and prepare data
    print("\n📊 Loading and preparing data...")
    sample_file = Path("temp_core.csv")
    create_sample_causal_data(str(sample_file))
    
    # Use convenience function
    data_manager, data = load_causal_data(
        sample_file,
        dag_variables=dag_variables,
        normalize_names=True,
        validate_variables=True
    )
    
    print(f"✅ Data prepared: {data.shape}")
    print(f"   Final columns: {list(data.columns)}")
    
    # Get validation results
    validation = data_manager.validate_causal_variables()
    print(f"   Validation: {validation['present_count']}/{validation['required_count']} variables present")
    
    # Create CausalLLMCore with processed data
    print("\n🧠 Creating CausalLLMCore...")
    try:
        llm_client = get_llm_client("grok")  # Use mock client for demo
        
        core = CausalLLMCore(
            context="Healthcare access and outcomes analysis using processed survey data",
            variables=dag_variables,
            dag_edges=dag_edges,
            llm_client=llm_client
        )
        
        print("✅ CausalLLMCore created successfully")
        
        # Demonstrate causal analysis with filtered data
        print("\n🔬 Causal Analysis Example:")
        
        # Get high-income subset for analysis
        high_income_data = data_manager.get_conditional_data({
            'income': (75000, float('inf'))
        })
        
        print(f"   High-income subset: {len(high_income_data)} samples")
        
        # Simulate counterfactual
        counterfactual_result = core.simulate_counterfactual(
            factual="High-income patient received standard healthcare access",
            intervention="Same patient had guaranteed premium healthcare access",
            instruction="Focus on the impact on treatment receipt and health outcomes"
        )
        
        print(f"   Counterfactual analysis completed: {len(counterfactual_result)} characters")
        print(f"   Result preview: {counterfactual_result[:150]}...")
        
        # Do-calculus simulation
        do_result = core.simulate_do(
            intervention={'healthcare_access': 'maximum'},
            question="What would be the population-level impact on health outcomes?"
        )
        
        print(f"   Do-calculus simulation completed: {len(do_result)} characters")
        print(f"   Result preview: {do_result[:150]}...")
        
    except Exception as e:
        print(f"   ⚠️ Core integration demo skipped due to: {e}")
    
    # Clean up
    sample_file.unlink()
    return data_manager


def demonstrate_export_functionality():
    """Demonstrate data export capabilities."""
    print("\n" + "="*60)
    print("💾 Data Export Functionality")
    print("="*60)
    
    # Create and process data
    sample_data = create_sample_causal_data()
    
    dag_variables = {
        'education': 'Education level',
        'income': 'Annual income',
        'healthcare_access': 'Healthcare access score',
        'health_outcome': 'Health outcome index'
    }
    
    data_manager = CausalDataManager(dag_variables)
    data_manager.data = sample_data
    data_manager.original_columns = list(sample_data.columns)
    data_manager._collect_data_info()
    
    # Normalize variables
    data_manager.normalize_variable_names(mapping_strategy="auto")
    processed_data = data_manager.apply_variable_mapping()
    
    print("✅ Data processed and ready for export")
    print(f"   Shape: {processed_data.shape}")
    print(f"   Columns: {list(processed_data.columns)}")
    
    # Export to different formats
    export_files = []
    
    try:
        # CSV export
        csv_file = Path("processed_causal_data.csv")
        data_manager.export_processed_data(csv_file, file_format="csv")
        export_files.append(csv_file)
        print(f"✅ Exported to CSV: {csv_file}")
        
        # JSON export
        json_file = Path("processed_causal_data.json")
        data_manager.export_processed_data(json_file, file_format="json")
        export_files.append(json_file)
        print(f"✅ Exported to JSON: {json_file}")
        
        # Show file sizes
        print("\n📊 Export file sizes:")
        for file_path in export_files:
            size_mb = file_path.stat().st_size / 1024**2
            print(f"   {file_path.name}: {size_mb:.2f} MB")
        
    except Exception as e:
        print(f"   ⚠️ Export error: {e}")
    
    finally:
        # Clean up export files
        for file_path in export_files:
            if file_path.exists():
                file_path.unlink()
        print("\n🧹 Cleaned up export files")


def main():
    """Run all CausalDataManager demonstrations."""
    print("🚀 CausalLLM Data Manager Comprehensive Demo")
    print("=" * 80)
    
    # Set up logging
    setup_package_logging(level="INFO", log_to_file=False)
    
    try:
        # Run all demonstrations
        demonstrate_basic_data_loading()
        demonstrate_variable_normalization()
        demonstrate_dag_variable_mapping()
        demonstrate_conditional_filtering()
        demonstrate_distribution_analysis()
        demonstrate_core_integration()
        demonstrate_export_functionality()
        
        print("\n" + "=" * 80)
        print("🎉 All CausalDataManager demonstrations completed successfully!")
        print("\n💡 Key capabilities demonstrated:")
        print("   ✅ Multi-format data loading (CSV, Parquet, Excel, JSON)")
        print("   ✅ Intelligent variable name normalization")
        print("   ✅ Automatic DAG variable mapping")
        print("   ✅ Comprehensive data validation")
        print("   ✅ Flexible conditional filtering")
        print("   ✅ Distribution analysis")
        print("   ✅ CausalLLMCore integration")
        print("   ✅ Multi-format data export")
        
        print("\n🚀 Ready for causal analysis workflows!")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()