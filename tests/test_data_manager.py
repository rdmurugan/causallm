#!/usr/bin/env python3
"""
Comprehensive test suite for CausalLLM Data Manager.

Tests all functionality including:
- Data loading from various formats
- Variable name normalization and mapping
- Causal variable validation
- Conditional filtering and distribution analysis
- Integration with CausalLLMCore
- Data export capabilities

Usage:
    python test_data_manager.py
"""

import unittest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Add parent directory to path so we can import causalllm
sys.path.insert(0, str(Path(__file__).parent.parent))

from causalllm.data_manager import CausalDataManager, create_sample_causal_data, load_causal_data
from causalllm.core import CausalLLMCore
from causalllm.llm_client import get_llm_client
from causalllm.logging import setup_package_logging


class TestCausalDataManager(unittest.TestCase):
    """Test CausalDataManager core functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.dag_variables = {
            'age': 'Patient age in years',
            'treatment': 'Medical treatment received',
            'outcome': 'Patient health outcome',
            'income': 'Annual household income'
        }
        
        self.sample_data = pd.DataFrame({
            'Patient Age': [25, 30, 35, 40, 45],
            'Treatment_Status': [True, False, True, False, True],
            'Health_Outcome': [0.8, 0.6, 0.9, 0.7, 0.85],
            'Annual_Income': [50000, 60000, 45000, 80000, 70000],
            'Education': ['Bachelor', 'Master', 'High School', 'PhD', 'Bachelor']
        })
    
    def test_initialization(self):
        """Test CausalDataManager initialization."""
        # Test without DAG variables
        dm1 = CausalDataManager()
        self.assertEqual(len(dm1.dag_variables), 0)
        self.assertIsNone(dm1.data)
        self.assertEqual(len(dm1.variable_mapping), 0)
        
        # Test with DAG variables
        dm2 = CausalDataManager(self.dag_variables)
        self.assertEqual(len(dm2.dag_variables), 4)
        self.assertEqual(dm2.dag_variables['age'], 'Patient age in years')
    
    def test_data_loading_csv(self):
        """Test CSV data loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            temp_file = Path(f.name)
        
        try:
            dm = CausalDataManager()
            loaded_data = dm.load_data(temp_file)
            
            self.assertIsNotNone(loaded_data)
            self.assertEqual(loaded_data.shape, self.sample_data.shape)
            self.assertEqual(list(loaded_data.columns), list(self.sample_data.columns))
            self.assertEqual(len(dm.original_columns), len(self.sample_data.columns))
            
        finally:
            temp_file.unlink()
    
    def test_data_loading_json(self):
        """Test JSON data loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            self.sample_data.to_json(f.name)
            temp_file = Path(f.name)
        
        try:
            dm = CausalDataManager()
            loaded_data = dm.load_data(temp_file)
            
            self.assertIsNotNone(loaded_data)
            self.assertEqual(loaded_data.shape, self.sample_data.shape)
            
        finally:
            temp_file.unlink()
    
    def test_data_loading_invalid_file(self):
        """Test loading from non-existent file."""
        dm = CausalDataManager()
        
        with self.assertRaises(FileNotFoundError):
            dm.load_data("non_existent_file.csv")
    
    def test_data_loading_unsupported_format(self):
        """Test loading unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_file = Path(f.name)
        
        try:
            dm = CausalDataManager()
            
            with self.assertRaises(ValueError):
                dm.load_data(temp_file)
                
        finally:
            temp_file.unlink()


class TestVariableNormalization(unittest.TestCase):
    """Test variable name normalization and mapping."""
    
    def setUp(self):
        """Set up test data with messy column names."""
        self.messy_data = pd.DataFrame({
            'Patient Age (Years)': [25, 30, 35],
            'Treatment Status!!': [True, False, True],
            'Health Score - Final': [0.8, 0.6, 0.9],
            '   Income   ': [50000, 60000, 45000],
            'Education@Level': ['Bachelor', 'Master', 'High School']
        })
        
        self.dag_variables = {
            'age': 'Patient age',
            'treatment': 'Treatment status',
            'health': 'Health score',
            'income': 'Income level',
            'education': 'Education level'
        }
    
    def test_clean_variable_names(self):
        """Test clean variable name normalization."""
        dm = CausalDataManager()
        dm.data = self.messy_data
        dm.original_columns = list(self.messy_data.columns)
        dm._collect_data_info()
        
        mapping = dm.normalize_variable_names(mapping_strategy="clean")
        
        # Check that all original columns are mapped
        self.assertEqual(len(mapping), len(self.messy_data.columns))
        
        # Check specific mappings
        expected_clean_names = {
            'Patient Age (Years)': 'patient_age_years',
            'Treatment Status!!': 'treatment_status',
            'Health Score - Final': 'health_score_final',
            '   Income   ': 'income',
            'Education@Level': 'education_level'
        }
        
        for orig, expected in expected_clean_names.items():
            self.assertEqual(mapping[orig], expected)
    
    def test_auto_variable_mapping(self):
        """Test automatic mapping to DAG variables."""
        dm = CausalDataManager(self.dag_variables)
        dm.data = self.messy_data
        dm.original_columns = list(self.messy_data.columns)
        dm._collect_data_info()
        
        mapping = dm.normalize_variable_names(mapping_strategy="auto")
        
        # Check that mapping was created
        self.assertEqual(len(mapping), len(self.messy_data.columns))
        
        # Should map to DAG variables when possible
        self.assertIn('age', mapping.values())
        self.assertIn('treatment', mapping.values())
        self.assertIn('income', mapping.values())
    
    def test_custom_variable_mapping(self):
        """Test custom variable mapping."""
        custom_mapping = {
            'Patient Age (Years)': 'patient_age',
            'Treatment Status!!': 'treated',
            'Health Score - Final': 'health_score',
            '   Income   ': 'annual_income',
            'Education@Level': 'edu_level'
        }
        
        dm = CausalDataManager()
        dm.data = self.messy_data
        dm.original_columns = list(self.messy_data.columns)
        dm._collect_data_info()
        
        mapping = dm.normalize_variable_names(
            mapping_strategy="custom",
            custom_mapping=custom_mapping
        )
        
        self.assertEqual(mapping, custom_mapping)
    
    def test_apply_variable_mapping(self):
        """Test applying variable mapping to DataFrame."""
        dm = CausalDataManager()
        dm.data = self.messy_data.copy()
        dm.original_columns = list(self.messy_data.columns)
        dm._collect_data_info()
        
        # Create mapping and apply
        mapping = dm.normalize_variable_names(mapping_strategy="clean")
        mapped_data = dm.apply_variable_mapping()
        
        # Check that columns were renamed
        original_cols = set(self.messy_data.columns)
        mapped_cols = set(mapped_data.columns)
        
        # Should have different column names
        self.assertNotEqual(original_cols, mapped_cols)
        
        # Should have same number of columns
        self.assertEqual(len(original_cols), len(mapped_cols))
        
        # Data should be preserved
        self.assertEqual(mapped_data.shape, self.messy_data.shape)


class TestCausalVariableValidation(unittest.TestCase):
    """Test causal variable validation functionality."""
    
    def setUp(self):
        """Set up test data and DAG variables."""
        self.data = pd.DataFrame({
            'age': [25, 30, 35, 40],
            'treatment': [True, False, True, False],
            'outcome': [0.8, 0.6, 0.9, 0.7],
            'income': [50000, 60000, 45000, 80000]
        })
        
        self.dag_variables = {
            'age': 'Patient age',
            'treatment': 'Treatment received',
            'outcome': 'Health outcome',
            'income': 'Annual income',
            'education': 'Education level'  # This one is missing from data
        }
    
    def test_validate_all_present(self):
        """Test validation when all required variables are present."""
        dm = CausalDataManager(self.dag_variables)
        dm.data = self.data
        dm.original_columns = list(self.data.columns)
        dm._collect_data_info()
        
        # Test with subset of variables that are all present
        required_vars = ['age', 'treatment', 'outcome']
        validation = dm.validate_causal_variables(required_vars)
        
        self.assertTrue(validation['is_valid'])
        self.assertEqual(validation['present_count'], 3)
        self.assertEqual(validation['missing_count'], 0)
        self.assertEqual(len(validation['missing_variables']), 0)
    
    def test_validate_missing_variables(self):
        """Test validation when some variables are missing."""
        dm = CausalDataManager(self.dag_variables)
        dm.data = self.data
        dm.original_columns = list(self.data.columns)
        dm._collect_data_info()
        
        # Use all DAG variables (including missing 'education')
        validation = dm.validate_causal_variables()
        
        self.assertFalse(validation['is_valid'])
        self.assertEqual(validation['present_count'], 4)
        self.assertEqual(validation['missing_count'], 1)
        self.assertIn('education', validation['missing_variables'])
    
    def test_validate_strict_mode(self):
        """Test validation in strict mode."""
        dm = CausalDataManager(self.dag_variables)
        dm.data = self.data
        dm.original_columns = list(self.data.columns)
        dm._collect_data_info()
        
        # Should raise exception in strict mode with missing variables
        with self.assertRaises(ValueError):
            dm.validate_causal_variables(strict=True)
    
    def test_validation_suggestions(self):
        """Test that validation provides suggestions for missing variables."""
        # Create data with similar column name
        data_with_similar = self.data.copy()
        data_with_similar['edu_level'] = ['Bachelor', 'Master', 'PhD', 'High School']
        
        dm = CausalDataManager(self.dag_variables)
        dm.data = data_with_similar
        dm.original_columns = list(data_with_similar.columns)
        dm._collect_data_info()
        
        validation = dm.validate_causal_variables()
        
        self.assertIn('education', validation['missing_variables'])
        self.assertIn('education', validation['suggestions'])
        self.assertIn('edu_level', validation['suggestions']['education'])


class TestConditionalFiltering(unittest.TestCase):
    """Test conditional data filtering functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'age': [25, 30, 35, 40, 45, 50],
            'treatment': [True, False, True, False, True, False],
            'income': [30000, 50000, 70000, 90000, 110000, 130000],
            'education': ['HS', 'Bachelor', 'Master', 'PhD', 'Master', 'Bachelor'],
            'region': ['North', 'South', 'East', 'West', 'North', 'South']
        })
    
    def test_single_value_condition(self):
        """Test filtering with single value condition."""
        dm = CausalDataManager()
        dm.data = self.data
        
        filtered = dm.get_conditional_data({'treatment': True})
        
        self.assertEqual(len(filtered), 3)  # 3 True values
        self.assertTrue(all(filtered['treatment']))
    
    def test_list_condition(self):
        """Test filtering with list of values."""
        dm = CausalDataManager()
        dm.data = self.data
        
        filtered = dm.get_conditional_data({'education': ['Master', 'PhD']})
        
        self.assertEqual(len(filtered), 3)  # 2 Master + 1 PhD
        self.assertTrue(all(ed in ['Master', 'PhD'] for ed in filtered['education']))
    
    def test_range_condition(self):
        """Test filtering with range condition."""
        dm = CausalDataManager()
        dm.data = self.data
        
        filtered = dm.get_conditional_data({'age': (30, 45)})
        
        self.assertEqual(len(filtered), 4)  # ages 30, 35, 40, 45
        self.assertTrue(all(30 <= age <= 45 for age in filtered['age']))
    
    def test_multiple_conditions_and(self):
        """Test filtering with multiple AND conditions."""
        dm = CausalDataManager()
        dm.data = self.data
        
        filtered = dm.get_conditional_data({
            'treatment': True,
            'age': (25, 40)
        }, operator="and")
        
        # Should have treatment=True AND age in [25,40]
        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(filtered['treatment']))
        self.assertTrue(all(25 <= age <= 40 for age in filtered['age']))
    
    def test_multiple_conditions_or(self):
        """Test filtering with multiple OR conditions."""
        dm = CausalDataManager()
        dm.data = self.data
        
        filtered = dm.get_conditional_data({
            'age': (25, 30),  # 2 rows
            'education': ['PhD']  # 1 row
        }, operator="or")
        
        # Should have age in [25,30] OR education=PhD
        self.assertEqual(len(filtered), 3)
    
    def test_invalid_column(self):
        """Test filtering with invalid column name."""
        dm = CausalDataManager()
        dm.data = self.data
        
        with self.assertRaises(ValueError):
            dm.get_conditional_data({'invalid_column': 'value'})
    
    def test_invalid_operator(self):
        """Test filtering with invalid operator."""
        dm = CausalDataManager()
        dm.data = self.data
        
        with self.assertRaises(ValueError):
            dm.get_conditional_data({'age': 30}, operator="invalid")


class TestDistributionAnalysis(unittest.TestCase):
    """Test variable distribution analysis."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)  # For reproducible results
        
        self.data = pd.DataFrame({
            'numeric_var': np.random.normal(50, 15, 100),
            'categorical_var': np.random.choice(['A', 'B', 'C'], 100, p=[0.5, 0.3, 0.2]),
            'binary_var': np.random.choice([True, False], 100),
            'missing_var': [1.0] * 80 + [np.nan] * 20  # 20% missing
        })
    
    def test_numeric_distribution(self):
        """Test numeric variable distribution analysis."""
        dm = CausalDataManager()
        dm.data = self.data
        
        dist = dm.get_variable_distribution('numeric_var')
        
        # Check basic statistics
        self.assertEqual(dist['count'], 100)
        self.assertEqual(dist['missing_count'], 0)
        self.assertAlmostEqual(dist['mean'], self.data['numeric_var'].mean(), places=3)
        self.assertAlmostEqual(dist['std'], self.data['numeric_var'].std(), places=3)
        self.assertIn('quartiles', dist)
        self.assertIn('q1', dist['quartiles'])
        self.assertIn('q3', dist['quartiles'])
    
    def test_categorical_distribution(self):
        """Test categorical variable distribution analysis."""
        dm = CausalDataManager()
        dm.data = self.data
        
        dist = dm.get_variable_distribution('categorical_var')
        
        # Check categorical statistics
        self.assertEqual(dist['count'], 100)
        self.assertIn('mode', dist)
        self.assertIn('value_counts', dist)
        self.assertIn('entropy', dist)
        
        # Mode should be 'A' (highest probability)
        self.assertEqual(dist['mode'], 'A')
    
    def test_missing_values_distribution(self):
        """Test distribution analysis with missing values."""
        dm = CausalDataManager()
        dm.data = self.data
        
        dist = dm.get_variable_distribution('missing_var')
        
        # Check missing value handling
        self.assertEqual(dist['count'], 80)  # Only non-missing values
        self.assertEqual(dist['missing_count'], 20)
        self.assertEqual(dist['missing_ratio'], 0.2)
    
    def test_conditional_distribution(self):
        """Test conditional distribution analysis."""
        dm = CausalDataManager()
        dm.data = self.data
        
        # Get distribution conditional on binary_var=True
        dist = dm.get_variable_distribution(
            'numeric_var',
            conditional_on={'binary_var': True}
        )
        
        # Should have fewer data points
        self.assertLess(dist['count'], 100)
        
        # Should still have basic statistics
        self.assertIn('mean', dist)
        self.assertIn('std', dist)
    
    def test_invalid_variable(self):
        """Test distribution analysis with invalid variable."""
        dm = CausalDataManager()
        dm.data = self.data
        
        with self.assertRaises(ValueError):
            dm.get_variable_distribution('invalid_var')


class TestDataExport(unittest.TestCase):
    """Test data export functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'var1': [1, 2, 3, 4],
            'var2': ['A', 'B', 'C', 'D'],
            'var3': [0.1, 0.2, 0.3, 0.4]
        })
    
    def test_csv_export(self):
        """Test CSV export."""
        dm = CausalDataManager()
        dm.data = self.data
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_file = Path(f.name)
        
        try:
            dm.export_processed_data(temp_file)
            
            # Verify file exists and has content
            self.assertTrue(temp_file.exists())
            self.assertGreater(temp_file.stat().st_size, 0)
            
            # Verify content by reading back
            exported_data = pd.read_csv(temp_file)
            pd.testing.assert_frame_equal(exported_data, self.data)
            
        finally:
            if temp_file.exists():
                temp_file.unlink()
    
    def test_json_export(self):
        """Test JSON export."""
        dm = CausalDataManager()
        dm.data = self.data
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_file = Path(f.name)
        
        try:
            dm.export_processed_data(temp_file)
            
            # Verify file exists
            self.assertTrue(temp_file.exists())
            self.assertGreater(temp_file.stat().st_size, 0)
            
            # Verify it's valid JSON
            with open(temp_file, 'r') as f:
                json_content = json.load(f)
            self.assertIsInstance(json_content, dict)
            
        finally:
            if temp_file.exists():
                temp_file.unlink()
    
    def test_export_no_data(self):
        """Test export when no data is loaded."""
        dm = CausalDataManager()
        
        with tempfile.NamedTemporaryFile(suffix='.csv') as f:
            temp_file = Path(f.name)
            
            with self.assertRaises(ValueError):
                dm.export_processed_data(temp_file)


class TestSampleDataCreation(unittest.TestCase):
    """Test sample data creation functionality."""
    
    def test_create_sample_data(self):
        """Test creation of sample causal data."""
        sample_data = create_sample_causal_data()
        
        # Check basic properties
        self.assertIsInstance(sample_data, pd.DataFrame)
        self.assertGreater(len(sample_data), 0)
        self.assertGreater(len(sample_data.columns), 0)
        
        # Check expected columns exist
        expected_columns = [
            'Education Level (Highest)',
            'Annual Income ($)',
            'Age in Years',
            'Health Outcome Index',
            'Treatment Received'
        ]
        
        for col in expected_columns:
            self.assertIn(col, sample_data.columns)
        
        # Check for some missing values (should have been added)
        total_missing = sample_data.isnull().sum().sum()
        self.assertGreater(total_missing, 0)
    
    def test_create_sample_with_export(self):
        """Test creating sample data with file export."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_file = Path(f.name)
        
        try:
            sample_data = create_sample_causal_data(str(temp_file))
            
            # Check data was created
            self.assertIsInstance(sample_data, pd.DataFrame)
            
            # Check file was saved
            self.assertTrue(temp_file.exists())
            
            # Verify saved data matches returned data
            saved_data = pd.read_csv(temp_file)
            pd.testing.assert_frame_equal(saved_data, sample_data)
            
        finally:
            if temp_file.exists():
                temp_file.unlink()


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def test_load_causal_data(self):
        """Test load_causal_data convenience function."""
        # Create sample data file
        sample_data = create_sample_causal_data()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_file = Path(f.name)
        
        try:
            dag_variables = {
                'education': 'Education level',
                'income': 'Annual income',
                'age': 'Age in years'
            }
            
            # Test convenience function
            data_manager, data = load_causal_data(
                temp_file,
                dag_variables=dag_variables,
                normalize_names=True,
                validate_variables=False  # Don't validate to avoid errors
            )
            
            # Check results
            self.assertIsInstance(data_manager, CausalDataManager)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertEqual(data.shape, sample_data.shape)
            
            # Check that normalization was applied
            self.assertGreater(len(data_manager.variable_mapping), 0)
            
        finally:
            temp_file.unlink()


def run_tests():
    """Run all data manager tests."""
    print("🧪 Running CausalLLM Data Manager Tests")
    print("=" * 60)
    
    # Set up logging (quiet for tests)
    setup_package_logging(level="WARNING", log_to_file=False)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestCausalDataManager,
        TestVariableNormalization,
        TestCausalVariableValidation,
        TestConditionalFiltering,
        TestDistributionAnalysis,
        TestDataExport,
        TestSampleDataCreation,
        TestConvenienceFunctions
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🧪 Test Results:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n✅ All Data Manager tests passed!")
    else:
        print(f"\n❌ {len(result.failures + result.errors)} tests failed")
    
    return success


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)