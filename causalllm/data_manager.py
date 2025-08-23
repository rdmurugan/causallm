"""
CausalLLM Data Manager for loading, cleaning, and validating datasets for causal analysis.

This module provides utilities for:
- Loading CSV/Parquet datasets  
- Cleaning and normalizing variable names to match DAG nodes
- Validating causal variables existence and types
- Providing pandas slices for conditional distributions
- Data preprocessing for causal inference workflows
"""

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Set
import pandas as pd
import numpy as np
from causalllm.logging import get_logger, get_structured_logger


class CausalDataManager:
    """
    Comprehensive data manager for causal analysis workflows.
    
    Handles dataset loading, variable validation, name normalization,
    and provides utilities for causal inference with pandas DataFrames.
    """
    
    def __init__(self, dag_variables: Optional[Dict[str, str]] = None):
        """
        Initialize CausalDataManager.
        
        Args:
            dag_variables: Optional dictionary mapping DAG node names to descriptions.
                          Used for variable validation and mapping.
        """
        self.logger = get_logger("causalllm.data_manager")
        self.struct_logger = get_structured_logger("data_manager")
        
        self.logger.info("Initializing CausalDataManager")
        
        self.dag_variables = dag_variables or {}
        self.data: Optional[pd.DataFrame] = None
        self.original_columns: List[str] = []
        self.variable_mapping: Dict[str, str] = {}  # original -> normalized
        self.reverse_mapping: Dict[str, str] = {}   # normalized -> original
        self.data_info: Dict[str, Any] = {}
        
        self.struct_logger.log_interaction(
            "initialization",
            {
                "dag_variables_count": len(self.dag_variables),
                "has_dag_variables": bool(self.dag_variables)
            }
        )
        
        self.logger.info("CausalDataManager initialized successfully")
    
    def load_data(self, 
                  file_path: Union[str, Path], 
                  file_format: Optional[str] = None,
                  **kwargs) -> pd.DataFrame:
        """
        Load dataset from CSV or Parquet file.
        
        Args:
            file_path: Path to the data file
            file_format: Optional format specification ('csv' or 'parquet'). 
                        If None, inferred from file extension.
            **kwargs: Additional arguments passed to pandas read functions
            
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
            RuntimeError: If loading fails
        """
        file_path = Path(file_path)
        self.logger.info(f"Loading data from: {file_path}")
        
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Infer format from extension if not provided
        if file_format is None:
            file_format = file_path.suffix.lower().lstrip('.')
            
        self.logger.debug(f"Detected file format: {file_format}")
        
        try:
            # Load data based on format
            if file_format == 'csv':
                self.data = pd.read_csv(file_path, **kwargs)
                self.logger.info("CSV file loaded successfully")
                
            elif file_format in ['parquet', 'pq']:
                try:
                    self.data = pd.read_parquet(file_path, **kwargs)
                    self.logger.info("Parquet file loaded successfully")
                except ImportError as e:
                    self.logger.error("Parquet support requires pyarrow or fastparquet")
                    raise ImportError("Parquet support requires 'pyarrow' or 'fastparquet'. Install with: pip install pyarrow") from e
                    
            elif file_format in ['xlsx', 'xls']:
                self.data = pd.read_excel(file_path, **kwargs)
                self.logger.info("Excel file loaded successfully")
                
            elif file_format == 'json':
                self.data = pd.read_json(file_path, **kwargs)
                self.logger.info("JSON file loaded successfully")
                
            else:
                supported_formats = ['csv', 'parquet', 'pq', 'xlsx', 'xls', 'json']
                self.logger.error(f"Unsupported file format: {file_format}")
                raise ValueError(f"Unsupported file format '{file_format}'. Supported: {supported_formats}")
            
            # Store original column information
            self.original_columns = self.data.columns.tolist()
            
            # Collect data information
            self._collect_data_info()
            
            self.struct_logger.log_interaction(
                "data_loading",
                {
                    "file_path": str(file_path),
                    "file_format": file_format,
                    "rows": len(self.data),
                    "columns": len(self.data.columns),
                    "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024**2,
                    "column_names": self.original_columns[:10]  # First 10 columns for logging
                }
            )
            
            self.logger.info(f"Data loaded: {len(self.data)} rows, {len(self.data.columns)} columns")
            return self.data
            
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {e}")
            self.struct_logger.log_error(e, {
                "file_path": str(file_path),
                "file_format": file_format,
                "operation": "data_loading"
            })
            raise RuntimeError(f"Failed to load data from {file_path}: {e}") from e
    
    def _collect_data_info(self) -> None:
        """Collect comprehensive information about the loaded dataset."""
        if self.data is None:
            return
            
        self.data_info = {
            "shape": self.data.shape,
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "memory_usage": self.data.memory_usage(deep=True).sum(),
            "numeric_columns": self.data.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": self.data.select_dtypes(include=['object', 'category']).columns.tolist(),
            "datetime_columns": self.data.select_dtypes(include=['datetime']).columns.tolist(),
            "unique_values": {col: self.data[col].nunique() for col in self.data.columns},
            "sample_values": {col: self.data[col].dropna().iloc[:3].tolist() 
                            if len(self.data[col].dropna()) > 0 else [] 
                            for col in self.data.columns}
        }
        
        self.logger.debug(f"Data info collected: {len(self.data_info)} metrics")
    
    def normalize_variable_names(self, 
                                mapping_strategy: str = "auto",
                                custom_mapping: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Normalize variable names to match DAG nodes and create mapping.
        
        Args:
            mapping_strategy: Strategy for normalization:
                            - "auto": Automatic fuzzy matching with DAG variables
                            - "clean": Clean names without DAG matching  
                            - "custom": Use provided custom mapping
            custom_mapping: Custom mapping dictionary {original_name: normalized_name}
            
        Returns:
            Dict[str, str]: Mapping from original to normalized names
            
        Raises:
            ValueError: If data not loaded or invalid strategy
        """
        if self.data is None:
            self.logger.error("No data loaded for variable normalization")
            raise ValueError("No data loaded. Call load_data() first.")
            
        self.logger.info(f"Normalizing variable names using strategy: {mapping_strategy}")
        
        if mapping_strategy == "custom":
            if not custom_mapping:
                raise ValueError("Custom mapping required for 'custom' strategy")
            self.variable_mapping = custom_mapping.copy()
            
        elif mapping_strategy == "clean":
            self.variable_mapping = self._clean_variable_names()
            
        elif mapping_strategy == "auto":
            if self.dag_variables:
                self.variable_mapping = self._auto_map_to_dag_variables()
            else:
                self.logger.warning("No DAG variables provided, falling back to 'clean' strategy")
                self.variable_mapping = self._clean_variable_names()
        else:
            valid_strategies = ["auto", "clean", "custom"]
            raise ValueError(f"Invalid mapping strategy '{mapping_strategy}'. Valid options: {valid_strategies}")
        
        # Create reverse mapping
        self.reverse_mapping = {v: k for k, v in self.variable_mapping.items()}
        
        # Apply mapping to DataFrame if requested
        mapped_columns = []
        for col in self.data.columns:
            if col in self.variable_mapping:
                mapped_columns.append(self.variable_mapping[col])
            else:
                mapped_columns.append(col)
        
        # Check for duplicate mapped names
        if len(set(mapped_columns)) != len(mapped_columns):
            duplicates = [name for name in mapped_columns if mapped_columns.count(name) > 1]
            self.logger.warning(f"Variable mapping resulted in duplicate names: {duplicates}")
        
        self.struct_logger.log_interaction(
            "variable_normalization",
            {
                "strategy": mapping_strategy,
                "original_columns": len(self.original_columns),
                "mapped_columns": len(self.variable_mapping),
                "unmapped_columns": len(self.original_columns) - len(self.variable_mapping),
                "mapping_sample": dict(list(self.variable_mapping.items())[:5])
            }
        )
        
        self.logger.info(f"Variable normalization completed: {len(self.variable_mapping)} mappings created")
        return self.variable_mapping
    
    def _clean_variable_names(self) -> Dict[str, str]:
        """Clean variable names using standard conventions."""
        mapping = {}
        
        for col in self.original_columns:
            # Convert to lowercase
            clean_name = col.lower()
            
            # Replace spaces and special characters with underscores
            clean_name = re.sub(r'[^\w\s]', '_', clean_name)
            clean_name = re.sub(r'\s+', '_', clean_name)
            
            # Remove multiple consecutive underscores
            clean_name = re.sub(r'_+', '_', clean_name)
            
            # Remove leading/trailing underscores
            clean_name = clean_name.strip('_')
            
            # Handle empty names
            if not clean_name:
                clean_name = f"variable_{self.original_columns.index(col)}"
                
            mapping[col] = clean_name
            
        return mapping
    
    def _auto_map_to_dag_variables(self) -> Dict[str, str]:
        """Automatically map data columns to DAG variables using fuzzy matching."""
        from difflib import SequenceMatcher
        
        mapping = {}
        dag_nodes = set(self.dag_variables.keys())
        used_nodes = set()
        
        self.logger.debug(f"Auto-mapping {len(self.original_columns)} columns to {len(dag_nodes)} DAG nodes")
        
        for col in self.original_columns:
            best_match = None
            best_score = 0.0
            
            # Clean column name for comparison
            clean_col = self._clean_single_name(col)
            
            # Try exact match first
            if clean_col in dag_nodes and clean_col not in used_nodes:
                best_match = clean_col
                best_score = 1.0
            else:
                # Fuzzy matching
                for dag_node in dag_nodes:
                    if dag_node in used_nodes:
                        continue
                        
                    # Compare with node name
                    score1 = SequenceMatcher(None, clean_col.lower(), dag_node.lower()).ratio()
                    
                    # Compare with node description if available
                    score2 = 0.0
                    if dag_node in self.dag_variables:
                        desc_clean = self._clean_single_name(self.dag_variables[dag_node])
                        score2 = SequenceMatcher(None, clean_col.lower(), desc_clean.lower()).ratio()
                    
                    # Use the better score
                    score = max(score1, score2)
                    
                    if score > best_score and score > 0.6:  # Minimum threshold
                        best_match = dag_node
                        best_score = score
            
            if best_match:
                mapping[col] = best_match
                used_nodes.add(best_match)
                self.logger.debug(f"Mapped '{col}' -> '{best_match}' (score: {best_score:.3f})")
            else:
                # Fallback to cleaned name
                clean_name = self._clean_single_name(col)
                mapping[col] = clean_name
                self.logger.debug(f"No DAG match for '{col}', using cleaned name: '{clean_name}'")
        
        return mapping
    
    def _clean_single_name(self, name: str) -> str:
        """Clean a single variable name."""
        clean_name = name.lower()
        clean_name = re.sub(r'[^\w\s]', '_', clean_name)
        clean_name = re.sub(r'\s+', '_', clean_name)
        clean_name = re.sub(r'_+', '_', clean_name)
        clean_name = clean_name.strip('_')
        return clean_name if clean_name else "variable"
    
    def validate_causal_variables(self, 
                                required_variables: Optional[List[str]] = None,
                                strict: bool = False) -> Dict[str, Any]:
        """
        Validate that required causal variables exist in the dataset.
        
        Args:
            required_variables: List of required variable names. If None, uses DAG variables.
            strict: If True, raises exception on missing variables
            
        Returns:
            Dict containing validation results
            
        Raises:
            ValueError: If strict=True and validation fails
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Use DAG variables if none specified
        if required_variables is None:
            required_variables = list(self.dag_variables.keys()) if self.dag_variables else []
        
        self.logger.info(f"Validating {len(required_variables)} causal variables")
        
        # Get current column names (may be normalized)
        current_columns = set(self.data.columns)
        mapped_columns = set(self.variable_mapping.values()) if self.variable_mapping else current_columns
        
        # Check for missing variables
        missing_variables = []
        present_variables = []
        
        for var in required_variables:
            if var in current_columns or var in mapped_columns:
                present_variables.append(var)
            else:
                missing_variables.append(var)
        
        # Validation results
        validation_results = {
            "is_valid": len(missing_variables) == 0,
            "required_count": len(required_variables),
            "present_count": len(present_variables),
            "missing_count": len(missing_variables),
            "present_variables": present_variables,
            "missing_variables": missing_variables,
            "coverage_ratio": len(present_variables) / len(required_variables) if required_variables else 1.0,
            "suggestions": self._suggest_missing_variables(missing_variables, current_columns)
        }
        
        self.struct_logger.log_interaction(
            "variable_validation",
            {
                "required_variables": len(required_variables),
                "present_variables": len(present_variables),
                "missing_variables": len(missing_variables),
                "is_valid": validation_results["is_valid"],
                "coverage_ratio": validation_results["coverage_ratio"]
            }
        )
        
        # Log results
        if validation_results["is_valid"]:
            self.logger.info("✅ All required causal variables are present")
        else:
            self.logger.warning(f"❌ Missing {len(missing_variables)} required variables: {missing_variables}")
            
            if validation_results["suggestions"]:
                self.logger.info("💡 Suggested mappings for missing variables:")
                for missing, suggestions in validation_results["suggestions"].items():
                    self.logger.info(f"  {missing} -> {suggestions[:3]}")  # Top 3 suggestions
        
        # Raise exception if strict mode and validation failed
        if strict and not validation_results["is_valid"]:
            raise ValueError(f"Validation failed. Missing required variables: {missing_variables}")
        
        return validation_results
    
    def _suggest_missing_variables(self, 
                                 missing_variables: List[str], 
                                 available_columns: Set[str]) -> Dict[str, List[str]]:
        """Suggest possible mappings for missing variables using fuzzy matching."""
        from difflib import get_close_matches
        
        suggestions = {}
        
        for missing_var in missing_variables:
            # Get close matches
            matches = get_close_matches(
                missing_var.lower(), 
                [col.lower() for col in available_columns], 
                n=5, 
                cutoff=0.4
            )
            
            # Map back to original case
            original_matches = []
            for match in matches:
                for col in available_columns:
                    if col.lower() == match:
                        original_matches.append(col)
                        break
            
            if original_matches:
                suggestions[missing_var] = original_matches
        
        return suggestions
    
    def apply_variable_mapping(self) -> pd.DataFrame:
        """
        Apply variable mapping to the DataFrame column names.
        
        Returns:
            pd.DataFrame: DataFrame with renamed columns
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if not self.variable_mapping:
            self.logger.warning("No variable mapping defined. Use normalize_variable_names() first.")
            return self.data
        
        self.logger.info("Applying variable mapping to DataFrame")
        
        # Create rename mapping for existing columns
        rename_mapping = {}
        for col in self.data.columns:
            if col in self.variable_mapping:
                rename_mapping[col] = self.variable_mapping[col]
        
        # Apply renaming
        self.data = self.data.rename(columns=rename_mapping)
        
        self.logger.info(f"Applied variable mapping: {len(rename_mapping)} columns renamed")
        return self.data
    
    def get_conditional_data(self, 
                           conditions: Dict[str, Union[Any, List[Any], Tuple[Any, Any]]],
                           operator: str = "and") -> pd.DataFrame:
        """
        Get pandas slice based on conditional filtering.
        
        Args:
            conditions: Dictionary of conditions {column: value/values/range}
                       - Single value: exact match
                       - List: isin() match  
                       - Tuple: range (min, max) inclusive
            operator: "and" or "or" for combining conditions
            
        Returns:
            pd.DataFrame: Filtered DataFrame
            
        Raises:
            ValueError: If invalid conditions or operator
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.logger.info(f"Applying conditional filtering with {len(conditions)} conditions")
        self.logger.debug(f"Conditions: {conditions}, Operator: {operator}")
        
        if not conditions:
            return self.data
        
        if operator not in ["and", "or"]:
            raise ValueError("Operator must be 'and' or 'or'")
        
        # Build condition masks
        masks = []
        
        for col, condition in conditions.items():
            if col not in self.data.columns:
                available_cols = list(self.data.columns)
                self.logger.error(f"Column '{col}' not found. Available: {available_cols}")
                raise ValueError(f"Column '{col}' not found in data. Available columns: {available_cols}")
            
            # Handle different condition types
            if isinstance(condition, (list, set)):
                # List of values - use isin()
                mask = self.data[col].isin(condition)
                self.logger.debug(f"Applied isin condition on '{col}': {len(condition)} values")
                
            elif isinstance(condition, tuple) and len(condition) == 2:
                # Range condition - between values
                min_val, max_val = condition
                mask = (self.data[col] >= min_val) & (self.data[col] <= max_val)
                self.logger.debug(f"Applied range condition on '{col}': [{min_val}, {max_val}]")
                
            else:
                # Single value - exact match
                mask = self.data[col] == condition
                self.logger.debug(f"Applied exact match on '{col}': {condition}")
            
            masks.append(mask)
        
        # Combine masks
        if operator == "and":
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask & mask
        else:  # operator == "or"
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask | mask
        
        # Apply filter
        filtered_data = self.data[combined_mask]
        
        self.struct_logger.log_interaction(
            "conditional_filtering",
            {
                "conditions_count": len(conditions),
                "operator": operator,
                "original_rows": len(self.data),
                "filtered_rows": len(filtered_data),
                "reduction_ratio": 1 - (len(filtered_data) / len(self.data)),
                "conditions": conditions
            }
        )
        
        self.logger.info(f"Conditional filtering complete: {len(self.data)} -> {len(filtered_data)} rows")
        return filtered_data
    
    def get_variable_distribution(self, 
                                variable: str,
                                conditional_on: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get distribution information for a variable, optionally conditional on other variables.
        
        Args:
            variable: Variable name to analyze
            conditional_on: Optional conditions for conditional distribution
            
        Returns:
            Dictionary containing distribution statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if variable not in self.data.columns:
            raise ValueError(f"Variable '{variable}' not found in data")
        
        # Get conditional data if specified
        if conditional_on:
            data_subset = self.get_conditional_data(conditional_on)
        else:
            data_subset = self.data
        
        self.logger.info(f"Analyzing distribution for '{variable}' on {len(data_subset)} rows")
        
        var_data = data_subset[variable].dropna()
        
        if len(var_data) == 0:
            return {"error": "No valid data points after filtering"}
        
        # Basic statistics
        distribution_info = {
            "count": len(var_data),
            "missing_count": len(data_subset) - len(var_data),
            "missing_ratio": (len(data_subset) - len(var_data)) / len(data_subset),
            "dtype": str(var_data.dtype),
            "unique_values": var_data.nunique()
        }
        
        # Numeric statistics
        if pd.api.types.is_numeric_dtype(var_data):
            distribution_info.update({
                "mean": float(var_data.mean()),
                "std": float(var_data.std()),
                "min": float(var_data.min()),
                "max": float(var_data.max()),
                "median": float(var_data.median()),
                "quartiles": {
                    "q1": float(var_data.quantile(0.25)),
                    "q3": float(var_data.quantile(0.75))
                },
                "skewness": float(var_data.skew()),
                "kurtosis": float(var_data.kurtosis())
            })
        
        # Categorical statistics  
        if pd.api.types.is_categorical_dtype(var_data) or var_data.dtype == 'object':
            value_counts = var_data.value_counts()
            distribution_info.update({
                "mode": value_counts.index[0] if len(value_counts) > 0 else None,
                "mode_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "value_counts": value_counts.head(10).to_dict(),  # Top 10 values
                "entropy": float(-np.sum((value_counts / len(var_data)) * np.log2(value_counts / len(var_data))))
            })
        
        self.struct_logger.log_interaction(
            "variable_distribution",
            {
                "variable": variable,
                "conditional": bool(conditional_on),
                "data_points": len(var_data),
                "unique_values": distribution_info["unique_values"],
                "has_missing": distribution_info["missing_count"] > 0
            }
        )
        
        return distribution_info
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of the loaded dataset.
        
        Returns:
            Dictionary containing dataset summary information
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        summary = {
            "shape": self.data.shape,
            "columns": {
                "original": self.original_columns,
                "current": list(self.data.columns),
                "mapped": len(self.variable_mapping) > 0
            },
            "data_types": {
                "numeric": len(self.data.select_dtypes(include=[np.number]).columns),
                "categorical": len(self.data.select_dtypes(include=['object', 'category']).columns),
                "datetime": len(self.data.select_dtypes(include=['datetime']).columns),
                "boolean": len(self.data.select_dtypes(include=['bool']).columns)
            },
            "missing_data": {
                "total_missing": int(self.data.isnull().sum().sum()),
                "missing_by_column": self.data.isnull().sum().to_dict(),
                "complete_rows": int((~self.data.isnull().any(axis=1)).sum())
            },
            "memory_usage": {
                "total_mb": self.data.memory_usage(deep=True).sum() / 1024**2,
                "by_column_mb": (self.data.memory_usage(deep=True) / 1024**2).to_dict()
            },
            "variable_mapping": {
                "has_mapping": bool(self.variable_mapping),
                "mapping_count": len(self.variable_mapping),
                "mapping": self.variable_mapping
            }
        }
        
        # Add DAG validation if DAG variables are available
        if self.dag_variables:
            try:
                validation = self.validate_causal_variables()
                summary["dag_validation"] = validation
            except Exception as e:
                summary["dag_validation"] = {"error": str(e)}
        
        return summary
    
    def export_processed_data(self, 
                            output_path: Union[str, Path],
                            file_format: Optional[str] = None,
                            **kwargs) -> None:
        """
        Export processed data to file.
        
        Args:
            output_path: Output file path
            file_format: Output format ('csv', 'parquet', etc.)
            **kwargs: Additional arguments for pandas export functions
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        output_path = Path(output_path)
        
        if file_format is None:
            file_format = output_path.suffix.lower().lstrip('.')
        
        self.logger.info(f"Exporting processed data to: {output_path} (format: {file_format})")
        
        try:
            if file_format == 'csv':
                self.data.to_csv(output_path, index=False, **kwargs)
            elif file_format in ['parquet', 'pq']:
                self.data.to_parquet(output_path, index=False, **kwargs)
            elif file_format in ['xlsx', 'xls']:
                self.data.to_excel(output_path, index=False, **kwargs)
            elif file_format == 'json':
                self.data.to_json(output_path, **kwargs)
            else:
                raise ValueError(f"Unsupported export format: {file_format}")
            
            self.logger.info(f"Data exported successfully to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export data: {e}")
            raise RuntimeError(f"Export failed: {e}") from e


def create_sample_causal_data(output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create a sample dataset for causal analysis demonstrations.
    
    Args:
        output_path: Optional path to save the sample data
        
    Returns:
        pd.DataFrame: Sample causal dataset
    """
    logger = get_logger("causalllm.data_manager.sample")
    logger.info("Creating sample causal dataset")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data with causal relationships
    # Education -> Income -> Healthcare Access -> Health Outcome
    
    # Generate education levels (categorical)
    education_levels = np.random.choice(
        ['High School', 'Bachelor', 'Master', 'PhD'], 
        n_samples, 
        p=[0.4, 0.3, 0.2, 0.1]
    )
    
    # Education affects income (with noise)
    education_income_map = {'High School': 40000, 'Bachelor': 60000, 'Master': 80000, 'PhD': 100000}
    base_income = [education_income_map[ed] for ed in education_levels]
    income = [base + np.random.normal(0, 10000) for base in base_income]
    income = np.maximum(income, 20000)  # Minimum income
    
    # Age as confounding variable
    age = np.random.normal(40, 12, n_samples)
    age = np.clip(age, 22, 65)
    
    # Income and age affect healthcare access
    healthcare_access = (
        0.5 * (np.array(income) - 40000) / 60000 + 
        0.3 * (age - 40) / 25 + 
        np.random.normal(0, 0.2, n_samples)
    )
    healthcare_access = np.clip(healthcare_access, 0, 1)
    
    # Healthcare access affects health outcome
    health_outcome = (
        0.7 * healthcare_access + 
        0.2 * (1 - (age - 22) / 43) +  # Younger people healthier
        np.random.normal(0, 0.15, n_samples)
    )
    health_outcome = np.clip(health_outcome, 0, 1)
    
    # Create treatment variable (binary)
    treatment_prob = 0.3 + 0.4 * healthcare_access
    treatment = np.random.binomial(1, treatment_prob, n_samples)
    
    # Treatment affects health outcome
    health_outcome = health_outcome + 0.15 * treatment + np.random.normal(0, 0.1, n_samples)
    health_outcome = np.clip(health_outcome, 0, 1)
    
    # Create DataFrame with intentionally messy column names
    sample_data = pd.DataFrame({
        'Education Level (Highest)': education_levels,
        'Annual Income ($)': income,
        'Age in Years': age,
        'Healthcare_Access_Score': healthcare_access,
        'Health Outcome Index': health_outcome,
        'Treatment Received': treatment.astype(bool),
        'Geographic Region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'Insurance Type': np.random.choice(['Public', 'Private', 'None'], n_samples, p=[0.3, 0.6, 0.1])
    })
    
    # Add some missing values
    missing_mask = np.random.random(n_samples) < 0.05  # 5% missing
    sample_data.loc[missing_mask, 'Annual Income ($)'] = np.nan
    
    missing_mask = np.random.random(n_samples) < 0.03  # 3% missing
    sample_data.loc[missing_mask, 'Healthcare_Access_Score'] = np.nan
    
    logger.info(f"Sample dataset created: {len(sample_data)} rows, {len(sample_data.columns)} columns")
    
    # Save if output path provided
    if output_path:
        sample_data.to_csv(output_path, index=False)
        logger.info(f"Sample data saved to: {output_path}")
    
    return sample_data


# Convenience functions
def load_causal_data(file_path: Union[str, Path], 
                    dag_variables: Optional[Dict[str, str]] = None,
                    normalize_names: bool = True,
                    validate_variables: bool = True,
                    **load_kwargs) -> Tuple[CausalDataManager, pd.DataFrame]:
    """
    Convenience function to load and prepare causal data in one step.
    
    Args:
        file_path: Path to data file
        dag_variables: Optional DAG variable definitions
        normalize_names: Whether to normalize variable names
        validate_variables: Whether to validate causal variables
        **load_kwargs: Additional arguments for data loading
        
    Returns:
        Tuple of (CausalDataManager instance, loaded DataFrame)
    """
    # Create data manager
    data_manager = CausalDataManager(dag_variables)
    
    # Load data
    data = data_manager.load_data(file_path, **load_kwargs)
    
    # Normalize names if requested
    if normalize_names:
        data_manager.normalize_variable_names()
        data = data_manager.apply_variable_mapping()
    
    # Validate variables if requested
    if validate_variables and dag_variables:
        validation_result = data_manager.validate_causal_variables()
        if not validation_result["is_valid"]:
            logger = get_logger("causalllm.data_manager.convenience")
            logger.warning(f"Variable validation failed: {validation_result['missing_variables']}")
    
    return data_manager, data