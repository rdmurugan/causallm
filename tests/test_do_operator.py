"""
Test suite for do-operator functionality
Tests do-calculus simulation and intervention analysis
"""
import pytest
from unittest.mock import patch, Mock
from causallm.core.do_operator import DoOperatorSimulator


class TestDoOperatorSimulator:
    """Test DoOperatorSimulator functionality."""
    
    def test_initialization_basic(self, causal_context, variable_descriptions):
        """Test basic initialization of DoOperatorSimulator."""
        simulator = DoOperatorSimulator(causal_context, variable_descriptions)
        
        assert simulator.base_context == causal_context
        assert simulator.variables == variable_descriptions
        assert len(simulator.variables) == len(variable_descriptions)
    
    def test_initialization_empty_context(self, variable_descriptions):
        """Test initialization with empty context."""
        simulator = DoOperatorSimulator("", variable_descriptions)
        
        assert simulator.base_context == ""
        assert simulator.variables == variable_descriptions
    
    def test_initialization_empty_variables(self, causal_context):
        """Test initialization with empty variables."""
        simulator = DoOperatorSimulator(causal_context, {})
        
        assert simulator.base_context == causal_context
        assert simulator.variables == {}
    
    def test_intervene_basic(self, causal_context, variable_descriptions):
        """Test basic intervention functionality."""
        simulator = DoOperatorSimulator(causal_context, variable_descriptions)
        
        interventions = {"X1": "high_dose_treatment"}
        result = simulator.intervene(interventions)
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain the new intervention value
        assert "high_dose_treatment" in result
        # Variables should be updated
        assert simulator.variables["X1"] == "high_dose_treatment"
    
    def test_intervene_multiple_variables(self, causal_context, variable_descriptions):
        """Test intervention on multiple variables."""
        simulator = DoOperatorSimulator(causal_context, variable_descriptions)
        
        interventions = {
            "X1": "experimental_treatment",
            "X2": "enhanced_response"
        }
        result = simulator.intervene(interventions)
        
        assert isinstance(result, str)
        assert "experimental_treatment" in result
        assert "enhanced_response" in result
        
        # Both variables should be updated
        assert simulator.variables["X1"] == "experimental_treatment"
        assert simulator.variables["X2"] == "enhanced_response"
    
    def test_intervene_invalid_variable(self, causal_context, variable_descriptions):
        """Test intervention with invalid variable."""
        simulator = DoOperatorSimulator(causal_context, variable_descriptions)
        
        interventions = {"nonexistent_variable": "some_value"}
        
        with pytest.raises(ValueError, match="not in base context"):
            simulator.intervene(interventions)
    
    def test_intervene_preserve_original_variables(self, causal_context, variable_descriptions):
        """Test that intervention preserves non-intervened variables."""
        simulator = DoOperatorSimulator(causal_context, variable_descriptions)
        original_x2 = simulator.variables["X2"]
        original_x3 = simulator.variables["X3"]
        
        interventions = {"X1": "new_treatment"}
        simulator.intervene(interventions)
        
        # Non-intervened variables should remain unchanged
        assert simulator.variables["X2"] == original_x2
        assert simulator.variables["X3"] == original_x3
        # Intervened variable should be changed
        assert simulator.variables["X1"] == "new_treatment"
    
    def test_intervene_empty_interventions(self, causal_context, variable_descriptions):
        """Test intervention with empty interventions dict."""
        simulator = DoOperatorSimulator(causal_context, variable_descriptions)
        original_context = simulator.base_context
        
        result = simulator.intervene({})
        
        # Should return original context unchanged
        assert result == original_context
        # Variables should remain unchanged
        assert simulator.variables == variable_descriptions
    
    def test_generate_do_prompt_basic(self, causal_context, variable_descriptions):
        """Test basic do-calculus prompt generation."""
        simulator = DoOperatorSimulator(causal_context, variable_descriptions)
        
        interventions = {"X1": "treatment_A"}
        result = simulator.generate_do_prompt(interventions)
        
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Should contain key components
        assert "Base scenario:" in result
        assert "Intervention applied:" in result
        assert "do(X1 := treatment_A)" in result
        assert "Resulting scenario:" in result
        assert causal_context in result
    
    def test_generate_do_prompt_with_question(self, causal_context, variable_descriptions):
        """Test do-calculus prompt generation with custom question."""
        simulator = DoOperatorSimulator(causal_context, variable_descriptions)
        
        interventions = {"X1": "new_drug"}
        question = "What is the expected change in patient outcomes?"
        result = simulator.generate_do_prompt(interventions, question)
        
        assert question in result
        assert "do(X1 := new_drug)" in result
        # Should not contain default question
        assert "What is the expected impact" not in result
    
    def test_generate_do_prompt_multiple_interventions(self, causal_context, variable_descriptions):
        """Test do-calculus prompt with multiple interventions."""
        simulator = DoOperatorSimulator(causal_context, variable_descriptions)
        
        interventions = {
            "X1": "treatment_protocol_A",
            "X2": "high_response"
        }
        result = simulator.generate_do_prompt(interventions)
        
        assert "do(X1 := treatment_protocol_A, X2 := high_response)" in result
        assert "treatment_protocol_A" in result
        assert "high_response" in result
    
    def test_generate_do_prompt_no_question(self, causal_context, variable_descriptions):
        """Test do-calculus prompt without custom question."""
        simulator = DoOperatorSimulator(causal_context, variable_descriptions)
        
        interventions = {"X1": "intervention"}
        result = simulator.generate_do_prompt(interventions)
        
        # Should contain default question
        assert "What is the expected impact of this intervention?" in result
    
    def test_generate_do_prompt_empty_question(self, causal_context, variable_descriptions):
        """Test do-calculus prompt with empty question."""
        simulator = DoOperatorSimulator(causal_context, variable_descriptions)
        
        interventions = {"X1": "intervention"}
        result = simulator.generate_do_prompt(interventions, "")
        
        # Should use default question when empty string provided
        assert "What is the expected impact of this intervention?" in result
    
    def test_context_modification_isolation(self, causal_context, variable_descriptions):
        """Test that interventions don't affect original context."""
        simulator = DoOperatorSimulator(causal_context, variable_descriptions)
        original_context = simulator.base_context
        
        interventions = {"X1": "modified_value"}
        modified_context = simulator.intervene(interventions)
        
        # Original context should be unchanged
        assert simulator.base_context == original_context
        # Modified context should be different
        assert modified_context != original_context
        assert "modified_value" in modified_context
    
    def test_variable_state_persistence(self, causal_context, variable_descriptions):
        """Test that variable states persist between interventions."""
        simulator = DoOperatorSimulator(causal_context, variable_descriptions)
        
        # First intervention
        simulator.intervene({"X1": "first_intervention"})
        assert simulator.variables["X1"] == "first_intervention"
        
        # Second intervention on different variable
        simulator.intervene({"X2": "second_intervention"})
        assert simulator.variables["X1"] == "first_intervention"  # Should persist
        assert simulator.variables["X2"] == "second_intervention"
        
        # Third intervention on same variable as first
        simulator.intervene({"X1": "third_intervention"})
        assert simulator.variables["X1"] == "third_intervention"  # Should update
        assert simulator.variables["X2"] == "second_intervention"  # Should persist
    
    def test_prompt_structure_consistency(self, causal_context, variable_descriptions):
        """Test that prompt structure is consistent."""
        simulator = DoOperatorSimulator(causal_context, variable_descriptions)
        
        interventions = {"X1": "test_value"}
        prompt = simulator.generate_do_prompt(interventions)
        
        # Check for consistent structure
        lines = prompt.split('\n')
        
        # Should have proper sections
        base_scenario_found = any("Base scenario:" in line for line in lines)
        intervention_found = any("Intervention applied:" in line for line in lines)
        resulting_scenario_found = any("Resulting scenario:" in line for line in lines)
        
        assert base_scenario_found
        assert intervention_found
        assert resulting_scenario_found
        
        # Should have proper do-calculus notation
        assert "do(" in prompt
        assert ":=" in prompt
    
    @patch('causallm.core.do_operator.get_logger')
    @patch('causallm.core.do_operator.get_structured_logger')
    def test_logging_functionality(self, mock_structured_logger, mock_get_logger, 
                                 causal_context, variable_descriptions):
        """Test that logging works correctly."""
        mock_logger = Mock()
        mock_struct_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_structured_logger.return_value = mock_struct_logger
        
        simulator = DoOperatorSimulator(causal_context, variable_descriptions)
        
        # Should log initialization
        mock_logger.info.assert_called()
        mock_struct_logger.log_interaction.assert_called()
        
        # Reset mocks
        mock_logger.reset_mock()
        mock_struct_logger.reset_mock()
        
        # Test intervention logging
        interventions = {"X1": "test"}
        simulator.intervene(interventions)
        
        mock_logger.info.assert_called()
        mock_struct_logger.log_interaction.assert_called()
        
        # Reset mocks
        mock_logger.reset_mock()
        mock_struct_logger.reset_mock()
        
        # Test prompt generation logging
        simulator.generate_do_prompt(interventions)
        
        mock_logger.info.assert_called()
        mock_struct_logger.log_interaction.assert_called()


class TestDoOperatorComplexScenarios:
    """Test complex scenarios for do-operator."""
    
    def test_large_context_intervention(self):
        """Test intervention on large context."""
        # Create large context
        large_context = "This is a comprehensive medical study " * 100
        variables = {f"var_{i}": f"description_{i}" for i in range(20)}
        
        simulator = DoOperatorSimulator(large_context, variables)
        
        interventions = {"var_5": "modified_treatment"}
        result = simulator.intervene(interventions)
        
        assert "modified_treatment" in result
        assert len(result) >= len(large_context)
    
    def test_special_characters_in_variables(self):
        """Test intervention with special characters in variable names and values."""
        context = "Medical study with special variables: $treatment and @outcome"
        variables = {
            "$treatment": "standard_protocol",
            "@outcome": "recovery_rate",
            "var_with_underscore": "normal_variable"
        }
        
        simulator = DoOperatorSimulator(context, variables)
        
        interventions = {"$treatment": "experimental_protocol_v2.1"}
        result = simulator.intervene(interventions)
        
        assert "experimental_protocol_v2.1" in result
        assert simulator.variables["$treatment"] == "experimental_protocol_v2.1"
    
    def test_unicode_in_context_and_variables(self):
        """Test intervention with Unicode characters."""
        context = "Étude médicale avec des variables spéciales: traitement et résultat"
        variables = {
            "traitement": "protocole_standard",
            "résultat": "taux_de_récupération"
        }
        
        simulator = DoOperatorSimulator(context, variables)
        
        interventions = {"traitement": "protocole_expérimental"}
        result = simulator.intervene(interventions)
        
        assert "protocole_expérimental" in result
        assert simulator.variables["traitement"] == "protocole_expérimental"
    
    def test_nested_variable_replacements(self):
        """Test intervention where variables appear multiple times in context."""
        context = """
        Patient received treatment X1 at baseline. 
        The treatment X1 showed efficacy.
        Final assessment: X1 was successful.
        """
        variables = {"X1": "treatment"}
        
        simulator = DoOperatorSimulator(context, variables)
        
        interventions = {"X1": "new_therapy"}
        result = simulator.intervene(interventions)
        
        # All occurrences should be replaced
        assert result.count("new_therapy") >= 3
        assert "treatment" not in result  # Original value should be gone
    
    def test_intervention_order_independence(self, causal_context, variable_descriptions):
        """Test that intervention order doesn't affect final result."""
        simulator1 = DoOperatorSimulator(causal_context, variable_descriptions.copy())
        simulator2 = DoOperatorSimulator(causal_context, variable_descriptions.copy())
        
        interventions = {"X1": "treatment_A", "X2": "response_B"}
        
        # Apply interventions in different orders
        simulator1.intervene({"X1": "treatment_A"})
        simulator1.intervene({"X2": "response_B"})
        
        simulator2.intervene({"X2": "response_B"})
        simulator2.intervene({"X1": "treatment_A"})
        
        # Final states should be identical
        assert simulator1.variables == simulator2.variables
    
    def test_repeated_interventions(self, causal_context, variable_descriptions):
        """Test repeated interventions on same variable."""
        simulator = DoOperatorSimulator(causal_context, variable_descriptions)
        
        # Apply multiple interventions on same variable
        simulator.intervene({"X1": "treatment_v1"})
        assert simulator.variables["X1"] == "treatment_v1"
        
        simulator.intervene({"X1": "treatment_v2"})
        assert simulator.variables["X1"] == "treatment_v2"
        
        simulator.intervene({"X1": "treatment_v3"})
        assert simulator.variables["X1"] == "treatment_v3"
        
        # Only latest intervention should persist
        final_context = simulator.intervene({"X1": "final_treatment"})
        assert "final_treatment" in final_context
        assert "treatment_v1" not in final_context
        assert "treatment_v2" not in final_context
        assert "treatment_v3" not in final_context
    
    def test_prompt_generation_with_long_interventions(self, causal_context, variable_descriptions):
        """Test prompt generation with very long intervention values."""
        simulator = DoOperatorSimulator(causal_context, variable_descriptions)
        
        long_intervention = "very_long_treatment_protocol_" * 20
        interventions = {"X1": long_intervention}
        
        prompt = simulator.generate_do_prompt(interventions)
        
        assert long_intervention in prompt
        assert f"do(X1 := {long_intervention})" in prompt
        assert len(prompt) > 1000  # Should be substantial
    
    def test_concurrent_interventions_simulation(self, causal_context):
        """Test simulation of concurrent interventions."""
        variables = {
            "drug_A": "not_administered",
            "drug_B": "not_administered", 
            "dosage": "standard",
            "timing": "morning"
        }
        
        simulator = DoOperatorSimulator(causal_context, variables)
        
        # Simulate concurrent interventions
        concurrent_interventions = {
            "drug_A": "administered",
            "drug_B": "administered",
            "dosage": "high",
            "timing": "evening"
        }
        
        prompt = simulator.generate_do_prompt(concurrent_interventions)
        
        # Should handle multiple concurrent interventions
        assert "do(drug_A := administered, drug_B := administered, dosage := high, timing := evening)" in prompt
        assert "administered" in prompt
        assert "high" in prompt
        assert "evening" in prompt
    
    def test_memory_efficiency_large_interventions(self):
        """Test memory efficiency with large number of interventions."""
        import sys
        
        # Create scenario with many variables
        context = "Large scale clinical trial with many variables"
        variables = {f"var_{i}": f"value_{i}" for i in range(100)}
        
        simulator = DoOperatorSimulator(context, variables)
        
        # Measure memory before intervention
        initial_size = sys.getsizeof(simulator.variables)
        
        # Apply many interventions
        interventions = {f"var_{i}": f"new_value_{i}" for i in range(50)}
        result = simulator.intervene(interventions)
        
        # Memory should not grow excessively
        final_size = sys.getsizeof(simulator.variables)
        assert final_size < initial_size * 2  # Reasonable growth
        
        # Functionality should still work
        assert isinstance(result, str)
        assert len(result) > 0


class TestDoOperatorErrorHandling:
    """Test error handling in do-operator."""
    
    def test_intervene_error_logging(self, causal_context, variable_descriptions):
        """Test that intervention errors are logged properly."""
        simulator = DoOperatorSimulator(causal_context, variable_descriptions)
        
        with patch('causallm.core.do_operator.get_structured_logger') as mock_struct_logger:
            mock_logger = Mock()
            mock_struct_logger.return_value = mock_logger
            
            try:
                simulator.intervene({"invalid_var": "value"})
            except ValueError:
                pass
            
            # Should log the error
            mock_logger.log_error.assert_called()
    
    def test_generate_do_prompt_error_logging(self, causal_context, variable_descriptions):
        """Test that prompt generation errors are logged."""
        simulator = DoOperatorSimulator(causal_context, variable_descriptions)
        
        with patch.object(simulator, 'intervene', side_effect=ValueError("Test error")), \
             patch('causallm.core.do_operator.get_structured_logger') as mock_struct_logger:
            
            mock_logger = Mock()
            mock_struct_logger.return_value = mock_logger
            
            try:
                simulator.generate_do_prompt({"invalid": "test"})
            except ValueError:
                pass
            
            # Should log the error
            mock_logger.log_error.assert_called()
    
    def test_empty_variable_value_handling(self, causal_context, variable_descriptions):
        """Test handling of empty variable values."""
        simulator = DoOperatorSimulator(causal_context, variable_descriptions)
        
        interventions = {"X1": ""}
        result = simulator.intervene(interventions)
        
        # Should handle empty values gracefully
        assert isinstance(result, str)
        assert simulator.variables["X1"] == ""
    
    def test_none_variable_value_handling(self, causal_context, variable_descriptions):
        """Test handling of None variable values."""
        simulator = DoOperatorSimulator(causal_context, variable_descriptions)
        
        # This might raise an error or handle gracefully
        interventions = {"X1": None}
        
        try:
            result = simulator.intervene(interventions)
            # If it succeeds, check the result
            assert isinstance(result, str)
        except (TypeError, ValueError):
            # If it fails, that's also acceptable behavior
            pass
    
    def test_variable_name_case_sensitivity(self, causal_context, variable_descriptions):
        """Test that variable names are case-sensitive."""
        simulator = DoOperatorSimulator(causal_context, variable_descriptions)
        
        # Should fail with wrong case
        with pytest.raises(ValueError):
            simulator.intervene({"x1": "value"})  # lowercase instead of X1
        
        with pytest.raises(ValueError):
            simulator.intervene({"X1 ": "value"})  # extra space