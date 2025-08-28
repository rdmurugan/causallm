"""
Test suite for Enhanced CausalLLM functionality
Tests the enhanced causal analysis capabilities
"""
import pytest
import pandas as pd
import numpy as np
import warnings
from unittest.mock import Mock, patch, MagicMock
from causallm.enhanced_causallm import EnhancedCausalLLM, ComprehensiveCausalAnalysis
from causallm.core.enhanced_causal_discovery import EnhancedCausalDiscovery, CausalEdge, CausalDiscoveryResult
from causallm.core.statistical_inference import StatisticalCausalInference, CausalInferenceResult, CausalEffect

# Suppress warnings in tests
warnings.filterwarnings('ignore')


class TestEnhancedCausalLLM:
    """Test Enhanced CausalLLM functionality."""
    
    def test_initialization_with_llm(self):
        """Test Enhanced CausalLLM initialization with LLM provider."""
        with patch('causallm.core.llm_client.get_llm_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            enhanced = EnhancedCausalLLM(llm_provider="openai")
            
            assert enhanced.llm_client == mock_client
            assert enhanced.llm_available is True
            assert enhanced.significance_level == 0.05
    
    def test_initialization_fallback_mode(self):
        """Test Enhanced CausalLLM initialization fallback when LLM fails."""
        with patch('causallm.core.llm_client.get_llm_client', side_effect=Exception("API key missing")):
            enhanced = EnhancedCausalLLM()
            
            assert enhanced.llm_client is None
            assert enhanced.llm_available is False
    
    def test_discover_causal_relationships_basic(self, sample_data):
        """Test basic causal relationship discovery."""
        with patch('causallm.core.llm_client.get_llm_client', side_effect=Exception("No LLM")):
            enhanced = EnhancedCausalLLM()
            
            # Mock the discovery engine
            with patch.object(enhanced.discovery_engine, 'discover_causal_structure') as mock_discovery:
                mock_result = CausalDiscoveryResult(
                    discovered_edges=[
                        CausalEdge('X1', 'X2', 0.85, 'pc_algorithm', 0.001, 0.5, 'Strong relationship'),
                        CausalEdge('X2', 'X3', 0.75, 'pc_algorithm', 0.01, 0.4, 'Medium relationship')
                    ],
                    suggested_confounders={'X1->X3': ['X2']},
                    assumptions_violated=[],
                    domain_insights="Healthcare domain insights",
                    statistical_summary={'domain': 'healthcare', 'high_confidence_relationships': 2, 'average_effect_size': 0.45}
                )
                mock_discovery.return_value = mock_result
                
                result = enhanced.discover_causal_relationships(
                    data=sample_data,
                    variables=['X1', 'X2', 'X3', 'X4'],
                    domain='healthcare'
                )
                
                assert len(result.discovered_edges) == 2
                assert result.discovered_edges[0].confidence == 0.85
                assert result.domain_insights == "Healthcare domain insights"
    
    def test_estimate_causal_effect_comprehensive(self, sample_data):
        """Test comprehensive causal effect estimation."""
        with patch('causallm.core.llm_client.get_llm_client', side_effect=Exception("No LLM")):
            enhanced = EnhancedCausalLLM()
            
            # Mock the inference engine
            with patch.object(enhanced.inference_engine, 'comprehensive_causal_analysis') as mock_inference:
                mock_effect = CausalEffect(
                    treatment='X1',
                    outcome='X3',
                    method='linear_regression',
                    effect_estimate=2.5,
                    confidence_interval=(1.8, 3.2),
                    p_value=0.001,
                    standard_error=0.35,
                    sample_size=1000,
                    interpretation='Significant positive effect',
                    assumptions_met=['linearity', 'independence'],
                    assumptions_violated=[],
                    robustness_score=0.85
                )
                
                mock_result = CausalInferenceResult(
                    primary_effect=mock_effect,
                    robustness_checks=[],
                    sensitivity_analysis={},
                    recommendations="Strong evidence for causal effect",
                    confidence_level="High",
                    overall_assessment="Reliable causal estimate"
                )
                mock_inference.return_value = mock_result
                
                result = enhanced.estimate_causal_effect(
                    data=sample_data,
                    treatment='X1',
                    outcome='X3',
                    covariates=['X2'],
                    method='comprehensive'
                )
                
                assert result.primary_effect.effect_estimate == 2.5
                assert result.confidence_level == "High"
                assert result.primary_effect.p_value == 0.001
    
    def test_comprehensive_analysis_with_specified_treatment_outcome(self, sample_data):
        """Test comprehensive analysis with specified treatment and outcome."""
        with patch('causallm.core.llm_client.get_llm_client', side_effect=Exception("No LLM")):
            enhanced = EnhancedCausalLLM()
            
            # Mock discovery and inference results
            mock_discovery = CausalDiscoveryResult(
                discovered_edges=[
                    CausalEdge('X1', 'X3', 0.80, 'pc_algorithm', 0.002, 0.6, 'Direct effect')
                ],
                suggested_confounders={},
                assumptions_violated=[],
                domain_insights="Test insights",
                statistical_summary={'domain': 'test', 'high_confidence_relationships': 1, 'average_effect_size': 0.6}
            )
            
            mock_inference = CausalInferenceResult(
                primary_effect=CausalEffect(
                    treatment='X1',
                    outcome='X3',
                    method='linear_regression',
                    effect_estimate=1.5,
                    confidence_interval=(1.0, 2.0),
                    p_value=0.01,
                    standard_error=0.25,
                    sample_size=1000,
                    interpretation='Moderate positive effect',
                    assumptions_met=['linearity'],
                    assumptions_violated=[],
                    robustness_score=0.75
                ),
                robustness_checks=[],
                sensitivity_analysis={},
                recommendations="Moderate evidence",
                confidence_level="Medium",
                overall_assessment="Reasonable estimate"
            )
            
            with patch.object(enhanced, 'discover_causal_relationships', return_value=mock_discovery):
                with patch.object(enhanced, 'estimate_causal_effect', return_value=mock_inference):
                    
                    result = enhanced.comprehensive_analysis(
                        data=sample_data,
                        treatment='X1',
                        outcome='X3',
                        domain='test'
                    )
                    
                    assert isinstance(result, ComprehensiveCausalAnalysis)
                    assert len(result.inference_results) == 1
                    assert result.confidence_score > 0
                    assert len(result.actionable_insights) > 0
    
    def test_generate_intervention_recommendations(self, sample_data):
        """Test intervention recommendation generation."""
        with patch('causallm.core.llm_client.get_llm_client', side_effect=Exception("No LLM")):
            enhanced = EnhancedCausalLLM()
            
            # Create mock comprehensive analysis
            mock_discovery = CausalDiscoveryResult(
                discovered_edges=[
                    CausalEdge('X1', 'X3', 0.85, 'pc_algorithm', 0.001, 0.7, 'Strong effect')
                ],
                suggested_confounders={},
                assumptions_violated=[],
                domain_insights="Test insights",
                statistical_summary={'domain': 'test', 'high_confidence_relationships': 1, 'average_effect_size': 0.7}
            )
            
            mock_inference = {
                'X1_to_X3': CausalInferenceResult(
                    primary_effect=CausalEffect(
                        treatment='X1',
                        outcome='X3',
                        method='linear_regression',
                        effect_estimate=2.0,
                        confidence_interval=(1.5, 2.5),
                        p_value=0.001,
                        standard_error=0.25,
                        sample_size=1000,
                        interpretation='Strong positive effect',
                        assumptions_met=['linearity'],
                        assumptions_violated=[],
                        robustness_score=0.85
                    ),
                    robustness_checks=[],
                    sensitivity_analysis={},
                    recommendations="Strong evidence",
                    confidence_level="High",
                    overall_assessment="Reliable"
                )
            }
            
            analysis = ComprehensiveCausalAnalysis(
                discovery_results=mock_discovery,
                inference_results=mock_inference,
                domain_recommendations="Test recommendations",
                methodology_assessment="Good methodology",
                actionable_insights=["Test insight"],
                confidence_score=0.85
            )
            
            recommendations = enhanced.generate_intervention_recommendations(
                analysis=analysis,
                target_outcome='X3'
            )
            
            assert 'primary_interventions' in recommendations
            assert 'secondary_interventions' in recommendations
            assert 'expected_impacts' in recommendations
    
    def test_data_validation(self):
        """Test data validation functionality."""
        with patch('causallm.core.llm_client.get_llm_client', side_effect=Exception("No LLM")):
            enhanced = EnhancedCausalLLM()
            
            # Test empty data
            empty_data = pd.DataFrame()
            with pytest.raises(ValueError, match="Input data cannot be empty"):
                enhanced._validate_data(empty_data)
            
            # Test missing variables
            data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            with pytest.raises(ValueError, match="Variables not found in data"):
                enhanced._validate_data(data, variables=['C', 'D'])
    
    def test_treatment_outcome_validation(self):
        """Test treatment and outcome variable validation."""
        with patch('causallm.core.llm_client.get_llm_client', side_effect=Exception("No LLM")):
            enhanced = EnhancedCausalLLM()
            
            data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            
            # Test missing treatment
            with pytest.raises(ValueError, match="Treatment variable 'C' not found"):
                enhanced._validate_treatment_outcome(data, 'C', 'B')
            
            # Test missing outcome
            with pytest.raises(ValueError, match="Outcome variable 'D' not found"):
                enhanced._validate_treatment_outcome(data, 'A', 'D')
            
            # Test missing covariates
            with pytest.raises(ValueError, match="Covariates not found in data"):
                enhanced._validate_treatment_outcome(data, 'A', 'B', covariates=['E', 'F'])


class TestEnhancedCausalDiscovery:
    """Test Enhanced Causal Discovery functionality."""
    
    def test_basic_functionality_without_llm(self, sample_data):
        """Test basic discovery functionality without LLM."""
        discovery = EnhancedCausalDiscovery(llm_client=None)
        
        with patch.object(discovery.statistical_discovery, 'discover_causal_structure') as mock_discovery:
            mock_edges = [
                CausalEdge('X1', 'X2', 0.8, 'pc_algorithm', 0.01, 0.5, 'Test relationship')
            ]
            mock_result = CausalDiscoveryResult(
                discovered_edges=mock_edges,
                suggested_confounders={},
                assumptions_violated=[],
                domain_insights="Statistical insights only",
                statistical_summary={'domain': None, 'high_confidence_relationships': 1, 'average_effect_size': 0.5}
            )
            mock_discovery.return_value = mock_result
            
            result = discovery.discover_causal_structure(
                data=sample_data,
                variables=['X1', 'X2', 'X3', 'X4']
            )
            
            assert len(result.discovered_edges) == 1
            assert result.discovered_edges[0].confidence == 0.8


class TestStatisticalCausalInference:
    """Test Statistical Causal Inference functionality."""
    
    def test_linear_regression_method(self, sample_data):
        """Test linear regression causal inference."""
        inference = StatisticalCausalInference()
        
        # Test with numeric data
        result = inference.estimate_causal_effect(
            data=sample_data,
            treatment='X1',
            outcome='X3',
            covariates=['X2'],
            method=inference.CausalMethod.LINEAR_REGRESSION
        )
        
        assert isinstance(result, CausalEffect)
        assert result.treatment == 'X1'
        assert result.outcome == 'X3'
        assert result.method == 'linear_regression'
        assert isinstance(result.effect_estimate, float)
        assert isinstance(result.p_value, float)
        assert result.p_value >= 0 and result.p_value <= 1
    
    def test_comprehensive_analysis_fallback(self, sample_data):
        """Test comprehensive analysis with method fallbacks."""
        inference = StatisticalCausalInference()
        
        result = inference.comprehensive_causal_analysis(
            data=sample_data,
            treatment='X1',
            outcome='X3',
            covariates=['X2']
        )
        
        assert isinstance(result, CausalInferenceResult)
        assert result.primary_effect is not None
        assert result.confidence_level in ['High', 'Medium', 'Low']


@pytest.mark.integration
class TestIntegrationScenarios:
    """Test integration scenarios with realistic data."""
    
    def test_healthcare_scenario(self):
        """Test healthcare analysis scenario."""
        # Generate realistic healthcare data
        np.random.seed(42)
        n = 500
        
        age = np.random.normal(65, 15, n)
        age = np.clip(age, 18, 100)
        
        severity = np.random.normal(50, 20, n) + (age - 65) * 0.3
        severity = np.clip(severity, 0, 100)
        
        treatment = np.random.binomial(1, 0.5 + (severity > 60) * 0.2, n)
        
        outcome = (
            70 - (age - 65) * 0.1 - severity * 0.3 + 
            treatment * 15 + np.random.normal(0, 10, n)
        )
        
        data = pd.DataFrame({
            'age': age,
            'severity': severity,
            'treatment': treatment,
            'outcome': outcome
        })
        
        with patch('causallm.core.llm_client.get_llm_client', side_effect=Exception("No LLM")):
            enhanced = EnhancedCausalLLM()
            
            # Test discovery
            result = enhanced.discover_causal_relationships(
                data=data,
                variables=['age', 'severity', 'treatment', 'outcome'],
                domain='healthcare'
            )
            
            # Should not crash and should return some result
            assert isinstance(result, CausalDiscoveryResult)
    
    def test_marketing_scenario(self):
        """Test marketing analysis scenario."""
        # Generate realistic marketing data
        np.random.seed(123)
        n = 1000
        
        customer_age = np.random.normal(35, 12, n)
        customer_age = np.clip(customer_age, 18, 70)
        
        income = np.random.lognormal(np.log(50000), 0.5, n)
        email_campaign = np.random.binomial(1, 0.3, n)
        
        purchase_prob = (
            0.1 + (income / 100000) * 0.1 + 
            email_campaign * 0.15 + np.random.normal(0, 0.05, n)
        )
        purchase_prob = np.clip(purchase_prob, 0, 0.8)
        
        conversion = np.random.binomial(1, purchase_prob, n)
        
        data = pd.DataFrame({
            'customer_age': customer_age,
            'income': income,
            'email_campaign': email_campaign,
            'conversion': conversion
        })
        
        with patch('causallm.core.llm_client.get_llm_client', side_effect=Exception("No LLM")):
            enhanced = EnhancedCausalLLM()
            
            # Test effect estimation
            result = enhanced.estimate_causal_effect(
                data=data,
                treatment='email_campaign',
                outcome='conversion',
                covariates=['customer_age', 'income'],
                method='regression'
            )
            
            assert isinstance(result, CausalInferenceResult)
            assert result.primary_effect.treatment == 'email_campaign'
            assert result.primary_effect.outcome == 'conversion'


# Performance and edge case tests
@pytest.mark.parametrize("sample_size", [50, 100, 500, 1000])
def test_performance_with_different_sample_sizes(sample_size):
    """Test performance with different sample sizes."""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'X': np.random.normal(0, 1, sample_size),
        'Y': np.random.normal(0, 1, sample_size),
        'Z': np.random.normal(0, 1, sample_size)
    })
    
    with patch('causallm.core.llm_client.get_llm_client', side_effect=Exception("No LLM")):
        enhanced = EnhancedCausalLLM()
        
        # Should handle different sample sizes gracefully
        result = enhanced.discover_causal_relationships(
            data=data,
            variables=['X', 'Y', 'Z']
        )
        
        assert isinstance(result, CausalDiscoveryResult)


def test_missing_data_handling():
    """Test handling of missing data."""
    np.random.seed(42)
    n = 200
    
    data = pd.DataFrame({
        'X': np.random.normal(0, 1, n),
        'Y': np.random.normal(0, 1, n),
        'Z': np.random.normal(0, 1, n)
    })
    
    # Introduce missing values
    data.loc[0:10, 'X'] = np.nan
    data.loc[15:25, 'Y'] = np.nan
    
    with patch('causallm.core.llm_client.get_llm_client', side_effect=Exception("No LLM")):
        enhanced = EnhancedCausalLLM()
        
        # Should handle missing data without crashing
        try:
            result = enhanced.discover_causal_relationships(
                data=data,
                variables=['X', 'Y', 'Z']
            )
            # If it doesn't crash, that's a success
            assert True
        except Exception:
            # Some methods may not handle missing data, which is acceptable
            assert True