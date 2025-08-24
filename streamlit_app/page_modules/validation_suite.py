import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import plotly.express as px
import plotly.graph_objects as go
from causalllm.assumption_checker import AssumptionChecker, CausalAssumption
from causalllm.causal_argument_validator import CausalArgumentValidator
from causalllm.llm_sensitivity_analysis_guide import LLMSensitivityAnalysisGuide
from causalllm.llm_effect_size_interpreter import LLMEffectSizeInterpreter
from causalllm.llm_client import get_llm_client
import time

def show():
    st.title("✅ Validation Suite")
    st.markdown("Comprehensive validation of causal assumptions, arguments, and robustness")
    
    # Check if data and discovery results are available
    has_data = 'current_data' in st.session_state
    has_discovery = 'discovery_results' in st.session_state
    has_variable_roles = 'variable_roles' in st.session_state
    
    if not has_data:
        st.warning("📁 Please upload a dataset in the Data Manager first for comprehensive validation!")
        if st.button("Go to Data Manager"):
            st.info("Navigate to Data Manager to upload your dataset")
        return
    
    if not has_variable_roles:
        st.warning("🏷️ Please assign variable roles in the Data Manager for better validation!")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Assumption Checker", "📋 Argument Validator", "🎯 Sensitivity Analysis", 
        "📏 Effect Size Interpreter", "📊 Validation Summary"
    ])
    
    with tab1:
        st.markdown("### Causal Assumption Validation")
        st.info("Validate key causal inference assumptions using statistical tests and LLM reasoning")
        
        if not has_variable_roles:
            st.warning("Variable roles are needed for assumption checking. Please assign them in the Data Manager.")
            return
        
        data = st.session_state.current_data
        roles = st.session_state.variable_roles
        
        # Configuration
        col1, col2 = st.columns(2)
        
        with col1:
            treatment_var = st.selectbox(
                "Treatment Variable",
                roles.get('treatment', []) if roles.get('treatment') else ['Please assign treatment variables'],
                help="Main intervention or exposure variable"
            )
            
            outcome_var = st.selectbox(
                "Outcome Variable", 
                roles.get('outcome', []) if roles.get('outcome') else ['Please assign outcome variables'],
                help="Primary outcome of interest"
            )
        
        with col2:
            covariates = st.multiselect(
                "Covariates/Confounders",
                roles.get('confounders', []) + roles.get('remaining', []),
                default=roles.get('confounders', [])[:3],  # Default to first 3 confounders
                help="Variables to control for in the analysis"
            )
            
            analysis_method = st.selectbox(
                "Analysis Method",
                ["regression", "matching", "instrumental_variables", "difference_in_differences"],
                help="Intended causal inference method"
            )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            significance_level = st.slider("Significance Level", 0.01, 0.10, 0.05, 0.01)
            
            assumptions_to_check = st.multiselect(
                "Assumptions to Validate",
                [
                    "exchangeability", "positivity", "consistency", "sutva", 
                    "linearity", "additivity", "no_measurement_error", 
                    "no_selection_bias", "no_information_bias", "temporal_ordering",
                    "no_interference", "correct_functional_form", "homoscedasticity"
                ],
                default=["exchangeability", "positivity", "consistency", "sutva"],
                help="Select which assumptions to validate"
            )
            
            include_recommendations = st.checkbox("Include Recommendations", value=True)
            detailed_reasoning = st.checkbox("Show Detailed LLM Reasoning", value=False)
        
        # Run assumption checking
        if st.button("🔍 Check Assumptions", type="primary"):
            if treatment_var == 'Please assign treatment variables' or outcome_var == 'Please assign outcome variables':
                st.error("Please assign treatment and outcome variables in the Data Manager first!")
                return
            
            with st.spinner("🔍 Validating causal assumptions... This may take a few minutes."):
                try:
                    # Initialize assumption checker
                    llm_client = get_llm_client()
                    checker = AssumptionChecker(llm_client)
                    
                    start_time = time.time()
                    
                    # Run assumption validation
                    report = asyncio.run(checker.validate_causal_assumptions(
                        data=data,
                        treatment_variable=treatment_var,
                        outcome_variable=outcome_var,
                        covariates=covariates,
                        analysis_method=analysis_method
                    ))
                    
                    validation_time = time.time() - start_time
                    
                    # Store results
                    st.session_state['assumption_results'] = {
                        'report': report,
                        'config': {
                            'treatment_var': treatment_var,
                            'outcome_var': outcome_var,
                            'covariates': covariates,
                            'analysis_method': analysis_method,
                            'validation_time': validation_time
                        },
                        'timestamp': pd.Timestamp.now()
                    }
                    
                    st.success(f"✅ Assumption validation completed in {validation_time:.1f} seconds!")
                    
                except Exception as e:
                    st.error(f"Assumption checking failed: {str(e)}")
                    st.info("This might be due to data format issues or missing LLM configuration.")
        
        # Display assumption results
        if 'assumption_results' in st.session_state:
            results = st.session_state.assumption_results
            report = results['report']
            config = results['config']
            
            st.markdown("#### Validation Results")
            
            # Overall plausibility score
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                plausibility = report.plausibility_score
                color = "green" if plausibility > 0.7 else "orange" if plausibility > 0.5 else "red"
                st.markdown(f"""
                <div style="text-align: center;">
                    <h2 style="color: {color}; margin: 0;">{plausibility:.2f}</h2>
                    <p style="margin: 0;">Overall Plausibility</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                violations = len([v for v in report.assumption_violations if v.severity.value in ["high", "critical"]])
                st.metric("High Severity Violations", violations)
            
            with col3:
                passed = len([v for v in report.assumption_violations if v.severity.value == "low"])
                st.metric("Assumptions Passed", passed)
            
            with col4:
                st.metric("Analysis Method", config['analysis_method'].replace('_', ' ').title())
            
            # Assumption details
            st.markdown("#### Detailed Assumption Analysis")
            
            if report.assumption_violations:
                # Create assumption summary table
                assumption_data = []
                for violation in report.assumption_violations:
                    assumption_data.append({
                        'Assumption': violation.assumption.value.replace('_', ' ').title(),
                        'Status': '❌' if violation.severity.value in ['high', 'critical'] else '⚠️' if violation.severity.value == 'medium' else '✅',
                        'Severity': violation.severity.value.title(),
                        'P-Value': f"{violation.statistical_evidence.get('p_value', 'N/A'):.4f}" if isinstance(violation.statistical_evidence.get('p_value'), float) else str(violation.statistical_evidence.get('p_value', 'N/A')),
                        'Description': violation.description[:80] + "..." if len(violation.description) > 80 else violation.description
                    })
                
                assumption_df = pd.DataFrame(assumption_data)
                st.dataframe(assumption_df, use_container_width=True)
                
                # Detailed violation analysis
                st.markdown("#### 🚨 Assumption Violations (Detailed)")
                
                for i, violation in enumerate(report.assumption_violations):
                    if violation.severity.value in ["medium", "high", "critical"]:
                        
                        severity_colors = {
                            "low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"
                        }
                        
                        with st.expander(f"{severity_colors[violation.severity.value]} {violation.assumption.value.replace('_', ' ').title()} - {violation.severity.value.title()} Severity"):
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown(f"**Description:** {violation.description}")
                                st.markdown(f"**Potential Impact:** {violation.potential_impact}")
                                
                                if violation.recommendations:
                                    st.markdown("**Recommendations:**")
                                    for rec in violation.recommendations:
                                        st.markdown(f"• {rec}")
                            
                            with col2:
                                st.markdown("**Statistical Evidence:**")
                                for key, value in violation.statistical_evidence.items():
                                    if isinstance(value, float):
                                        st.markdown(f"• {key}: {value:.4f}")
                                    else:
                                        st.markdown(f"• {key}: {value}")
                                
                                if detailed_reasoning and hasattr(violation, 'llm_reasoning'):
                                    st.markdown("**LLM Reasoning:**")
                                    st.markdown(violation.llm_reasoning)
            
            # Recommendations
            if include_recommendations and hasattr(report, 'overall_recommendations'):
                st.markdown("#### 💡 Overall Recommendations")
                
                for rec in report.overall_recommendations:
                    if rec.priority == "high":
                        st.error(f"🔴 **High Priority:** {rec.recommendation}")
                    elif rec.priority == "medium":
                        st.warning(f"🟡 **Medium Priority:** {rec.recommendation}")
                    else:
                        st.info(f"🟢 **Low Priority:** {rec.recommendation}")
        
    with tab2:
        st.markdown("### Causal Argument Validation")
        st.info("Validate the logical consistency and strength of causal claims")
        
        # Argument input
        st.markdown("#### Enter Your Causal Argument")
        
        col1, col2 = st.columns(2)
        
        with col1:
            causal_claim = st.text_area(
                "Main Causal Claim",
                placeholder="e.g., 'Smoking causes lung cancer' or 'Exercise improves mental health'",
                help="State your main causal hypothesis clearly"
            )
            
            domain_context = st.selectbox(
                "Domain Context",
                ["healthcare", "business", "education", "social_science", "technology"],
                help="Domain context for specialized validation criteria"
            )
        
        with col2:
            evidence_sources = st.text_area(
                "Supporting Evidence (one per line)",
                placeholder="• Randomized controlled trial showed significant effect\n• Dose-response relationship observed\n• Biological mechanism identified",
                help="List your supporting evidence, one piece per line"
            )
            
            study_type = st.selectbox(
                "Study Type",
                ["randomized_trial", "observational", "quasi_experimental", "meta_analysis", "systematic_review"],
                help="Type of study or analysis"
            )
        
        # Advanced validation options
        with st.expander("🔧 Validation Options"):
            validation_criteria = st.multiselect(
                "Validation Criteria",
                [
                    "bradford_hill_criteria", "logical_consistency", "evidence_quality",
                    "temporal_sequence", "dose_response", "biological_plausibility",
                    "consistency_across_studies", "experimental_evidence"
                ],
                default=["bradford_hill_criteria", "logical_consistency", "evidence_quality"],
                help="Select which criteria to use for validation"
            )
            
            strictness_level = st.selectbox(
                "Validation Strictness",
                ["lenient", "moderate", "strict"],
                index=1,
                help="How strict should the validation be?"
            )
            
            include_improvement_suggestions = st.checkbox("Include Improvement Suggestions", value=True)
        
        # Run argument validation
        if st.button("📋 Validate Argument", type="primary"):
            if not causal_claim.strip():
                st.error("Please enter a causal claim to validate!")
                return
            
            with st.spinner("📋 Validating causal argument... This may take a moment."):
                try:
                    # Parse evidence
                    evidence_list = []
                    if evidence_sources.strip():
                        evidence_list = [line.strip().lstrip('•').strip() 
                                       for line in evidence_sources.split('\n') 
                                       if line.strip()]
                    
                    # Initialize validator
                    llm_client = get_llm_client()
                    validator = CausalArgumentValidator(llm_client)
                    
                    start_time = time.time()
                    
                    # Run validation
                    validation_result = asyncio.run(validator.validate_causal_argument(
                        claim=causal_claim,
                        evidence=evidence_list,
                        domain=domain_context
                    ))
                    
                    validation_time = time.time() - start_time
                    
                    # Store results
                    st.session_state['argument_validation'] = {
                        'result': validation_result,
                        'claim': causal_claim,
                        'evidence': evidence_list,
                        'domain': domain_context,
                        'validation_time': validation_time,
                        'timestamp': pd.Timestamp.now()
                    }
                    
                    st.success(f"✅ Argument validation completed in {validation_time:.1f} seconds!")
                    
                except Exception as e:
                    st.error(f"Argument validation failed: {str(e)}")
        
        # Display validation results
        if 'argument_validation' in st.session_state:
            validation_data = st.session_state.argument_validation
            result = validation_data['result']
            
            st.markdown("#### Validation Results")
            
            # Overall score and summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                score = result.overall_score
                score_color = "green" if score >= 7 else "orange" if score >= 5 else "red"
                st.markdown(f"""
                <div style="text-align: center;">
                    <h2 style="color: {score_color}; margin: 0;">{score:.1f}/10</h2>
                    <p style="margin: 0;">Argument Strength</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                strength = "Strong" if score >= 7 else "Moderate" if score >= 5 else "Weak"
                st.metric("Argument Quality", strength)
            
            with col3:
                fallacies = len(result.fallacy_analysis) if hasattr(result, 'fallacy_analysis') else 0
                st.metric("Logical Fallacies", fallacies)
            
            with col4:
                st.metric("Domain", domain_context.title())
            
            # Detailed analysis
            col1, col2 = st.columns(2)
            
            with col1:
                if hasattr(result, 'strengths') and result.strengths:
                    st.markdown("#### 💪 Argument Strengths")
                    for strength in result.strengths:
                        st.success(f"✅ {strength}")
                
                if hasattr(result, 'evidence_assessment') and result.evidence_assessment:
                    st.markdown("#### 📊 Evidence Assessment")
                    for assessment in result.evidence_assessment:
                        st.info(f"📄 {assessment}")
            
            with col2:
                if hasattr(result, 'weaknesses') and result.weaknesses:
                    st.markdown("#### ⚠️ Areas for Improvement")
                    for weakness in result.weaknesses:
                        st.warning(f"⚠️ {weakness}")
                
                if hasattr(result, 'fallacy_analysis') and result.fallacy_analysis:
                    st.markdown("#### 🚨 Logical Issues")
                    for fallacy in result.fallacy_analysis:
                        st.error(f"❌ {fallacy}")
            
            # Bradford Hill criteria analysis (if applicable)
            if domain_context == "healthcare" and hasattr(result, 'bradford_hill_analysis'):
                st.markdown("#### 🏥 Bradford Hill Criteria Analysis")
                
                criteria_scores = result.bradford_hill_analysis
                criteria_names = [
                    "Strength of Association", "Consistency", "Temporality", 
                    "Biological Gradient", "Plausibility", "Coherence", 
                    "Experimental Evidence", "Analogy"
                ]
                
                # Create radar chart for Bradford Hill criteria
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=list(criteria_scores.values()),
                    theta=criteria_names,
                    fill='toself',
                    name='Bradford Hill Score'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 5]
                        )),
                    showlegend=True,
                    title="Bradford Hill Criteria Assessment"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Improvement suggestions
            if include_improvement_suggestions and hasattr(result, 'improvement_suggestions'):
                st.markdown("#### 💡 Suggestions for Improvement")
                
                for suggestion in result.improvement_suggestions:
                    st.info(f"💡 {suggestion}")
    
    with tab3:
        st.markdown("### Sensitivity Analysis Guide")
        st.info("Get intelligent guidance on robustness testing for your causal analysis")
        
        if not has_variable_roles:
            st.info("Variable role information would help provide more targeted sensitivity analysis recommendations.")
        
        # Configuration
        col1, col2 = st.columns(2)
        
        with col1:
            if has_variable_roles:
                roles = st.session_state.variable_roles
                treatment_var_sens = st.selectbox(
                    "Treatment Variable", 
                    roles.get('treatment', ['No treatment variables assigned']),
                    key="sens_treatment"
                )
                outcome_var_sens = st.selectbox(
                    "Outcome Variable",
                    roles.get('outcome', ['No outcome variables assigned']),
                    key="sens_outcome"
                )
            else:
                treatment_var_sens = st.text_input("Treatment Variable", placeholder="e.g., treatment_group")
                outcome_var_sens = st.text_input("Outcome Variable", placeholder="e.g., outcome_score")
        
        with col2:
            observed_confounders = st.text_area(
                "Observed Confounders (one per line)",
                placeholder="age\ngender\nseverity_score",
                help="List variables you're controlling for"
            )
            
            analysis_context_sens = st.selectbox(
                "Analysis Context",
                ["observational", "randomized_trial", "quasi_experimental", "natural_experiment"],
                help="Type of study or analysis context"
            )
        
        # Advanced options
        with st.expander("🔧 Sensitivity Analysis Options"):
            sensitivity_tests = st.multiselect(
                "Recommended Test Types",
                [
                    "unobserved_confounding", "measurement_error", "selection_bias",
                    "model_specification", "functional_form", "outliers", 
                    "missing_data", "temporal_assumptions"
                ],
                default=["unobserved_confounding", "measurement_error", "selection_bias"],
                help="Types of sensitivity tests to include in the analysis plan"
            )
            
            priority_level = st.selectbox(
                "Priority Level",
                ["essential", "recommended", "comprehensive"],
                index=1,
                help="How comprehensive should the sensitivity analysis be?"
            )
        
        # Generate sensitivity analysis plan
        if st.button("🎯 Generate Sensitivity Plan", type="primary"):
            if not treatment_var_sens or not outcome_var_sens:
                st.error("Please specify treatment and outcome variables!")
                return
            
            with st.spinner("🎯 Generating sensitivity analysis plan..."):
                try:
                    # Parse observed confounders
                    confounders_list = []
                    if observed_confounders.strip():
                        confounders_list = [line.strip() for line in observed_confounders.split('\n') if line.strip()]
                    
                    # Initialize sensitivity guide
                    llm_client = get_llm_client()
                    sensitivity_guide = LLMSensitivityAnalysisGuide(llm_client)
                    
                    start_time = time.time()
                    
                    # Generate plan
                    plan = asyncio.run(sensitivity_guide.generate_sensitivity_analysis_plan(
                        treatment_variable=treatment_var_sens,
                        outcome_variable=outcome_var_sens,
                        observed_confounders=confounders_list,
                        analysis_context=analysis_context_sens
                    ))
                    
                    analysis_time = time.time() - start_time
                    
                    # Store results
                    st.session_state['sensitivity_plan'] = {
                        'plan': plan,
                        'config': {
                            'treatment_var': treatment_var_sens,
                            'outcome_var': outcome_var_sens,
                            'confounders': confounders_list,
                            'context': analysis_context_sens,
                            'analysis_time': analysis_time
                        },
                        'timestamp': pd.Timestamp.now()
                    }
                    
                    st.success(f"✅ Sensitivity analysis plan generated in {analysis_time:.1f} seconds!")
                    
                except Exception as e:
                    st.error(f"Sensitivity plan generation failed: {str(e)}")
        
        # Display sensitivity plan
        if 'sensitivity_plan' in st.session_state:
            plan_data = st.session_state.sensitivity_plan
            plan = plan_data['plan']
            config = plan_data['config']
            
            st.markdown("#### 🎯 Sensitivity Analysis Plan")
            
            # Plan overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Recommended Tests", len(plan.recommended_tests))
            with col2:
                high_priority = len([t for t in plan.recommended_tests if t.priority.value == "high"])
                st.metric("High Priority Tests", high_priority)
            with col3:
                st.metric("Analysis Context", config['context'].replace('_', ' ').title())
            with col4:
                st.metric("Total Confounders", len(config['confounders']))
            
            # Test recommendations
            if plan.recommended_tests:
                st.markdown("#### 📋 Recommended Sensitivity Tests")
                
                # Group tests by priority
                high_priority_tests = [t for t in plan.recommended_tests if t.priority.value == "high"]
                medium_priority_tests = [t for t in plan.recommended_tests if t.priority.value == "medium"]
                low_priority_tests = [t for t in plan.recommended_tests if t.priority.value == "low"]
                
                for priority, tests in [("High Priority", high_priority_tests), 
                                      ("Medium Priority", medium_priority_tests),
                                      ("Low Priority", low_priority_tests)]:
                    if tests:
                        st.markdown(f"##### 🎯 {priority} Tests")
                        
                        for i, test in enumerate(tests):
                            priority_colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                            
                            with st.expander(f"{priority_colors[test.priority.value]} {test.test_type.value.replace('_', ' ').title()} - {test.priority.value.title()}"):
                                
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.markdown(f"**Description:** {test.description}")
                                    st.markdown(f"**Rationale:** {test.rationale}")
                                    
                                    if test.implementation_steps:
                                        st.markdown("**Implementation Steps:**")
                                        for step in test.implementation_steps:
                                            st.markdown(f"• {step}")
                                
                                with col2:
                                    st.markdown(f"**Method:** {test.method}")
                                    st.markdown(f"**Expected Time:** {test.expected_time_minutes} minutes")
                                    
                                    if test.software_recommendations:
                                        st.markdown("**Software:**")
                                        for software in test.software_recommendations:
                                            st.markdown(f"• {software}")
                                    
                                    if test.interpretation_guidance:
                                        st.markdown(f"**Interpretation:** {test.interpretation_guidance}")
            
            # Implementation timeline
            if hasattr(plan, 'implementation_timeline') and plan.implementation_timeline:
                st.markdown("#### ⏰ Implementation Timeline")
                
                timeline_data = []
                for phase in plan.implementation_timeline:
                    timeline_data.append({
                        'Phase': phase.phase_name,
                        'Tests': ', '.join([t.test_type.value for t in phase.tests]),
                        'Duration': f"{phase.estimated_duration_hours} hours",
                        'Dependencies': ', '.join(phase.dependencies) if phase.dependencies else 'None'
                    })
                
                timeline_df = pd.DataFrame(timeline_data)
                st.dataframe(timeline_df, use_container_width=True)
            
            # Export plan
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Export Plan"):
                    # Create export data
                    export_text = f"# Sensitivity Analysis Plan\n\n"
                    export_text += f"**Treatment:** {config['treatment_var']}\n"
                    export_text += f"**Outcome:** {config['outcome_var']}\n"
                    export_text += f"**Context:** {config['context']}\n\n"
                    
                    for test in plan.recommended_tests:
                        export_text += f"## {test.test_type.value.replace('_', ' ').title()}\n"
                        export_text += f"**Priority:** {test.priority.value.title()}\n"
                        export_text += f"**Description:** {test.description}\n"
                        export_text += f"**Method:** {test.method}\n\n"
                    
                    st.download_button(
                        "Download Plan",
                        export_text,
                        file_name=f"sensitivity_plan_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
            
            with col2:
                if st.button("🔄 Generate Code"):
                    st.info("Code generation feature coming soon!")
    
    with tab4:
        st.markdown("### Effect Size Interpreter")
        st.info("Get intelligent interpretation of statistical effect sizes with domain context")
        
        # Effect size input
        col1, col2 = st.columns(2)
        
        with col1:
            effect_size_value = st.number_input(
                "Effect Size Value",
                value=0.5,
                help="Enter the numerical effect size from your analysis"
            )
            
            effect_size_metric = st.selectbox(
                "Effect Size Metric",
                ["cohen_d", "correlation", "r_squared", "odds_ratio", "risk_ratio", "mean_difference", "standardized_mean_difference"],
                help="Type of effect size measure"
            )
        
        with col2:
            interpretation_domain = st.selectbox(
                "Domain Context",
                ["healthcare", "business", "education", "psychology", "social_science"],
                help="Domain for contextualized interpretation"
            )
            
            audience_level = st.selectbox(
                "Audience Level",
                ["technical", "general", "executive"],
                index=1,
                help="Target audience for the interpretation"
            )
        
        # Additional context
        analysis_context = st.text_area(
            "Analysis Context (Optional)",
            placeholder="e.g., 'Treatment effect on patient recovery in a randomized trial of 500 patients'",
            help="Provide context about your study for more relevant interpretation"
        )
        
        # Interpretation options
        with st.expander("🔧 Interpretation Options"):
            include_benchmarks = st.checkbox("Include Domain Benchmarks", value=True)
            include_confidence_intervals = st.checkbox("Interpret Confidence Intervals", value=True)
            include_practical_significance = st.checkbox("Assess Practical Significance", value=True)
            include_limitations = st.checkbox("Include Limitations", value=True)
        
        # Run interpretation
        if st.button("📏 Interpret Effect Size", type="primary"):
            with st.spinner("📏 Interpreting effect size..."):
                try:
                    # Initialize interpreter
                    llm_client = get_llm_client()
                    interpreter = LLMEffectSizeInterpreter(llm_client)
                    
                    start_time = time.time()
                    
                    # Run interpretation
                    interpretation = asyncio.run(interpreter.interpret_effect_size(
                        effect_size=effect_size_value,
                        effect_type=effect_size_metric,
                        domain=interpretation_domain,
                        context=analysis_context or f"Effect size analysis in {interpretation_domain}"
                    ))
                    
                    interpretation_time = time.time() - start_time
                    
                    # Store results
                    st.session_state['effect_interpretation'] = {
                        'interpretation': interpretation,
                        'config': {
                            'effect_size': effect_size_value,
                            'metric': effect_size_metric,
                            'domain': interpretation_domain,
                            'context': analysis_context,
                            'interpretation_time': interpretation_time
                        },
                        'timestamp': pd.Timestamp.now()
                    }
                    
                    st.success(f"✅ Effect size interpretation completed in {interpretation_time:.1f} seconds!")
                    
                except Exception as e:
                    st.error(f"Effect size interpretation failed: {str(e)}")
        
        # Display interpretation results
        if 'effect_interpretation' in st.session_state:
            interp_data = st.session_state.effect_interpretation
            interpretation = interp_data['interpretation']
            config = interp_data['config']
            
            st.markdown("#### 📏 Effect Size Interpretation")
            
            # Main interpretation
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Effect Size", f"{config['effect_size']:.3f}")
            with col2:
                magnitude = interpretation.magnitude.value.title()
                st.metric("Magnitude", magnitude)
            with col3:
                st.metric("Metric Type", config['metric'].replace('_', ' ').title())
            with col4:
                st.metric("Domain", config['domain'].title())
            
            # Detailed interpretation
            st.markdown("#### 📝 Interpretation Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Main Interpretation:**")
                st.info(interpretation.interpretation)
                
                if interpretation.plain_language_explanation:
                    st.markdown("**Plain Language:**")
                    st.success(interpretation.plain_language_explanation)
            
            with col2:
                if interpretation.domain_specific_interpretation:
                    st.markdown("**Domain-Specific Insights:**")
                    st.info(interpretation.domain_specific_interpretation)
                
                if interpretation.practical_implications:
                    st.markdown("**Practical Implications:**")
                    for implication in interpretation.practical_implications:
                        st.markdown(f"• {implication}")
            
            # Benchmarks and comparisons
            if include_benchmarks and interpretation.domain_benchmarks:
                st.markdown("#### 📊 Domain Benchmarks")
                
                benchmark_data = []
                for benchmark in interpretation.domain_benchmarks:
                    benchmark_data.append({
                        'Context': benchmark.context,
                        'Typical Range': f"{benchmark.typical_range[0]:.2f} - {benchmark.typical_range[1]:.2f}",
                        'Your Effect': f"{config['effect_size']:.3f}",
                        'Comparison': 'Above typical' if config['effect_size'] > benchmark.typical_range[1] else 
                                    'Below typical' if config['effect_size'] < benchmark.typical_range[0] else 'Within typical range'
                    })
                
                if benchmark_data:
                    benchmark_df = pd.DataFrame(benchmark_data)
                    st.dataframe(benchmark_df, use_container_width=True)
            
            # Recommendations
            if interpretation.recommendations:
                st.markdown("#### 💡 Recommendations")
                for rec in interpretation.recommendations:
                    st.info(f"💡 {rec}")
            
            # Limitations
            if include_limitations and interpretation.limitations:
                st.markdown("#### ⚠️ Limitations & Caveats")
                for limitation in interpretation.limitations:
                    st.warning(f"⚠️ {limitation}")
    
    with tab5:
        st.markdown("### Validation Summary")
        st.info("Comprehensive summary of all validation results and recommendations")
        
        # Check what validations have been run
        has_assumptions = 'assumption_results' in st.session_state
        has_arguments = 'argument_validation' in st.session_state
        has_sensitivity = 'sensitivity_plan' in st.session_state
        has_effect_interp = 'effect_interpretation' in st.session_state
        
        if not any([has_assumptions, has_arguments, has_sensitivity, has_effect_interp]):
            st.info("Run validations in the other tabs to see a comprehensive summary here.")
            return
        
        # Validation overview
        st.markdown("#### 📊 Validation Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            assumption_status = "✅ Complete" if has_assumptions else "⏸️ Pending"
            st.metric("Assumption Checking", assumption_status)
        
        with col2:
            argument_status = "✅ Complete" if has_arguments else "⏸️ Pending"
            st.metric("Argument Validation", argument_status)
        
        with col3:
            sensitivity_status = "✅ Complete" if has_sensitivity else "⏸️ Pending"
            st.metric("Sensitivity Planning", sensitivity_status)
        
        with col4:
            effect_status = "✅ Complete" if has_effect_interp else "⏸️ Pending"
            st.metric("Effect Interpretation", effect_status)
        
        # Detailed summaries
        if has_assumptions:
            st.markdown("#### 🔍 Assumption Validation Summary")
            
            assumption_results = st.session_state.assumption_results
            report = assumption_results['report']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Overall Plausibility", f"{report.plausibility_score:.2f}")
            with col2:
                violations = len([v for v in report.assumption_violations if v.severity.value in ["high", "critical"]])
                st.metric("Critical Issues", violations)
            with col3:
                st.metric("Analysis Method", assumption_results['config']['analysis_method'].replace('_', ' ').title())
            
            if violations > 0:
                st.warning(f"⚠️ {violations} critical assumption violations detected. Review detailed results in Assumption Checker tab.")
            else:
                st.success("✅ No critical assumption violations detected.")
        
        if has_arguments:
            st.markdown("#### 📋 Argument Validation Summary")
            
            arg_results = st.session_state.argument_validation
            result = arg_results['result']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Argument Strength", f"{result.overall_score:.1f}/10")
            with col2:
                fallacies = len(result.fallacy_analysis) if hasattr(result, 'fallacy_analysis') else 0
                st.metric("Logical Issues", fallacies)
            with col3:
                st.metric("Domain", arg_results['domain'].title())
            
            if result.overall_score >= 7:
                st.success("✅ Strong causal argument detected.")
            elif result.overall_score >= 5:
                st.warning("⚠️ Moderate argument strength. Consider strengthening evidence.")
            else:
                st.error("❌ Weak argument. Significant improvements needed.")
        
        if has_sensitivity:
            st.markdown("#### 🎯 Sensitivity Analysis Summary")
            
            sens_results = st.session_state.sensitivity_plan
            plan = sens_results['plan']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Recommended Tests", len(plan.recommended_tests))
            with col2:
                high_priority = len([t for t in plan.recommended_tests if t.priority.value == "high"])
                st.metric("High Priority Tests", high_priority)
            with col3:
                st.metric("Analysis Context", sens_results['config']['context'].replace('_', ' ').title())
            
            if high_priority > 0:
                st.warning(f"⚠️ {high_priority} high-priority sensitivity tests recommended.")
            else:
                st.info("💡 Moderate sensitivity testing recommended.")
        
        if has_effect_interp:
            st.markdown("#### 📏 Effect Size Summary")
            
            effect_results = st.session_state.effect_interpretation
            interpretation = effect_results['interpretation']
            config = effect_results['config']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Effect Size", f"{config['effect_size']:.3f}")
            with col2:
                st.metric("Magnitude", interpretation.magnitude.value.title())
            with col3:
                st.metric("Domain", config['domain'].title())
            
            if interpretation.magnitude.value in ['large', 'very_large']:
                st.success("✅ Large effect size detected.")
            elif interpretation.magnitude.value == 'medium':
                st.info("📊 Medium effect size.")
            else:
                st.warning("⚠️ Small effect size. Consider practical significance.")
        
        # Overall recommendations
        st.markdown("#### 💡 Overall Validation Recommendations")
        
        overall_recs = []
        
        # Based on assumption checking
        if has_assumptions:
            report = st.session_state.assumption_results['report']
            if report.plausibility_score < 0.6:
                overall_recs.append("🔴 **Critical**: Address assumption violations before drawing causal conclusions.")
            elif report.plausibility_score < 0.8:
                overall_recs.append("🟡 **Important**: Review assumption violations and consider robustness checks.")
        
        # Based on argument validation
        if has_arguments:
            score = st.session_state.argument_validation['result'].overall_score
            if score < 5:
                overall_recs.append("🔴 **Critical**: Strengthen causal argument with additional evidence.")
            elif score < 7:
                overall_recs.append("🟡 **Important**: Consider additional evidence to strengthen causal claims.")
        
        # Based on sensitivity analysis
        if has_sensitivity:
            high_priority = len([t for t in st.session_state.sensitivity_plan['plan'].recommended_tests 
                               if t.priority.value == "high"])
            if high_priority > 2:
                overall_recs.append("🟡 **Important**: Conduct high-priority sensitivity tests before publication.")
        
        # Based on effect size
        if has_effect_interp:
            magnitude = st.session_state.effect_interpretation['interpretation'].magnitude.value
            if magnitude in ['trivial', 'small']:
                overall_recs.append("🟡 **Consider**: Evaluate practical significance given small effect size.")
        
        # General recommendations
        if not overall_recs:
            overall_recs.append("✅ **Good**: Validation results look reasonable. Consider peer review.")
        
        overall_recs.append("📚 **Always**: Document validation process and limitations in your report.")
        overall_recs.append("🔄 **Consider**: Replication and external validation of findings.")
        
        for rec in overall_recs:
            if rec.startswith("🔴"):
                st.error(rec)
            elif rec.startswith("🟡"):
                st.warning(rec)
            else:
                st.info(rec)
        
        # Export comprehensive report
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Generate Full Report"):
                st.info("Comprehensive report generation coming soon!")
        
        with col2:
            if st.button("📊 Export Summary"):
                # Create summary data for export
                summary_data = {
                    'timestamp': pd.Timestamp.now(),
                    'assumption_plausibility': st.session_state.assumption_results['report'].plausibility_score if has_assumptions else None,
                    'argument_strength': st.session_state.argument_validation['result'].overall_score if has_arguments else None,
                    'sensitivity_tests': len(st.session_state.sensitivity_plan['plan'].recommended_tests) if has_sensitivity else None,
                    'effect_magnitude': st.session_state.effect_interpretation['interpretation'].magnitude.value if has_effect_interp else None
                }
                
                st.json(summary_data)
        
        with col3:
            if st.button("🔄 Save Validation Session"):
                # Save all validation results to session history
                if 'validation_history' not in st.session_state:
                    st.session_state.validation_history = []
                
                validation_session = {
                    'timestamp': pd.Timestamp.now(),
                    'has_assumptions': has_assumptions,
                    'has_arguments': has_arguments,
                    'has_sensitivity': has_sensitivity,
                    'has_effect_interp': has_effect_interp,
                    'dataset': st.session_state.get('data_source', 'Unknown')
                }
                
                st.session_state.validation_history.append(validation_session)
                st.success("✅ Validation session saved!")

    # Update session statistics
    validation_count = sum([
        'assumption_results' in st.session_state,
        'argument_validation' in st.session_state,
        'sensitivity_plan' in st.session_state,
        'effect_interpretation' in st.session_state
    ])
    
    if validation_count > 0:
        if 'session_stats' not in st.session_state:
            st.session_state.session_stats = {'analyses': 0, 'datasets': 0, 'success_rate': 0}
        
        # Update analyses count with validation results
        current_analyses = st.session_state.session_stats.get('analyses', 0)
        st.session_state.session_stats['analyses'] = max(current_analyses, validation_count)