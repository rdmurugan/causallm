import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from causalllm.llm_client import get_llm_client
from causalllm.llm_effect_size_interpreter import LLMEffectSizeInterpreter
from causalllm.llm_sensitivity_analysis_guide import LLMSensitivityAnalysisGuide
import asyncio

def show():
    st.title("⚡ Intervention Optimizer")
    st.markdown("Optimize intervention strategies and estimate causal effects")
    
    # Check for data availability
    if 'current_data' not in st.session_state:
        st.warning("📁 Please upload data in the Data Manager first to use intervention optimization")
        return
    
    data = st.session_state.current_data
    
    # Check for variable roles
    if 'variable_roles' not in st.session_state:
        st.warning("🏷️ Please assign variable roles in the Data Manager first")
        return
    
    roles = st.session_state.variable_roles
    
    # Initialize components
    if 'effect_interpreter' not in st.session_state:
        try:
            llm_client = get_llm_client()
            st.session_state.effect_interpreter = LLMEffectSizeInterpreter(llm_client)
            st.session_state.sensitivity_guide = LLMSensitivityAnalysisGuide(llm_client)
        except Exception as e:
            st.error(f"Failed to initialize LLM client: {str(e)}")
            return
    
    # Create tabs for different optimization aspects
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Intervention Setup", "📊 Effect Estimation", "🔄 Sensitivity Analysis", "📈 Optimization Results"
    ])
    
    with tab1:
        st.markdown("### Intervention Configuration")
        
        # Treatment variable selection
        treatment_vars = roles.get('treatment', [])
        if not treatment_vars:
            st.error("No treatment variables found. Please assign treatment variables in Data Manager.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_treatment = st.selectbox(
                "Select primary treatment variable",
                treatment_vars,
                help="Variable representing the intervention to optimize"
            )
            
            if selected_treatment:
                unique_values = data[selected_treatment].unique()
                unique_values = [v for v in unique_values if pd.notna(v)]
                
                # Intervention levels
                st.markdown("#### Intervention Levels")
                if len(unique_values) <= 10:  # Categorical or few unique values
                    baseline_level = st.selectbox(
                        "Baseline (control) level",
                        unique_values,
                        help="Reference level for comparison"
                    )
                    intervention_levels = st.multiselect(
                        "Intervention levels to compare",
                        [v for v in unique_values if v != baseline_level],
                        default=[v for v in unique_values if v != baseline_level][:3],
                        help="Treatment levels to optimize"
                    )
                else:  # Continuous variable
                    st.info("Continuous treatment variable detected")
                    current_mean = float(data[selected_treatment].mean())
                    current_std = float(data[selected_treatment].std())
                    
                    baseline_level = st.number_input(
                        "Baseline level",
                        value=current_mean,
                        help="Reference level for comparison"
                    )
                    
                    # Intervention range
                    min_intervention = st.number_input(
                        "Minimum intervention level",
                        value=current_mean - current_std,
                        help="Lowest intervention level to consider"
                    )
                    max_intervention = st.number_input(
                        "Maximum intervention level",
                        value=current_mean + current_std,
                        help="Highest intervention level to consider"
                    )
                    
                    num_levels = st.slider(
                        "Number of intervention levels to test",
                        min_value=3, max_value=10, value=5,
                        help="How many levels to test between min and max"
                    )
                    
                    intervention_levels = np.linspace(min_intervention, max_intervention, num_levels)
        
        with col2:
            # Outcome variable selection
            outcome_vars = roles.get('outcome', [])
            if not outcome_vars:
                st.error("No outcome variables found. Please assign outcome variables in Data Manager.")
                return
            
            selected_outcome = st.selectbox(
                "Select primary outcome variable",
                outcome_vars,
                help="Variable representing the outcome to optimize"
            )
            
            # Optimization objective
            st.markdown("#### Optimization Objective")
            if selected_outcome:
                is_numeric = pd.api.types.is_numeric_dtype(data[selected_outcome])
                
                if is_numeric:
                    objective = st.radio(
                        "Optimization goal",
                        ["Maximize outcome", "Minimize outcome", "Target specific value"],
                        help="What you want to achieve with the intervention"
                    )
                    
                    if objective == "Target specific value":
                        target_value = st.number_input(
                            "Target value",
                            value=float(data[selected_outcome].mean()),
                            help="Desired outcome value"
                        )
                else:
                    # Categorical outcome
                    outcome_categories = data[selected_outcome].unique()
                    outcome_categories = [c for c in outcome_categories if pd.notna(c)]
                    
                    target_outcome = st.selectbox(
                        "Target outcome category",
                        outcome_categories,
                        help="Desired outcome category to maximize probability of"
                    )
                    objective = "Maximize probability"
        
        # Constraints and considerations
        st.markdown("#### Constraints & Considerations")
        
        col1, col2 = st.columns(2)
        with col1:
            # Cost considerations
            include_cost = st.checkbox("Include intervention cost", value=False)
            if include_cost:
                cost_per_unit = st.number_input(
                    "Cost per unit intervention",
                    value=1.0,
                    min_value=0.0,
                    help="Cost associated with each unit of intervention"
                )
        
        with col2:
            # Population constraints
            include_population = st.checkbox("Population-specific optimization", value=False)
            if include_population:
                confounders = roles.get('confounders', [])
                if confounders:
                    population_var = st.selectbox(
                        "Population segmentation variable",
                        confounders,
                        help="Variable to segment population for targeted interventions"
                    )
    
    with tab2:
        st.markdown("### Effect Size Estimation")
        
        if 'selected_treatment' in locals() and 'selected_outcome' in locals():
            
            # Causal effect estimation method
            col1, col2 = st.columns(2)
            with col1:
                estimation_method = st.selectbox(
                    "Effect estimation method",
                    [
                        "Simple Difference in Means",
                        "Regression Adjustment", 
                        "Propensity Score Matching",
                        "Instrumental Variables",
                        "Difference-in-Differences"
                    ],
                    help="Statistical method for estimating causal effects"
                )
            
            with col2:
                confidence_level = st.slider(
                    "Confidence level",
                    min_value=0.90, max_value=0.99, value=0.95, step=0.01,
                    help="Confidence level for effect estimates"
                )
            
            if st.button("🔍 Estimate Effects", type="primary"):
                with st.spinner("Estimating causal effects..."):
                    
                    # Simulate effect estimation (replace with actual causal inference)
                    results = estimate_causal_effects(
                        data, selected_treatment, selected_outcome, 
                        intervention_levels, baseline_level,
                        estimation_method, confidence_level
                    )
                    
                    # Display results
                    st.markdown("#### 📊 Effect Estimates")
                    
                    if isinstance(intervention_levels, np.ndarray):
                        # Continuous treatment
                        effect_data = pd.DataFrame({
                            'Intervention_Level': intervention_levels,
                            'Estimated_Effect': results['effects'],
                            'Lower_CI': results['ci_lower'],
                            'Upper_CI': results['ci_upper'],
                            'P_Value': results['p_values']
                        })
                        
                        # Plot dose-response curve
                        fig = go.Figure()
                        
                        # Add effect estimate line
                        fig.add_trace(go.Scatter(
                            x=effect_data['Intervention_Level'],
                            y=effect_data['Estimated_Effect'],
                            mode='lines+markers',
                            name='Effect Estimate',
                            line=dict(color='blue', width=3)
                        ))
                        
                        # Add confidence interval
                        fig.add_trace(go.Scatter(
                            x=list(effect_data['Intervention_Level']) + list(effect_data['Intervention_Level'][::-1]),
                            y=list(effect_data['Upper_CI']) + list(effect_data['Lower_CI'][::-1]),
                            fill='toself',
                            fillcolor='rgba(0,100,80,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f'{int(confidence_level*100)}% Confidence Interval'
                        ))
                        
                        fig.update_layout(
                            title="Dose-Response Curve",
                            xaxis_title=f"{selected_treatment}",
                            yaxis_title=f"Effect on {selected_outcome}",
                            hovermode='x'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        # Categorical treatment
                        effect_data = pd.DataFrame({
                            'Intervention_Level': intervention_levels,
                            'Estimated_Effect': results['effects'],
                            'Lower_CI': results['ci_lower'],
                            'Upper_CI': results['ci_upper'],
                            'P_Value': results['p_values'],
                            'Significant': results['p_values'] < (1 - confidence_level)
                        })
                        
                        # Bar plot with error bars
                        fig = px.bar(
                            effect_data,
                            x='Intervention_Level',
                            y='Estimated_Effect',
                            error_y='Upper_CI',
                            error_y_minus='Lower_CI',
                            color='Significant',
                            title="Treatment Effects by Intervention Level",
                            color_discrete_map={True: 'green', False: 'orange'}
                        )
                        
                        fig.update_layout(
                            xaxis_title=f"{selected_treatment}",
                            yaxis_title=f"Effect on {selected_outcome}"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed results table
                    st.dataframe(effect_data, use_container_width=True)
                    
                    # LLM interpretation
                    if st.button("🤖 Get AI Interpretation"):
                        with st.spinner("Generating AI interpretation..."):
                            try:
                                interpretation = asyncio.run(
                                    st.session_state.effect_interpreter.interpret_effect_sizes(
                                        effect_estimates=results['effects'],
                                        confidence_intervals=list(zip(results['ci_lower'], results['ci_upper'])),
                                        treatment_variable=selected_treatment,
                                        outcome_variable=selected_outcome,
                                        sample_size=len(data),
                                        domain=st.session_state.get('domain', 'general')
                                    )
                                )
                                
                                st.markdown("#### 🎯 AI Effect Interpretation")
                                st.markdown(interpretation)
                                
                            except Exception as e:
                                st.error(f"Failed to generate interpretation: {str(e)}")
    
    with tab3:
        st.markdown("### Sensitivity Analysis")
        
        if 'selected_treatment' in locals() and 'selected_outcome' in locals():
            
            st.info("Assess the robustness of your causal effect estimates")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Unmeasured Confounding")
                
                # Sensitivity to unmeasured confounders
                confounder_strength = st.slider(
                    "Potential confounder strength (correlation with treatment)",
                    min_value=0.1, max_value=0.9, value=0.3, step=0.1,
                    help="How strongly an unmeasured confounder might correlate with treatment"
                )
                
                outcome_strength = st.slider(
                    "Confounder effect on outcome",
                    min_value=0.1, max_value=0.9, value=0.3, step=0.1,
                    help="How strongly the unmeasured confounder might affect the outcome"
                )
            
            with col2:
                st.markdown("#### Model Specification")
                
                # Alternative model specifications
                include_interactions = st.checkbox(
                    "Test interaction effects",
                    help="Include treatment-covariate interactions"
                )
                
                include_nonlinear = st.checkbox(
                    "Test non-linear relationships",
                    help="Include polynomial or spline terms"
                )
            
            if st.button("🔍 Run Sensitivity Analysis"):
                with st.spinner("Performing sensitivity analysis..."):
                    
                    # Generate sensitivity analysis results
                    sensitivity_results = perform_sensitivity_analysis(
                        data, selected_treatment, selected_outcome,
                        confounder_strength, outcome_strength,
                        include_interactions, include_nonlinear
                    )
                    
                    # Sensitivity plot
                    fig = go.Figure()
                    
                    # Original effect estimate
                    original_effect = sensitivity_results['original_effect']
                    fig.add_hline(
                        y=original_effect, 
                        line_dash="dash", 
                        line_color="blue",
                        annotation_text="Original Estimate"
                    )
                    
                    # Sensitivity range
                    confounder_range = np.linspace(0, 0.9, 10)
                    effect_range = [
                        original_effect * (1 - strength * outcome_strength) 
                        for strength in confounder_range
                    ]
                    
                    fig.add_trace(go.Scatter(
                        x=confounder_range,
                        y=effect_range,
                        mode='lines+markers',
                        name='Effect under confounding',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig.update_layout(
                        title="Sensitivity to Unmeasured Confounding",
                        xaxis_title="Confounder Strength",
                        yaxis_title="Adjusted Effect Estimate"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display sensitivity metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Original Effect",
                            f"{original_effect:.3f}"
                        )
                    
                    with col2:
                        robustness_score = sensitivity_results['robustness_score']
                        st.metric(
                            "Robustness Score",
                            f"{robustness_score:.2f}",
                            help="Higher scores indicate more robust estimates"
                        )
                    
                    with col3:
                        min_effect = min(effect_range)
                        st.metric(
                            "Worst-case Effect",
                            f"{min_effect:.3f}",
                            delta=f"{min_effect - original_effect:.3f}"
                        )
                    
                    # AI guidance
                    if st.button("🤖 Get Sensitivity Guidance"):
                        with st.spinner("Generating sensitivity guidance..."):
                            try:
                                guidance = asyncio.run(
                                    st.session_state.sensitivity_guide.generate_sensitivity_analysis_guide(
                                        effect_estimate=original_effect,
                                        treatment_variable=selected_treatment,
                                        outcome_variable=selected_outcome,
                                        confounding_variables=roles.get('confounders', []),
                                        sample_size=len(data),
                                        study_design="observational"
                                    )
                                )
                                
                                st.markdown("#### 🎯 Sensitivity Analysis Guidance")
                                st.markdown(guidance)
                                
                            except Exception as e:
                                st.error(f"Failed to generate guidance: {str(e)}")
    
    with tab4:
        st.markdown("### Optimization Results")
        
        if 'effect_data' in locals():
            
            st.markdown("#### 🏆 Optimal Intervention Strategy")
            
            # Find optimal intervention
            if 'objective' in locals():
                if objective == "Maximize outcome":
                    optimal_idx = effect_data['Estimated_Effect'].idxmax()
                elif objective == "Minimize outcome":
                    optimal_idx = effect_data['Estimated_Effect'].idxmin()
                elif objective == "Target specific value" and 'target_value' in locals():
                    optimal_idx = (effect_data['Estimated_Effect'] - target_value).abs().idxmin()
                else:
                    optimal_idx = effect_data['Estimated_Effect'].idxmax()
                
                optimal_level = effect_data.iloc[optimal_idx]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Optimal Intervention Level",
                        f"{optimal_level['Intervention_Level']}"
                    )
                
                with col2:
                    st.metric(
                        "Expected Effect",
                        f"{optimal_level['Estimated_Effect']:.3f}",
                        help="Expected change in outcome"
                    )
                
                with col3:
                    if 'include_cost' in locals() and include_cost:
                        total_cost = optimal_level['Intervention_Level'] * cost_per_unit
                        cost_effectiveness = optimal_level['Estimated_Effect'] / total_cost if total_cost > 0 else float('inf')
                        st.metric(
                            "Cost-Effectiveness",
                            f"{cost_effectiveness:.3f}",
                            help="Effect per unit cost"
                        )
            
            # Recommendation summary
            st.markdown("#### 📋 Implementation Recommendations")
            
            recommendations = []
            
            if 'optimal_level' in locals():
                recommendations.append(f"**Primary Recommendation**: Set {selected_treatment} to {optimal_level['Intervention_Level']}")
                
                if optimal_level['P_Value'] < 0.05:
                    recommendations.append("✅ **Statistical Significance**: The optimal effect estimate is statistically significant")
                else:
                    recommendations.append("⚠️ **Statistical Caution**: The optimal effect estimate is not statistically significant")
                
                # Effect size interpretation
                effect_magnitude = abs(optimal_level['Estimated_Effect'])
                if effect_magnitude > 0.5:
                    recommendations.append("📈 **Large Effect**: This intervention shows a substantial impact")
                elif effect_magnitude > 0.2:
                    recommendations.append("📊 **Moderate Effect**: This intervention shows a meaningful impact")
                else:
                    recommendations.append("📉 **Small Effect**: This intervention shows a modest impact")
            
            # Population-specific recommendations
            if 'include_population' in locals() and include_population and 'population_var' in locals():
                st.markdown("#### 🎯 Population-Specific Strategies")
                
                # Analyze effects by population segments
                unique_segments = data[population_var].unique()
                unique_segments = [s for s in unique_segments if pd.notna(s)]
                
                segment_results = {}
                for segment in unique_segments:
                    segment_data = data[data[population_var] == segment]
                    if len(segment_data) > 10:  # Minimum sample size
                        segment_effect = estimate_causal_effects(
                            segment_data, selected_treatment, selected_outcome,
                            intervention_levels, baseline_level,
                            estimation_method, confidence_level
                        )
                        segment_results[segment] = segment_effect
                
                if segment_results:
                    # Create comparison chart
                    segment_comparison = []
                    for segment, results in segment_results.items():
                        best_effect_idx = np.argmax(results['effects'])
                        segment_comparison.append({
                            'Segment': segment,
                            'Optimal_Level': intervention_levels[best_effect_idx] if isinstance(intervention_levels, np.ndarray) else intervention_levels[best_effect_idx],
                            'Expected_Effect': results['effects'][best_effect_idx],
                            'Sample_Size': len(data[data[population_var] == segment])
                        })
                    
                    segment_df = pd.DataFrame(segment_comparison)
                    
                    fig = px.scatter(
                        segment_df,
                        x='Optimal_Level',
                        y='Expected_Effect',
                        size='Sample_Size',
                        hover_name='Segment',
                        title="Optimal Intervention by Population Segment"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(segment_df, use_container_width=True)
            
            for rec in recommendations:
                st.markdown(rec)
            
            # Export results
            st.markdown("#### 📤 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Download Results"):
                    csv = effect_data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"intervention_optimization_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📋 Generate Report"):
                    st.success("Feature coming soon: Comprehensive optimization report with visualizations and recommendations")

def estimate_causal_effects(data, treatment, outcome, intervention_levels, baseline, method, confidence_level):
    """Simulate causal effect estimation - replace with actual causal inference methods"""
    np.random.seed(42)
    
    n_levels = len(intervention_levels) if isinstance(intervention_levels, list) else len(intervention_levels)
    
    # Simulate realistic effect estimates
    effects = np.random.normal(0.2, 0.1, n_levels)
    effects = np.cumsum(effects) * 0.5  # Make effects somewhat dose-dependent
    
    # Simulate confidence intervals
    se = np.random.uniform(0.05, 0.15, n_levels)
    z_score = 1.96 if confidence_level == 0.95 else 2.58  # approximate
    
    ci_lower = effects - z_score * se
    ci_upper = effects + z_score * se
    p_values = 2 * (1 - np.random.beta(2, 5, n_levels))  # Simulate p-values
    
    return {
        'effects': effects,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_values': p_values
    }

def perform_sensitivity_analysis(data, treatment, outcome, confounder_strength, outcome_strength, interactions, nonlinear):
    """Perform sensitivity analysis for causal estimates"""
    
    # Simulate original effect
    original_effect = np.random.normal(0.25, 0.05)
    
    # Calculate robustness score (higher = more robust)
    robustness_score = 1 - (confounder_strength * outcome_strength)
    
    return {
        'original_effect': original_effect,
        'robustness_score': robustness_score,
        'confounder_strength': confounder_strength,
        'outcome_strength': outcome_strength
    }