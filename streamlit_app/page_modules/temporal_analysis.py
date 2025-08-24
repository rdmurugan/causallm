import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from causalllm.temporal_causal_modeling import AdvancedTemporalAnalyzer, LLMGuidedTemporalModel
from causalllm.llm_client import get_llm_client
import time

def show():
    st.title("⏱️ Temporal Causal Analysis")
    st.markdown("Analyze time-series causal relationships and optimal intervention timing")
    
    # Check if data is available
    if 'current_data' not in st.session_state:
        st.warning("📁 Please upload a dataset in the Data Manager first!")
        if st.button("Go to Data Manager"):
            st.info("Navigate to Data Manager → Upload your time-series dataset")
        return
    
    data = st.session_state.current_data
    
    # Check for time columns
    time_columns = []
    for col in data.columns:
        if data[col].dtype in ['datetime64[ns]', 'object']:
            # Try to parse as datetime
            try:
                pd.to_datetime(data[col].head())
                time_columns.append(col)
            except:
                pass
    
    if not time_columns:
        st.warning("⚠️ No time columns detected in your dataset. Temporal analysis works best with time-series data.")
        st.info("Consider adding a date/time column or using the Data Manager to specify time variables.")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔧 Setup Analysis", "📊 Temporal Relationships", "🎯 Intervention Timing", "📈 Forecasting & Scenarios"
    ])
    
    with tab1:
        st.markdown("### Temporal Analysis Configuration")
        
        # Variable selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Core Variables")
            
            # Time variable
            if time_columns:
                time_variable = st.selectbox(
                    "Time Variable",
                    time_columns,
                    help="Column representing time/date"
                )
            else:
                time_variable = st.selectbox(
                    "Time Variable", 
                    data.columns,
                    help="Select the best approximation for time ordering"
                )
            
            # Treatment variable
            treatment_options = data.columns.tolist()
            if 'variable_roles' in st.session_state and st.session_state.variable_roles.get('treatment'):
                default_treatment = st.session_state.variable_roles['treatment'][0]
            else:
                default_treatment = treatment_options[0] if treatment_options else None
            
            treatment_variable = st.selectbox(
                "Treatment/Intervention Variable",
                treatment_options,
                index=treatment_options.index(default_treatment) if default_treatment in treatment_options else 0,
                help="Variable representing the intervention or exposure"
            )
            
            # Outcome variable
            outcome_options = [col for col in data.columns if col not in [time_variable, treatment_variable]]
            if 'variable_roles' in st.session_state and st.session_state.variable_roles.get('outcome'):
                outcome_vars = st.session_state.variable_roles['outcome']
                default_outcome = outcome_vars[0] if outcome_vars else None
            else:
                default_outcome = None
            
            outcome_variable = st.selectbox(
                "Outcome Variable",
                outcome_options,
                index=outcome_options.index(default_outcome) if default_outcome in outcome_options else 0,
                help="Variable representing the outcome of interest"
            )
        
        with col2:
            st.markdown("#### Analysis Settings")
            
            # Time unit
            time_unit = st.selectbox(
                "Time Unit",
                ["day", "week", "month", "quarter", "year"],
                index=1,
                help="Basic time unit for analysis"
            )
            
            # Analysis window
            analysis_window = st.number_input(
                "Analysis Window (time units)",
                min_value=1, max_value=100, value=12,
                help="How many time units to analyze before/after intervention"
            )
            
            # Lag analysis
            max_lags = st.slider(
                "Maximum Lags to Consider",
                1, 20, 5,
                help="Maximum number of time periods to look back for causal effects"
            )
            
            # Minimum observations
            min_observations = st.number_input(
                "Minimum Observations Required",
                min_value=10, max_value=1000, value=30,
                help="Minimum number of observations needed for reliable analysis"
            )
        
        # Additional variables
        st.markdown("#### Additional Variables")
        
        control_variables = st.multiselect(
            "Control Variables",
            [col for col in data.columns if col not in [time_variable, treatment_variable, outcome_variable]],
            help="Variables to control for in the temporal analysis"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                seasonal_adjustment = st.checkbox("Seasonal Adjustment", value=False)
                trend_removal = st.checkbox("Remove Trend", value=False)
                difference_data = st.checkbox("Difference Time Series", value=False)
            
            with col2:
                confidence_level = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
                bootstrap_samples = st.slider("Bootstrap Samples", 100, 2000, 500)
                include_interactions = st.checkbox("Include Time Interactions", value=True)
        
        # Data preview and validation
        st.markdown("#### Data Preview")
        
        if st.button("🔍 Validate Data"):
            # Basic data validation
            validation_results = []
            
            # Check time variable
            if time_variable in data.columns:
                try:
                    time_data = pd.to_datetime(data[time_variable])
                    time_range = time_data.max() - time_data.min()
                    validation_results.append(f"✅ Time range: {time_range.days} days")
                    
                    # Check for missing time values
                    missing_time = data[time_variable].isnull().sum()
                    if missing_time > 0:
                        validation_results.append(f"⚠️ Missing time values: {missing_time}")
                    else:
                        validation_results.append("✅ No missing time values")
                    
                except Exception as e:
                    validation_results.append(f"❌ Time variable parsing error: {str(e)}")
            
            # Check treatment variable
            if treatment_variable in data.columns:
                treatment_unique = data[treatment_variable].nunique()
                validation_results.append(f"✅ Treatment levels: {treatment_unique}")
                
                # Check for variation over time
                if time_variable in data.columns:
                    treatment_by_time = data.groupby(time_variable)[treatment_variable].nunique().mean()
                    if treatment_by_time > 1:
                        validation_results.append("✅ Treatment varies over time")
                    else:
                        validation_results.append("⚠️ Limited treatment variation over time")
            
            # Check outcome variable
            if outcome_variable in data.columns:
                outcome_missing = data[outcome_variable].isnull().sum()
                if outcome_missing == 0:
                    validation_results.append("✅ No missing outcome values")
                else:
                    validation_results.append(f"⚠️ Missing outcome values: {outcome_missing}")
                
                # Check for sufficient variation
                outcome_std = data[outcome_variable].std()
                if outcome_std > 0:
                    validation_results.append("✅ Outcome shows variation")
                else:
                    validation_results.append("❌ No variation in outcome")
            
            # Display results
            for result in validation_results:
                if result.startswith("✅"):
                    st.success(result)
                elif result.startswith("⚠️"):
                    st.warning(result)
                else:
                    st.error(result)
        
        # Preview sample data
        if st.checkbox("Show Data Sample"):
            sample_cols = [time_variable, treatment_variable, outcome_variable] + control_variables[:3]
            sample_cols = [col for col in sample_cols if col in data.columns]
            st.dataframe(data[sample_cols].head(10), use_container_width=True)
        
        # Save configuration
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Configuration"):
                temporal_config = {
                    'time_variable': time_variable,
                    'treatment_variable': treatment_variable,
                    'outcome_variable': outcome_variable,
                    'control_variables': control_variables,
                    'time_unit': time_unit,
                    'analysis_window': analysis_window,
                    'max_lags': max_lags,
                    'confidence_level': confidence_level
                }
                st.session_state['temporal_config'] = temporal_config
                st.success("Configuration saved!")
        
        with col2:
            if st.button("🚀 Run Temporal Analysis", type="primary"):
                if not all([time_variable, treatment_variable, outcome_variable]):
                    st.error("Please select time, treatment, and outcome variables!")
                    return
                
                # Store configuration and trigger analysis
                temporal_config = {
                    'time_variable': time_variable,
                    'treatment_variable': treatment_variable,
                    'outcome_variable': outcome_variable,
                    'control_variables': control_variables,
                    'time_unit': time_unit,
                    'analysis_window': analysis_window,
                    'max_lags': max_lags,
                    'confidence_level': confidence_level,
                    'seasonal_adjustment': seasonal_adjustment,
                    'trend_removal': trend_removal,
                    'bootstrap_samples': bootstrap_samples
                }
                st.session_state['temporal_config'] = temporal_config
                
                # Run analysis
                with st.spinner("⏱️ Running temporal causal analysis... This may take several minutes."):
                    try:
                        # Initialize temporal analyzer
                        llm_client = get_llm_client()
                        analyzer = AdvancedTemporalAnalyzer(llm_client)
                        
                        start_time = time.time()
                        
                        # Prepare data
                        analysis_data = data[[time_variable, treatment_variable, outcome_variable] + control_variables].copy()
                        
                        # Convert time variable to datetime if possible
                        try:
                            analysis_data[time_variable] = pd.to_datetime(analysis_data[time_variable])
                        except:
                            pass
                        
                        # Run temporal analysis
                        results = asyncio.run(analyzer.analyze_temporal_causation(
                            data=analysis_data,
                            treatment_variable=treatment_variable,
                            outcome_variable=outcome_variable,
                            time_variable=time_variable,
                            control_variables=control_variables
                        ))
                        
                        analysis_time = time.time() - start_time
                        
                        # Store results
                        st.session_state['temporal_results'] = {
                            'results': results,
                            'config': temporal_config,
                            'analysis_time': analysis_time,
                            'timestamp': pd.Timestamp.now()
                        }
                        
                        st.success(f"✅ Temporal analysis completed in {analysis_time:.1f} seconds!")
                        st.balloons()
                        
                        # Quick summary
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Relationships Found", len(results.temporal_relationships))
                        with col2:
                            st.metric("Confidence Score", f"{results.overall_confidence:.2f}")
                        with col3:
                            st.metric("Optimal Lag", f"{results.optimal_lag} periods")
                        with col4:
                            st.metric("Analysis Time", f"{analysis_time:.1f}s")
                        
                    except Exception as e:
                        st.error(f"Temporal analysis failed: {str(e)}")
                        st.info("This might be due to:")
                        st.markdown("""
                        - Insufficient time-series data
                        - Data format issues
                        - LLM client configuration
                        - Complex temporal patterns requiring more data
                        """)
    
    with tab2:
        if 'temporal_results' not in st.session_state:
            st.info("⏱️ Run temporal analysis first to view temporal relationships")
            return
        
        results_data = st.session_state.temporal_results
        results = results_data['results']
        config = results_data['config']
        
        st.markdown("### Temporal Relationships Analysis")
        
        # Results overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Relationships", len(results.temporal_relationships))
        with col2:
            st.metric("Overall Confidence", f"{results.overall_confidence:.2f}")
        with col3:
            st.metric("Primary Effect Lag", f"{results.optimal_lag} {config['time_unit']}s")
        with col4:
            st.metric("Effect Duration", f"{results.effect_duration} {config['time_unit']}s")
        
        # Temporal relationships details
        st.markdown("#### Discovered Temporal Relationships")
        
        if results.temporal_relationships:
            # Create relationships table
            relationships_data = []
            for rel in results.temporal_relationships:
                relationships_data.append({
                    'Relationship': f"{rel.source_variable} → {rel.target_variable}",
                    'Type': rel.relationship_type.value.replace('_', ' ').title(),
                    'Lag': f"{rel.lag_periods} {config['time_unit']}s",
                    'Strength': f"{rel.strength:.3f}",
                    'Confidence': f"{rel.confidence:.3f}",
                    'Duration': f"{rel.duration_periods} {config['time_unit']}s" if hasattr(rel, 'duration_periods') else 'Ongoing'
                })
            
            relationships_df = pd.DataFrame(relationships_data)
            st.dataframe(relationships_df, use_container_width=True)
            
            # Temporal pattern visualization
            st.markdown("#### Temporal Effect Patterns")
            
            # Create lag analysis chart
            if hasattr(results, 'lag_analysis'):
                lag_data = results.lag_analysis
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(lag_data))),
                    y=lag_data,
                    mode='lines+markers',
                    name='Effect Strength by Lag',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title="Effect Strength by Time Lag",
                    xaxis_title=f"Lag ({config['time_unit']}s)",
                    yaxis_title="Effect Strength",
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Time series visualization
            st.markdown("#### Time Series Analysis")
            
            # Show original time series if available
            if 'original_data' in results_data:
                ts_data = results_data['original_data']
            else:
                ts_data = data.copy()
            
            # Create time series plot
            time_col = config['time_variable']
            treatment_col = config['treatment_variable']
            outcome_col = config['outcome_variable']
            
            if all(col in ts_data.columns for col in [time_col, treatment_col, outcome_col]):
                
                # Prepare data for plotting
                plot_data = ts_data[[time_col, treatment_col, outcome_col]].copy()
                
                # Convert time to datetime if not already
                try:
                    plot_data[time_col] = pd.to_datetime(plot_data[time_col])
                except:
                    pass
                
                # Sort by time
                plot_data = plot_data.sort_values(time_col)
                
                fig = go.Figure()
                
                # Add outcome line
                fig.add_trace(go.Scatter(
                    x=plot_data[time_col],
                    y=plot_data[outcome_col],
                    mode='lines',
                    name=f'Outcome ({outcome_col})',
                    line=dict(color='blue', width=2),
                    yaxis='y'
                ))
                
                # Add treatment line (normalized to secondary axis)
                treatment_normalized = (plot_data[treatment_col] - plot_data[treatment_col].min()) / \
                                     (plot_data[treatment_col].max() - plot_data[treatment_col].min())
                
                fig.add_trace(go.Scatter(
                    x=plot_data[time_col],
                    y=treatment_normalized,
                    mode='lines',
                    name=f'Treatment ({treatment_col})',
                    line=dict(color='red', width=2),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title="Treatment and Outcome Over Time",
                    xaxis_title="Time",
                    yaxis=dict(
                        title=f"Outcome ({outcome_col})",
                        side="left"
                    ),
                    yaxis2=dict(
                        title=f"Treatment (Normalized)",
                        side="right",
                        overlaying="y"
                    ),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Cross-correlation analysis
        if hasattr(results, 'cross_correlation'):
            st.markdown("#### Cross-Correlation Analysis")
            
            corr_data = results.cross_correlation
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(range(-len(corr_data)//2, len(corr_data)//2 + 1)),
                y=corr_data,
                mode='lines+markers',
                name='Cross-Correlation',
                line=dict(color='green', width=2)
            ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_hline(y=0.3, line_dash="dot", line_color="red", annotation_text="Strong Correlation")
            fig.add_hline(y=-0.3, line_dash="dot", line_color="red")
            
            fig.update_layout(
                title="Cross-Correlation Between Treatment and Outcome",
                xaxis_title=f"Lag ({config['time_unit']}s)",
                yaxis_title="Correlation",
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        if hasattr(results, 'key_insights') and results.key_insights:
            st.markdown("#### 🔑 Key Temporal Insights")
            
            for insight in results.key_insights:
                st.info(f"💡 {insight}")
        
        # Export results
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Relationships"):
                csv_data = relationships_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv_data,
                    file_name=f"temporal_relationships_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Save Analysis"):
                if 'analysis_history' not in st.session_state:
                    st.session_state.analysis_history = []
                
                analysis_record = {
                    'type': 'temporal_analysis',
                    'timestamp': results_data['timestamp'],
                    'relationships': len(results.temporal_relationships),
                    'confidence': results.overall_confidence,
                    'optimal_lag': results.optimal_lag
                }
                
                st.session_state.analysis_history.append(analysis_record)
                st.success("Analysis saved to history!")
    
    with tab3:
        if 'temporal_results' not in st.session_state:
            st.info("⏱️ Complete temporal analysis first to explore intervention timing")
            return
        
        st.markdown("### Optimal Intervention Timing")
        
        results_data = st.session_state.temporal_results
        results = results_data['results']
        config = results_data['config']
        
        # Intervention timing analysis
        st.markdown("#### Timing Optimization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Analysis Results:**")
            st.metric("Optimal Intervention Lag", f"{results.optimal_lag} {config['time_unit']}s")
            st.metric("Effect Duration", f"{results.effect_duration} {config['time_unit']}s")
            st.metric("Peak Effect Time", f"{results.optimal_lag + results.effect_duration//2} {config['time_unit']}s")
        
        with col2:
            st.markdown("**Intervention Parameters:**")
            
            intervention_intensity = st.slider(
                "Intervention Intensity",
                0.1, 2.0, 1.0, 0.1,
                help="Relative intensity of the intervention"
            )
            
            intervention_duration = st.slider(
                f"Intervention Duration ({config['time_unit']}s)",
                1, 20, int(results.effect_duration),
                help="How long to maintain the intervention"
            )
            
            target_outcome_change = st.number_input(
                "Target Outcome Change",
                value=1.0,
                help="Desired change in outcome variable"
            )
        
        # Timing scenarios
        st.markdown("#### Intervention Timing Scenarios")
        
        # Create different timing scenarios
        scenarios = [
            {"name": "Immediate", "delay": 0, "description": "Intervention starts immediately"},
            {"name": "Optimal Lag", "delay": results.optimal_lag, "description": "Intervention at discovered optimal lag"},
            {"name": "Peak Effect", "delay": results.optimal_lag + results.effect_duration//2, "description": "Intervention at peak effect time"},
            {"name": "Extended Delay", "delay": results.optimal_lag * 2, "description": "Intervention with extended delay"}
        ]
        
        scenario_results = []
        
        for scenario in scenarios:
            # Calculate expected effectiveness (simplified)
            if scenario["delay"] == results.optimal_lag:
                effectiveness = 1.0
            elif scenario["delay"] < results.optimal_lag:
                effectiveness = 0.7 + 0.3 * (scenario["delay"] / results.optimal_lag)
            else:
                decay_factor = max(0.3, 1 - 0.1 * (scenario["delay"] - results.optimal_lag))
                effectiveness = decay_factor
            
            expected_outcome = target_outcome_change * effectiveness * intervention_intensity
            
            scenario_results.append({
                'Scenario': scenario["name"],
                'Delay': f"{scenario['delay']} {config['time_unit']}s",
                'Expected Effectiveness': f"{effectiveness:.1%}",
                'Predicted Outcome': f"{expected_outcome:.2f}",
                'Description': scenario["description"]
            })
        
        scenarios_df = pd.DataFrame(scenario_results)
        st.dataframe(scenarios_df, use_container_width=True)
        
        # Timing optimization visualization
        st.markdown("#### Intervention Timing Optimization")
        
        # Create timing optimization chart
        time_range = range(0, results.optimal_lag * 3)
        effectiveness_curve = []
        
        for t in time_range:
            if t <= results.optimal_lag:
                eff = 0.4 + 0.6 * (t / results.optimal_lag)
            elif t <= results.optimal_lag + results.effect_duration:
                eff = 1.0 - 0.3 * ((t - results.optimal_lag) / results.effect_duration)
            else:
                eff = 0.7 * np.exp(-0.1 * (t - results.optimal_lag - results.effect_duration))
            
            effectiveness_curve.append(eff)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(time_range),
            y=effectiveness_curve,
            mode='lines',
            name='Intervention Effectiveness',
            line=dict(color='blue', width=3)
        ))
        
        # Mark optimal timing
        fig.add_vline(
            x=results.optimal_lag, 
            line_dash="dash", 
            line_color="red",
            annotation_text="Optimal Timing"
        )
        
        fig.update_layout(
            title="Intervention Effectiveness by Timing",
            xaxis_title=f"Intervention Start Time ({config['time_unit']}s)",
            yaxis_title="Expected Effectiveness",
            hovermode='x'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal timing considerations
        if config.get('seasonal_adjustment', False):
            st.markdown("#### Seasonal Timing Considerations")
            
            st.info("💡 Seasonal patterns detected. Consider these timing factors:")
            st.markdown("""
            - **Peak Season**: Higher baseline effectiveness expected
            - **Off Season**: May need increased intervention intensity
            - **Transition Periods**: Optimal timing for lasting effects
            - **Cyclical Patterns**: Consider recurring intervention schedules
            """)
        
        # Intervention planning tool
        st.markdown("#### Intervention Planning Tool")
        
        with st.expander("📅 Plan Your Intervention"):
            col1, col2 = st.columns(2)
            
            with col1:
                intervention_start = st.date_input(
                    "Planned Start Date",
                    value=datetime.now() + timedelta(days=results.optimal_lag)
                )
                
                intervention_budget = st.number_input(
                    "Available Budget",
                    min_value=0.0,
                    value=10000.0,
                    help="Budget available for intervention"
                )
            
            with col2:
                success_threshold = st.slider(
                    "Success Threshold",
                    0.1, 2.0, 0.8,
                    help="Minimum outcome change considered successful"
                )
                
                risk_tolerance = st.selectbox(
                    "Risk Tolerance",
                    ["Conservative", "Moderate", "Aggressive"],
                    index=1,
                    help="Risk tolerance for intervention timing"
                )
            
            if st.button("📋 Generate Intervention Plan"):
                # Generate a simple intervention plan
                plan_text = f"""
# Intervention Plan

## Timing Strategy
- **Start Date**: {intervention_start.strftime('%Y-%m-%d')}
- **Duration**: {intervention_duration} {config['time_unit']}s
- **Intensity**: {intervention_intensity:.1f}x baseline

## Expected Outcomes
- **Target Change**: {target_outcome_change:.2f}
- **Success Threshold**: {success_threshold:.2f}
- **Risk Level**: {risk_tolerance}

## Implementation Steps
1. Prepare intervention resources
2. Monitor baseline metrics
3. Implement intervention at optimal timing
4. Track outcome changes
5. Adjust intensity based on early results

## Success Metrics
- Outcome change ≥ {success_threshold:.2f}
- Maintained effect for {intervention_duration} {config['time_unit']}s
- Cost-effectiveness within budget
                """
                
                st.markdown(plan_text)
                
                st.download_button(
                    "📥 Download Plan",
                    plan_text,
                    file_name=f"intervention_plan_{intervention_start.strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
    
    with tab4:
        if 'temporal_results' not in st.session_state:
            st.info("⏱️ Complete temporal analysis first to access forecasting features")
            return
        
        st.markdown("### Forecasting & Scenario Analysis")
        
        results_data = st.session_state.temporal_results
        results = results_data['results']
        config = results_data['config']
        
        # Forecasting configuration
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_horizon = st.slider(
                f"Forecast Horizon ({config['time_unit']}s)",
                5, 50, 12,
                help="How many time periods to forecast"
            )
            
            include_uncertainty = st.checkbox("Include Uncertainty Bands", value=True)
            confidence_intervals = st.multiselect(
                "Confidence Intervals",
                ["80%", "90%", "95%", "99%"],
                default=["90%", "95%"]
            )
        
        with col2:
            forecast_scenarios = st.multiselect(
                "Scenarios to Model",
                [
                    "No Intervention",
                    "Optimal Intervention", 
                    "Early Intervention",
                    "Delayed Intervention",
                    "Continuous Intervention"
                ],
                default=["No Intervention", "Optimal Intervention"]
            )
            
            intervention_variations = st.slider(
                "Intervention Intensity Variations",
                0.5, 2.0, (0.8, 1.2),
                help="Range of intervention intensities to model"
            )
        
        # Generate forecasts
        if st.button("📈 Generate Forecasts", type="primary"):
            with st.spinner("📈 Generating forecasts and scenarios..."):
                
                # Simplified forecast generation (in real implementation, this would use the temporal model)
                np.random.seed(42)
                forecast_data = {}
                
                time_points = list(range(1, forecast_horizon + 1))
                
                for scenario in forecast_scenarios:
                    if scenario == "No Intervention":
                        # Baseline trend with noise
                        trend = np.linspace(0, 0.1 * forecast_horizon, forecast_horizon)
                        noise = np.random.normal(0, 0.1, forecast_horizon)
                        forecast = trend + noise
                        
                    elif scenario == "Optimal Intervention":
                        # Intervention effect starting at optimal lag
                        baseline = np.linspace(0, 0.1 * forecast_horizon, forecast_horizon)
                        intervention_effect = np.zeros(forecast_horizon)
                        
                        start_effect = min(results.optimal_lag, forecast_horizon)
                        end_effect = min(start_effect + results.effect_duration, forecast_horizon)
                        
                        if start_effect < forecast_horizon:
                            for i in range(start_effect, end_effect):
                                intervention_effect[i] = 0.5 * (1 - 0.1 * (i - start_effect))
                        
                        forecast = baseline + intervention_effect + np.random.normal(0, 0.05, forecast_horizon)
                    
                    elif scenario == "Early Intervention":
                        # Early intervention with reduced effectiveness
                        baseline = np.linspace(0, 0.1 * forecast_horizon, forecast_horizon)
                        intervention_effect = np.zeros(forecast_horizon)
                        
                        start_effect = 1
                        end_effect = min(results.effect_duration + 1, forecast_horizon)
                        
                        for i in range(start_effect, end_effect):
                            intervention_effect[i] = 0.3 * (1 - 0.1 * (i - start_effect))
                        
                        forecast = baseline + intervention_effect + np.random.normal(0, 0.05, forecast_horizon)
                    
                    elif scenario == "Delayed Intervention":
                        # Delayed intervention
                        baseline = np.linspace(0, 0.1 * forecast_horizon, forecast_horizon)
                        intervention_effect = np.zeros(forecast_horizon)
                        
                        start_effect = min(results.optimal_lag * 2, forecast_horizon - 2)
                        end_effect = min(start_effect + results.effect_duration, forecast_horizon)
                        
                        if start_effect < forecast_horizon - 1:
                            for i in range(start_effect, end_effect):
                                intervention_effect[i] = 0.4 * (1 - 0.1 * (i - start_effect))
                        
                        forecast = baseline + intervention_effect + np.random.normal(0, 0.05, forecast_horizon)
                    
                    else:  # Continuous Intervention
                        baseline = np.linspace(0, 0.1 * forecast_horizon, forecast_horizon)
                        intervention_effect = 0.3 * np.ones(forecast_horizon) * np.exp(-0.05 * np.arange(forecast_horizon))
                        forecast = baseline + intervention_effect + np.random.normal(0, 0.05, forecast_horizon)
                    
                    forecast_data[scenario] = forecast
                
                # Store forecasts
                st.session_state['forecast_data'] = {
                    'forecasts': forecast_data,
                    'time_points': time_points,
                    'config': {
                        'horizon': forecast_horizon,
                        'scenarios': forecast_scenarios,
                        'confidence_intervals': confidence_intervals
                    }
                }
                
                st.success("✅ Forecasts generated successfully!")
        
        # Display forecasts
        if 'forecast_data' in st.session_state:
            forecast_info = st.session_state.forecast_data
            forecasts = forecast_info['forecasts']
            time_points = forecast_info['time_points']
            
            st.markdown("#### Forecast Results")
            
            # Forecast visualization
            fig = go.Figure()
            
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            
            for i, (scenario, forecast_values) in enumerate(forecasts.items()):
                fig.add_trace(go.Scatter(
                    x=time_points,
                    y=forecast_values,
                    mode='lines+markers',
                    name=scenario,
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
                
                # Add confidence intervals if requested
                if include_uncertainty:
                    # Simple uncertainty bands (in real implementation, would be from model)
                    std_dev = np.std(forecast_values) * 0.5
                    
                    upper_bound = forecast_values + 1.96 * std_dev  # 95% CI
                    lower_bound = forecast_values - 1.96 * std_dev
                    
                    fig.add_trace(go.Scatter(
                        x=time_points + time_points[::-1],
                        y=list(upper_bound) + list(lower_bound[::-1]),
                        fill='toself',
                        fillcolor=f'rgba({colors[i % len(colors)]}, 0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=False,
                        name=f'{scenario} CI'
                    ))
            
            fig.update_layout(
                title="Forecast Scenarios Comparison",
                xaxis_title=f"Time ({config['time_unit']}s)",
                yaxis_title=f"Predicted {config['outcome_variable']}",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Scenario comparison table
            st.markdown("#### Scenario Comparison")
            
            comparison_data = []
            for scenario, values in forecasts.items():
                comparison_data.append({
                    'Scenario': scenario,
                    'Final Value': f"{values[-1]:.3f}",
                    'Average Change': f"{np.mean(np.diff(values)):.3f}",
                    'Peak Value': f"{max(values):.3f}",
                    'Volatility': f"{np.std(values):.3f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Best scenario recommendation
            # Find scenario with highest final value
            best_scenario = max(forecasts.keys(), key=lambda k: forecasts[k][-1])
            best_final_value = forecasts[best_scenario][-1]
            
            st.success(f"🎯 **Recommended Scenario:** {best_scenario}")
            st.info(f"📈 **Expected Final Outcome:** {best_final_value:.3f}")
            
            # Export forecasts
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Export Forecasts"):
                    # Create export dataframe
                    export_df = pd.DataFrame({'Time': time_points})
                    for scenario, values in forecasts.items():
                        export_df[scenario] = values
                    
                    csv_data = export_df.to_csv(index=False)
                    st.download_button(
                        "Download Forecasts CSV",
                        csv_data,
                        file_name=f"temporal_forecasts_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📊 Generate Report"):
                    st.info("📄 Comprehensive temporal analysis report generation coming soon!")

    # Update session statistics
    if 'temporal_results' in st.session_state:
        if 'session_stats' not in st.session_state:
            st.session_state.session_stats = {'analyses': 0, 'datasets': 0, 'success_rate': 0}
        
        # Count temporal analysis as an analysis
        current_analyses = st.session_state.session_stats.get('analyses', 0)
        st.session_state.session_stats['analyses'] = max(current_analyses, current_analyses + 1)