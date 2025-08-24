import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

def show():
    st.title("📊 Analytics Dashboard")
    st.markdown("Monitor usage patterns, analysis trends, and system performance")
    
    # Initialize analytics data if not exists
    if 'analytics_data' not in st.session_state:
        initialize_analytics_data()
    
    # Update current session stats
    update_session_stats()
    
    # Create dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "👥 Usage Analytics", "🔬 Analysis Trends", "⚡ Performance", "📊 Reports"
    ])
    
    with tab1:
        st.markdown("### Dashboard Overview")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        analytics = st.session_state.analytics_data
        
        with col1:
            total_sessions = len(analytics['sessions'])
            st.metric(
                "Total Sessions",
                total_sessions,
                delta=calculate_delta(analytics['sessions'], 'daily'),
                help="Total number of user sessions"
            )
        
        with col2:
            total_analyses = sum(s.get('analyses_run', 0) for s in analytics['sessions'])
            st.metric(
                "Analyses Run",
                total_analyses,
                delta=calculate_delta(analytics['analyses'], 'daily') if 'analyses' in analytics else 0,
                help="Total causal analyses performed"
            )
        
        with col3:
            avg_success_rate = np.mean([s.get('success_rate', 100) for s in analytics['sessions']])
            st.metric(
                "Success Rate",
                f"{avg_success_rate:.1f}%",
                delta=f"{avg_success_rate - 85:.1f}%" if avg_success_rate > 85 else None,
                help="Average analysis success rate"
            )
        
        with col4:
            active_datasets = len(set(s.get('dataset', 'unknown') for s in analytics['sessions'] if s.get('dataset')))
            st.metric(
                "Active Datasets",
                active_datasets,
                help="Number of unique datasets analyzed"
            )
        
        # Usage trend chart
        st.markdown("#### Usage Trends (Last 30 Days)")
        
        # Generate sample trend data
        trend_data = generate_trend_data()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trend_data['date'],
            y=trend_data['sessions'],
            mode='lines+markers',
            name='Sessions',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=trend_data['date'],
            y=trend_data['analyses'],
            mode='lines+markers',
            name='Analyses',
            line=dict(color='green', width=3),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Daily Usage Trends",
            xaxis_title="Date",
            yaxis=dict(title="Sessions", side='left'),
            yaxis2=dict(title="Analyses", side='right', overlaying='y'),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature usage distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Most Used Features")
            
            feature_usage = {
                'Data Manager': 85,
                'Causal Discovery': 72,
                'Interactive Q&A': 58,
                'Validation Suite': 45,
                'Intervention Optimizer': 38,
                'Temporal Analysis': 31,
                'Visualization': 29
            }
            
            fig = px.bar(
                x=list(feature_usage.keys()),
                y=list(feature_usage.values()),
                title="Feature Usage (%)",
                color=list(feature_usage.values()),
                color_continuous_scale="Blues"
            )
            
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Analysis Types Distribution")
            
            analysis_types = {
                'Observational Studies': 45,
                'Randomized Experiments': 25,
                'Natural Experiments': 15,
                'Time Series Analysis': 10,
                'Cross-sectional': 5
            }
            
            fig = px.pie(
                values=list(analysis_types.values()),
                names=list(analysis_types.keys()),
                title="Analysis Types"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### User & Usage Analytics")
        
        # Session statistics
        st.markdown("#### Session Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        session_durations = [s.get('duration', np.random.normal(45, 15)) for s in analytics['sessions']]
        session_durations = [max(5, d) for d in session_durations]  # Min 5 minutes
        
        with col1:
            avg_duration = np.mean(session_durations)
            st.metric(
                "Avg Session Duration",
                f"{avg_duration:.1f} min",
                help="Average time spent per session"
            )
        
        with col2:
            total_time = sum(session_durations)
            st.metric(
                "Total Usage Time",
                f"{total_time/60:.1f} hours",
                help="Total time across all sessions"
            )
        
        with col3:
            bounce_rate = len([s for s in analytics['sessions'] if s.get('analyses_run', 0) == 0]) / max(1, len(analytics['sessions'])) * 100
            st.metric(
                "Bounce Rate",
                f"{bounce_rate:.1f}%",
                delta=f"{bounce_rate - 30:.1f}%" if bounce_rate < 30 else None,
                help="Sessions with no analyses performed"
            )
        
        # Session duration distribution
        st.markdown("#### Session Duration Distribution")
        
        fig = px.histogram(
            x=session_durations,
            nbins=20,
            title="Session Duration Distribution",
            labels={'x': 'Duration (minutes)', 'y': 'Number of Sessions'}
        )
        
        fig.add_vline(x=np.mean(session_durations), line_dash="dash", 
                     annotation_text=f"Average: {np.mean(session_durations):.1f} min")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Peak usage hours
        st.markdown("#### Usage Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hour of day analysis
            hours = np.random.choice(range(24), size=len(analytics['sessions']), 
                                   p=np.array([0.5, 0.3, 0.2, 0.1, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 
                                             4.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4]) / 42.3)
            
            hourly_usage = pd.Series(hours).value_counts().sort_index()
            
            fig = px.bar(
                x=hourly_usage.index,
                y=hourly_usage.values,
                title="Usage by Hour of Day",
                labels={'x': 'Hour of Day', 'y': 'Number of Sessions'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Day of week analysis
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_weights = [1.2, 1.3, 1.4, 1.3, 1.1, 0.4, 0.3]  # Weekday bias
            day_usage = np.random.choice(days, size=len(analytics['sessions']), 
                                       p=np.array(day_weights) / sum(day_weights))
            
            daily_counts = pd.Series(day_usage).value_counts().reindex(days)
            
            fig = px.bar(
                x=days,
                y=daily_counts.values,
                title="Usage by Day of Week",
                labels={'x': 'Day of Week', 'y': 'Number of Sessions'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Geographic distribution (simulated)
        st.markdown("#### Geographic Distribution")
        
        countries = ['United States', 'United Kingdom', 'Germany', 'Canada', 'Australia', 
                    'France', 'Netherlands', 'Sweden', 'Switzerland', 'Japan']
        usage_counts = np.random.poisson(10, len(countries)) + np.arange(len(countries), 0, -1) * 2
        
        geo_data = pd.DataFrame({
            'Country': countries,
            'Sessions': usage_counts,
            'Usage_Percentage': usage_counts / sum(usage_counts) * 100
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                geo_data.head(10),
                x='Sessions',
                y='Country',
                orientation='h',
                title="Top 10 Countries by Usage"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(geo_data, use_container_width=True)
    
    with tab3:
        st.markdown("### Analysis Trends & Insights")
        
        # Analysis success rates over time
        st.markdown("#### Analysis Success Rates")
        
        success_data = generate_success_rate_data()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=success_data['date'],
            y=success_data['success_rate'],
            mode='lines+markers',
            name='Success Rate',
            line=dict(color='green', width=3),
            fill='tonexty'
        ))
        
        fig.add_hline(y=90, line_dash="dash", line_color="red", 
                     annotation_text="Target: 90%")
        
        fig.update_layout(
            title="Analysis Success Rate Trend",
            xaxis_title="Date",
            yaxis_title="Success Rate (%)",
            yaxis=dict(range=[80, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Common analysis patterns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Popular Analysis Workflows")
            
            workflows = {
                'Data → Discovery → Validation': 45,
                'Data → Q&A → Discovery': 25,
                'Data → Validation Only': 15,
                'Discovery → Optimization': 10,
                'Full Pipeline': 5
            }
            
            workflow_df = pd.DataFrame(list(workflows.items()), 
                                     columns=['Workflow', 'Frequency'])
            
            fig = px.treemap(
                workflow_df,
                path=['Workflow'],
                values='Frequency',
                title="Analysis Workflow Patterns"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Error Analysis")
            
            error_types = {
                'Data Quality Issues': 35,
                'Model Specification': 25,
                'Insufficient Sample Size': 20,
                'Missing Variables': 12,
                'Technical Errors': 8
            }
            
            fig = px.pie(
                values=list(error_types.values()),
                names=list(error_types.keys()),
                title="Common Error Types"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Domain-specific insights
        st.markdown("#### Domain-Specific Usage")
        
        domain_data = {
            'Healthcare': {'analyses': 142, 'avg_effect': 0.23, 'success_rate': 87},
            'Economics': {'analyses': 98, 'avg_effect': 0.31, 'success_rate': 92},
            'Education': {'analyses': 76, 'avg_effect': 0.18, 'success_rate': 89},
            'Technology': {'analyses': 54, 'avg_effect': 0.28, 'success_rate': 85},
            'Marketing': {'analyses': 43, 'avg_effect': 0.35, 'success_rate': 83},
            'Social Sciences': {'analyses': 38, 'avg_effect': 0.21, 'success_rate': 88}
        }
        
        domain_df = pd.DataFrame(domain_data).T
        domain_df.reset_index(inplace=True)
        domain_df.rename(columns={'index': 'Domain'}, inplace=True)
        
        # Bubble chart
        fig = px.scatter(
            domain_df,
            x='avg_effect',
            y='success_rate',
            size='analyses',
            color='Domain',
            title="Domain Analysis: Effect Size vs Success Rate",
            labels={
                'avg_effect': 'Average Effect Size',
                'success_rate': 'Success Rate (%)',
                'analyses': 'Number of Analyses'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed domain table
        st.dataframe(domain_df, use_container_width=True)
    
    with tab4:
        st.markdown("### Performance Metrics")
        
        # System performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_response_time = np.random.normal(2.3, 0.5)
            st.metric(
                "Avg Response Time",
                f"{avg_response_time:.1f}s",
                delta=f"{avg_response_time - 2.0:.1f}s" if avg_response_time > 2.0 else None,
                help="Average time for analysis completion"
            )
        
        with col2:
            uptime = 99.2
            st.metric(
                "System Uptime",
                f"{uptime:.1f}%",
                delta=f"{uptime - 99:.1f}%" if uptime > 99 else None,
                help="System availability percentage"
            )
        
        with col3:
            memory_usage = np.random.uniform(45, 75)
            st.metric(
                "Memory Usage",
                f"{memory_usage:.1f}%",
                delta=f"{memory_usage - 50:.1f}%" if memory_usage < 80 else None,
                help="Current memory utilization"
            )
        
        with col4:
            concurrent_users = np.random.poisson(15)
            st.metric(
                "Concurrent Users",
                concurrent_users,
                help="Current active users"
            )
        
        # Performance trends
        st.markdown("#### Performance Trends")
        
        perf_data = generate_performance_data()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Response Time', 'Memory Usage', 'CPU Usage', 'Active Sessions'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Scatter(x=perf_data['timestamp'], y=perf_data['response_time'],
                      name='Response Time (s)', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=perf_data['timestamp'], y=perf_data['memory_usage'],
                      name='Memory Usage (%)', line=dict(color='red')),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=perf_data['timestamp'], y=perf_data['cpu_usage'],
                      name='CPU Usage (%)', line=dict(color='green')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=perf_data['timestamp'], y=perf_data['active_sessions'],
                      name='Active Sessions', line=dict(color='purple')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Error monitoring
        st.markdown("#### Error Monitoring")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Error rate over time
            error_data = generate_error_data()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=error_data['timestamp'],
                y=error_data['error_rate'],
                mode='lines+markers',
                name='Error Rate (%)',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title="Error Rate Trend",
                xaxis_title="Time",
                yaxis_title="Error Rate (%)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Error severity distribution
            severity_data = {
                'Critical': 2,
                'High': 5,
                'Medium': 12,
                'Low': 23,
                'Info': 45
            }
            
            fig = px.bar(
                x=list(severity_data.keys()),
                y=list(severity_data.values()),
                title="Error Severity Distribution",
                color=list(severity_data.keys()),
                color_discrete_map={
                    'Critical': 'darkred',
                    'High': 'red', 
                    'Medium': 'orange',
                    'Low': 'yellow',
                    'Info': 'lightblue'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("### Reports & Export")
        
        # Report generation options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Generate Reports")
            
            report_type = st.selectbox(
                "Report type",
                ["Usage Summary", "Performance Report", "Analysis Insights", 
                 "Custom Dashboard", "Executive Summary"]
            )
            
            date_range = st.date_input(
                "Date range",
                value=[datetime.now() - timedelta(days=30), datetime.now()],
                help="Select date range for the report"
            )
            
            include_charts = st.checkbox("Include visualizations", value=True)
            include_raw_data = st.checkbox("Include raw data", value=False)
            
            if st.button("📊 Generate Report", type="primary"):
                with st.spinner("Generating report..."):
                    # Simulate report generation
                    report_content = generate_report(report_type, date_range, include_charts, include_raw_data)
                    
                st.success(f"✅ {report_type} generated successfully!")
                
                # Download buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        "📄 Download PDF",
                        data="Sample PDF content",
                        file_name=f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"
                    )
                
                with col2:
                    st.download_button(
                        "📊 Download Excel",
                        data="Sample Excel content",
                        file_name=f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col3:
                    st.download_button(
                        "📋 Download CSV",
                        data=generate_csv_report(),
                        file_name=f"analytics_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        
        with col2:
            st.markdown("#### Automated Reports")
            
            # Scheduled report settings
            enable_scheduled = st.checkbox("Enable scheduled reports", value=False)
            
            if enable_scheduled:
                schedule_frequency = st.selectbox(
                    "Report frequency",
                    ["Daily", "Weekly", "Monthly", "Quarterly"]
                )
                
                report_recipients = st.text_area(
                    "Email recipients",
                    placeholder="user1@example.com\nuser2@example.com",
                    help="One email per line"
                )
                
                if st.button("💾 Save Schedule"):
                    st.success("Report schedule saved!")
            
            st.markdown("#### Dashboard Customization")
            
            # Custom dashboard settings
            dashboard_widgets = st.multiselect(
                "Select dashboard widgets",
                [
                    "Usage Overview", "Performance Metrics", "Error Monitoring",
                    "Feature Usage", "Geographic Distribution", "Success Rates"
                ],
                default=["Usage Overview", "Performance Metrics"]
            )
            
            refresh_interval = st.selectbox(
                "Auto-refresh interval",
                ["Manual", "30 seconds", "1 minute", "5 minutes", "15 minutes"],
                index=2
            )
            
            if st.button("🎨 Customize Dashboard"):
                st.session_state.dashboard_config = {
                    'widgets': dashboard_widgets,
                    'refresh_interval': refresh_interval
                }
                st.success("Dashboard configuration saved!")
        
        # Export current session data
        st.markdown("#### Export Current Session")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Export Session Data"):
                session_data = get_current_session_data()
                st.download_button(
                    "Download Session Data",
                    data=json.dumps(session_data, indent=2),
                    file_name=f"session_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📈 Export Analytics"):
                analytics_export = st.session_state.analytics_data
                st.download_button(
                    "Download Analytics Data",
                    data=json.dumps(analytics_export, indent=2, default=str),
                    file_name=f"analytics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("🔄 Reset Analytics"):
                if st.button("⚠️ Confirm Reset", type="secondary"):
                    initialize_analytics_data(reset=True)
                    st.success("Analytics data reset!")
                    st.rerun()

def initialize_analytics_data(reset=False):
    """Initialize or reset analytics data"""
    if reset or 'analytics_data' not in st.session_state:
        # Generate sample historical data
        base_date = datetime.now() - timedelta(days=90)
        sessions = []
        
        for i in range(150):  # 150 sample sessions over 90 days
            session_date = base_date + timedelta(days=np.random.uniform(0, 90))
            sessions.append({
                'session_id': f"session_{i+1}",
                'timestamp': session_date,
                'duration': max(5, np.random.normal(35, 20)),  # Minutes
                'analyses_run': np.random.poisson(2),
                'success_rate': min(100, max(60, np.random.normal(88, 12))),
                'dataset': np.random.choice(['healthcare.csv', 'economics.xlsx', 'education.json', 
                                           'marketing.csv', 'tech.parquet'], p=[0.3, 0.25, 0.2, 0.15, 0.1]),
                'features_used': np.random.choice(['data_manager', 'discovery', 'qa', 'validation', 
                                                 'optimization', 'temporal', 'visualization'], 
                                                size=np.random.randint(1, 4), replace=False).tolist()
            })
        
        st.session_state.analytics_data = {
            'sessions': sessions,
            'initialized': datetime.now(),
            'version': '1.0'
        }

def update_session_stats():
    """Update current session statistics"""
    if 'session_stats' not in st.session_state:
        st.session_state.session_stats = {
            'session_start': datetime.now(),
            'pages_visited': set(),
            'analyses_run': 0,
            'errors_encountered': 0
        }
    
    # Add current page to visited pages
    st.session_state.session_stats['pages_visited'].add('analytics')

def calculate_delta(data_list, period='daily'):
    """Calculate change in metrics"""
    if len(data_list) < 2:
        return 0
    
    # Simplified delta calculation
    recent = len([d for d in data_list if isinstance(d, dict) and 
                  d.get('timestamp', datetime.min) > datetime.now() - timedelta(days=7)])
    previous = len(data_list) - recent
    
    return recent - previous

def generate_trend_data():
    """Generate sample trend data for charts"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                         end=datetime.now(), freq='D')
    
    base_sessions = 15
    base_analyses = 25
    
    sessions = [max(1, base_sessions + np.random.normal(0, 5) + 
                   3 * np.sin(i * 2 * np.pi / 7)) for i in range(len(dates))]  # Weekly pattern
    analyses = [max(1, base_analyses + np.random.normal(0, 8) + 
                   5 * np.sin(i * 2 * np.pi / 7)) for i in range(len(dates))]  # Weekly pattern
    
    return pd.DataFrame({
        'date': dates,
        'sessions': sessions,
        'analyses': analyses
    })

def generate_success_rate_data():
    """Generate success rate trend data"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                         end=datetime.now(), freq='D')
    
    # Simulate improving success rate over time
    base_rate = 85
    success_rates = [min(99, max(75, base_rate + i * 0.1 + np.random.normal(0, 3))) 
                    for i in range(len(dates))]
    
    return pd.DataFrame({
        'date': dates,
        'success_rate': success_rates
    })

def generate_performance_data():
    """Generate performance metrics data"""
    timestamps = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                              end=datetime.now(), freq='H')
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'response_time': [max(0.5, 2.0 + np.random.normal(0, 0.5)) for _ in timestamps],
        'memory_usage': [max(20, min(90, 60 + np.random.normal(0, 10))) for _ in timestamps],
        'cpu_usage': [max(10, min(95, 40 + np.random.normal(0, 15))) for _ in timestamps],
        'active_sessions': [max(1, np.random.poisson(12)) for _ in timestamps]
    })

def generate_error_data():
    """Generate error monitoring data"""
    timestamps = pd.date_range(start=datetime.now() - timedelta(days=7), 
                              end=datetime.now(), freq='4H')
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'error_rate': [max(0, np.random.exponential(2)) for _ in timestamps]
    })

def generate_report(report_type, date_range, include_charts, include_raw_data):
    """Generate report content based on parameters"""
    return f"""
    Report Type: {report_type}
    Date Range: {date_range[0]} to {date_range[1]}
    Include Charts: {include_charts}
    Include Raw Data: {include_raw_data}
    
    Generated at: {datetime.now()}
    """

def generate_csv_report():
    """Generate CSV format analytics report"""
    data = {
        'Metric': ['Total Sessions', 'Total Analyses', 'Average Success Rate', 'Average Duration'],
        'Value': [150, 287, 88.5, 35.2],
        'Unit': ['count', 'count', 'percentage', 'minutes']
    }
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

def get_current_session_data():
    """Get current session data for export"""
    return {
        'session_stats': st.session_state.get('session_stats', {}),
        'current_data_info': {
            'has_data': 'current_data' in st.session_state,
            'data_shape': st.session_state.current_data.shape if 'current_data' in st.session_state else None,
            'data_source': st.session_state.get('data_source', 'Unknown')
        },
        'variable_roles': st.session_state.get('variable_roles', {}),
        'analyses_performed': {
            'discovery': 'discovery_results' in st.session_state,
            'validation': 'validation_results' in st.session_state,
            'optimization': 'optimization_results' in st.session_state
        }
    }