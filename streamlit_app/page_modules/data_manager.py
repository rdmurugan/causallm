import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from causalllm.data_manager import CausalDataManager
from causalllm.llm_client import get_llm_client
import io

def show():
    st.title("📊 Data Manager")
    st.markdown("Upload, validate, and manage datasets for causal analysis")
    
    # Initialize data manager
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = CausalDataManager()
    
    manager = st.session_state.data_manager
    
    # Tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Upload Data", "🔍 Data Quality", "🏷️ Variable Roles", "📈 Data Exploration"
    ])
    
    with tab1:
        st.markdown("### Upload Dataset")
        
        # File upload options
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a dataset file",
                type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
                help="Supported formats: CSV, Excel, JSON, Parquet"
            )
        
        with col2:
            # Sample data option
            if st.button("📋 Use Sample Data"):
                if 'sample_data' in st.session_state:
                    st.session_state['current_data'] = st.session_state.sample_data
                    st.session_state['data_source'] = f"Sample: {st.session_state.get('sample_data_name', 'Unknown')}"
                    st.success("✅ Sample data loaded!")
                    st.rerun()
                else:
                    st.warning("No sample data available. Load sample data from the Home page.")
        
        if uploaded_file is not None:
            try:
                # Load data using the data manager
                with st.spinner("Loading data..."):
                    # Save uploaded file temporarily and load
                    file_bytes = uploaded_file.read()
                    
                    if uploaded_file.name.endswith('.csv'):
                        data = pd.read_csv(io.StringIO(file_bytes.decode('utf-8')))
                    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                        data = pd.read_excel(io.BytesIO(file_bytes))
                    elif uploaded_file.name.endswith('.json'):
                        data = pd.read_json(io.StringIO(file_bytes.decode('utf-8')))
                    elif uploaded_file.name.endswith('.parquet'):
                        data = pd.read_parquet(io.BytesIO(file_bytes))
                    else:
                        st.error("Unsupported file format")
                        return
                
                st.session_state['current_data'] = data
                st.session_state['data_source'] = uploaded_file.name
                
                st.success(f"✅ Successfully loaded {uploaded_file.name}")
                st.info(f"Dataset shape: {data.shape[0]:,} rows × {data.shape[1]} columns")
                
                # Show data preview
                st.markdown("#### Data Preview")
                st.dataframe(data.head(10), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Rest of the tabs only show if data is loaded
    if 'current_data' not in st.session_state:
        with tab2:
            st.info("📁 Please upload a dataset first to access data quality analysis")
        with tab3:
            st.info("📁 Please upload a dataset first to assign variable roles")  
        with tab4:
            st.info("📁 Please upload a dataset first to explore the data")
        return
    
    data = st.session_state.current_data
    
    with tab2:
        st.markdown("### Data Quality Assessment")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{data.shape[0]:,}")
        with col2:
            st.metric("Total Columns", data.shape[1])
        with col3:
            missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        with col4:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Columns", len(numeric_cols))
        
        # Missing data analysis
        st.markdown("#### Missing Data Analysis")
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            fig = px.bar(
                x=missing_data.values,
                y=missing_data.index,
                orientation='h',
                title="Missing Values by Column",
                labels={'x': 'Missing Count', 'y': 'Column'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values detected!")
        
        # Data type analysis
        st.markdown("#### Data Types")
        col1, col2 = st.columns(2)
        
        with col1:
            dtype_counts = data.dtypes.value_counts()
            # Convert dtype names to strings for JSON serialization
            dtype_names = [str(dtype) for dtype in dtype_counts.index]
            fig = px.pie(
                values=dtype_counts.values,
                names=dtype_names,
                title="Data Type Distribution"
            )
            st.plotly_chart(fig)
        
        with col2:
            # Show detailed data type info
            dtype_info = pd.DataFrame({
                'Column': data.columns,
                'Data Type': [str(dtype) for dtype in data.dtypes],  # Convert to strings
                'Non-Null Count': data.count(),
                'Unique Values': data.nunique()
            })
            st.dataframe(dtype_info, use_container_width=True)
        
        # Outlier detection for numeric columns
        if len(numeric_cols) > 0:
            st.markdown("#### Outlier Analysis")
            selected_col = st.selectbox("Select column for outlier analysis", numeric_cols)
            
            if selected_col:
                col_data = data[selected_col].dropna()
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Outliers Detected", len(outliers))
                    st.metric("Outlier Percentage", f"{len(outliers)/len(col_data)*100:.1f}%")
                
                with col2:
                    fig = px.box(y=col_data, title=f"Box Plot: {selected_col}")
                    st.plotly_chart(fig)
        
        # Data quality recommendations
        st.markdown("#### 💡 Data Quality Recommendations")
        recommendations = []
        
        if missing_pct > 5:
            recommendations.append("⚠️ High missing data percentage. Consider imputation strategies.")
        if len(missing_data) > data.shape[1] * 0.3:
            recommendations.append("⚠️ Many columns have missing values. Review data collection process.")
        if data.duplicated().sum() > 0:
            recommendations.append(f"⚠️ {data.duplicated().sum()} duplicate rows found. Consider removing duplicates.")
        
        if recommendations:
            for rec in recommendations:
                st.warning(rec)
        else:
            st.success("✅ Data quality looks good!")
    
    with tab3:
        st.markdown("### Variable Role Assignment")
        st.info("Assign roles to variables for causal analysis")
        
        variables = data.columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Treatment Variables")
            treatment_vars = st.multiselect(
                "Variables that represent interventions or treatments",
                variables,
                key="treatment_vars",
                help="Variables that you manipulate or that represent different conditions"
            )
            
            st.markdown("#### Outcome Variables") 
            outcome_vars = st.multiselect(
                "Variables that represent outcomes of interest",
                [v for v in variables if v not in treatment_vars],
                key="outcome_vars",
                help="Variables that you want to understand the effect on"
            )
        
        with col2:
            st.markdown("#### Confounding Variables")
            remaining_vars = [v for v in variables if v not in treatment_vars + outcome_vars]
            confounder_vars = st.multiselect(
                "Variables that might confound the relationship",
                remaining_vars,
                key="confounder_vars",
                help="Variables that might affect both treatment and outcome"
            )
            
            st.markdown("#### Time Variable (Optional)")
            time_var = st.selectbox(
                "Select time/date variable if applicable",
                ["None"] + remaining_vars,
                key="time_var",
                help="Variable representing time for temporal analysis"
            )
        
        # Store variable roles in session state
        if treatment_vars or outcome_vars:
            st.session_state['variable_roles'] = {
                'treatment': treatment_vars,
                'outcome': outcome_vars,
                'confounders': confounder_vars,
                'time': time_var if time_var != "None" else None,
                'remaining': [v for v in variables 
                             if v not in treatment_vars + outcome_vars + confounder_vars 
                             and v != time_var]
            }
            
            # Display summary
            st.markdown("#### 📋 Variable Role Summary")
            roles = st.session_state.variable_roles
            
            summary_data = []
            for role, vars_list in roles.items():
                if vars_list and role != 'remaining':
                    if isinstance(vars_list, list):
                        for var in vars_list:
                            summary_data.append({'Variable': var, 'Role': role.title()})
                    else:
                        summary_data.append({'Variable': vars_list, 'Role': role.title()})
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Save button
                if st.button("💾 Save Variable Roles", type="primary"):
                    st.success("✅ Variable roles saved! You can now proceed to Causal Discovery.")
    
    with tab4:
        st.markdown("### Data Exploration")
        
        # Statistical summary
        st.markdown("#### Statistical Summary")
        
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            st.dataframe(numeric_data.describe(), use_container_width=True)
        
        # Correlation analysis
        if len(numeric_data.columns) > 1:
            st.markdown("#### Correlation Matrix")
            
            # Filter out columns with too many NaNs
            corr_data = numeric_data.dropna(axis=1, thresh=len(numeric_data)*0.5)
            
            if len(corr_data.columns) > 1:
                corr_matrix = corr_data.corr()
                
                # Ensure column names are strings for JSON serialization
                corr_matrix.index = corr_matrix.index.astype(str)
                corr_matrix.columns = corr_matrix.columns.astype(str)
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Variable Correlation Matrix",
                    color_continuous_scale="RdBu"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Distribution plots
        st.markdown("#### Variable Distributions")
        
        plot_cols = st.columns(2)
        with plot_cols[0]:
            numeric_col = st.selectbox("Select numeric variable", numeric_data.columns)
        with plot_cols[1]:
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                categorical_col = st.selectbox("Select categorical variable (optional)", 
                                             ["None"] + categorical_cols.tolist())
            else:
                categorical_col = "None"
        
        if numeric_col:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(
                    data, 
                    x=numeric_col,
                    title=f"Distribution of {numeric_col}",
                    marginal="box"
                )
                st.plotly_chart(fig)
            
            with col2:
                if categorical_col != "None":
                    # Box plot by category
                    fig = px.box(
                        data,
                        x=categorical_col,
                        y=numeric_col,
                        title=f"{numeric_col} by {categorical_col}"
                    )
                    st.plotly_chart(fig)
                else:
                    # Show basic stats
                    col_stats = data[numeric_col].describe()
                    st.markdown(f"**{numeric_col} Statistics:**")
                    for stat, value in col_stats.items():
                        if isinstance(value, (int, float)):
                            st.metric(stat.title(), f"{value:.2f}")
        
        # Pairwise relationships
        if len(numeric_data.columns) >= 2:
            st.markdown("#### Pairwise Relationships")
            
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("X Variable", numeric_data.columns, key="x_var")
            with col2:
                y_var = st.selectbox("Y Variable", 
                                   [col for col in numeric_data.columns if col != x_var],
                                   key="y_var")
            
            if x_var and y_var:
                fig = px.scatter(
                    data, 
                    x=x_var, 
                    y=y_var,
                    title=f"{y_var} vs {x_var}",
                    trendline="ols" if len(data) < 1000 else None
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Data export
        st.markdown("#### Export Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download CSV"):
                csv = data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"processed_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📥 Download Excel"):
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    data.to_excel(writer, sheet_name='Data', index=False)
                    if 'variable_roles' in st.session_state:
                        roles_df = pd.DataFrame([
                            {'Variable': var, 'Role': role}
                            for role, vars_list in st.session_state.variable_roles.items()
                            for var in (vars_list if isinstance(vars_list, list) else [vars_list])
                            if vars_list and var
                        ])
                        roles_df.to_excel(writer, sheet_name='Variable_Roles', index=False)
                
                st.download_button(
                    label="Download Excel",
                    data=buffer.getvalue(),
                    file_name=f"processed_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col3:
            if st.button("🗑️ Clear Data"):
                for key in ['current_data', 'data_source', 'variable_roles']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("Data cleared!")
                st.rerun()

    # Update session stats
    if 'current_data' in st.session_state:
        if 'session_stats' not in st.session_state:
            st.session_state.session_stats = {'datasets': 0, 'analyses': 0, 'success_rate': 0}
        
        if st.session_state.session_stats['datasets'] == 0:
            st.session_state.session_stats['datasets'] = 1