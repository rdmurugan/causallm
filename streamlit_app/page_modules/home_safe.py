import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

def show():
    # Hero section
    st.markdown("""
    <div class="main-header" style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1 style="margin: 0; font-size: 2.5rem;">🧠 Welcome to CausalLLM Pro</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">AI-Powered Causal Intelligence Platform</p>
        <p style="margin: 0.2rem 0 0 0; opacity: 0.9;">From Correlation to Causation with 78+ Advanced Features</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick start section
    st.markdown("## 🚀 Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 1. Upload Data
        Start by uploading your dataset (CSV, Excel, or JSON) in the Data Manager.
        - Automatic quality assessment
        - Variable role assignment  
        - Missing data analysis
        """)
    
    with col2:
        st.markdown("""
        ### 🔍 2. Discover Relationships
        Use AI-powered causal discovery to find relationships in your data.
        - Multiple discovery algorithms
        - Confidence scoring
        - Interactive visualization
        """)
    
    with col3:
        st.markdown("""
        ### ✅ 3. Validate & Optimize  
        Validate assumptions and optimize interventions.
        - Statistical tests
        - Sensitivity analysis
        - Intervention design
        """)
    
    # Feature showcase
    st.markdown("## ⭐ Key Features")
    
    feature_data = {
        "Category": ["Discovery", "Validation", "Optimization", "Visualization", "Analysis"],
        "Features": [15, 12, 8, 6, 10],
        "Description": [
            "Automated causal structure discovery",
            "Comprehensive assumption checking", 
            "Intervention strategy optimization",
            "Professional graph visualization",
            "Advanced statistical analysis"
        ]
    }
    
    df = pd.DataFrame(feature_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Feature Distribution")
        st.bar_chart(df.set_index('Category')['Features'])
    
    with col2:
        st.markdown("### 📋 Feature Overview") 
        st.dataframe(df, use_container_width=True)
    
    # Sample data section
    st.markdown("## 📊 Sample Datasets")
    
    sample_datasets = {
        "Healthcare": {
            "description": "Clinical trial data with treatment outcomes",
            "variables": "treatment, age, severity, outcome", 
            "size": "1,000 patients",
            "domain": "Medical research"
        },
        "Economics": {
            "description": "Economic policy impact analysis",
            "variables": "policy, gdp, unemployment, inflation",
            "size": "500 regions",
            "domain": "Economic policy"
        },
        "Education": {
            "description": "Educational intervention effectiveness",
            "variables": "program, background, resources, achievement", 
            "size": "2,000 students",
            "domain": "Educational research"
        }
    }
    
    selected_dataset = st.selectbox(
        "Choose a sample dataset to explore:",
        list(sample_datasets.keys())
    )
    
    if selected_dataset:
        dataset_info = sample_datasets[selected_dataset]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Description:** {dataset_info['description']}")
            st.info(f"**Domain:** {dataset_info['domain']}")
        
        with col2:
            st.info(f"**Variables:** {dataset_info['variables']}")
            st.info(f"**Size:** {dataset_info['size']}")
        
        if st.button(f"📥 Load {selected_dataset} Dataset", type="primary"):
            # Generate sample data
            np.random.seed(42)  # For reproducible results
            
            if selected_dataset == "Healthcare":
                n_samples = 1000
                data = pd.DataFrame({
                    'treatment': np.random.choice(['Control', 'Treatment A', 'Treatment B'], n_samples),
                    'age': np.random.normal(50, 15, n_samples).astype(int),
                    'severity': np.random.uniform(1, 10, n_samples).round(1),
                    'outcome': np.random.normal(5, 2, n_samples).round(1)
                })
            elif selected_dataset == "Economics":
                n_samples = 500
                data = pd.DataFrame({
                    'policy': np.random.choice(['A', 'B', 'C'], n_samples),
                    'gdp': np.random.normal(50000, 10000, n_samples).round(0),
                    'unemployment': np.random.uniform(2, 15, n_samples).round(1),
                    'inflation': np.random.uniform(-2, 8, n_samples).round(2)
                })
            else:  # Education
                n_samples = 2000
                data = pd.DataFrame({
                    'program': np.random.choice(['Standard', 'Enhanced', 'Intensive'], n_samples),
                    'background': np.random.choice(['Low', 'Medium', 'High'], n_samples),
                    'resources': np.random.uniform(1, 10, n_samples).round(1),
                    'achievement': np.random.normal(75, 15, n_samples).round(1)
                })
            
            # Store in session state
            st.session_state['sample_data'] = data
            st.session_state['sample_data_name'] = selected_dataset
            
            st.success(f"✅ {selected_dataset} dataset loaded! ({len(data)} rows)")
            st.markdown("### Data Preview")
            st.dataframe(data.head(10), use_container_width=True)
            
            st.info("💡 **Next Steps:** Go to Data Manager to assign variable roles and start your causal analysis!")
    
    # Statistics section
    st.markdown("## 📈 Platform Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Features", "78+", delta="5 new")
        
    with col2:
        st.metric("Analysis Methods", "25+", delta="3 updated")
        
    with col3:
        st.metric("Sample Datasets", "12", delta="2 added")
        
    with col4:
        st.metric("Visualization Types", "15+", delta="4 new")
    
    # Getting started tips
    st.markdown("## 💡 Getting Started Tips")
    
    with st.expander("🔰 For Beginners"):
        st.markdown("""
        1. **Start Simple**: Use the sample datasets to familiarize yourself with the interface
        2. **Follow the Workflow**: Data Manager → Causal Discovery → Validation Suite
        3. **Use Interactive Q&A**: Ask natural language questions about your data
        4. **Check Documentation**: Each page has helpful tooltips and explanations
        """)
    
    with st.expander("⚡ For Advanced Users"):
        st.markdown("""
        1. **Custom Analysis**: Configure advanced settings in the Settings page
        2. **Multiple Methods**: Compare results across different causal discovery algorithms
        3. **Sensitivity Analysis**: Use the validation suite for robust analysis
        4. **Temporal Modeling**: Analyze time-series causal relationships
        """)
    
    with st.expander("🎯 Pro Tips"):
        st.markdown("""
        1. **Data Quality**: Always check data quality before analysis
        2. **Domain Knowledge**: Use domain templates for better results
        3. **Validation**: Never skip assumption validation
        4. **Visualization**: Create publication-ready graphs in the Visualization studio
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Ready to start your causal analysis journey? Begin with the <strong>Data Manager</strong> or try our <strong>Interactive Q&A</strong>!</p>
    </div>
    """, unsafe_allow_html=True)