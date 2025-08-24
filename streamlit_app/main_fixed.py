import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import page modules
import importlib.util

def load_page_module(module_name):
    """Load a page module dynamically"""
    module_path = Path(__file__).parent / "page_modules" / f"{module_name}.py"
    if module_path.exists():
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            # Create an error module
            error_module = type('ErrorModule', (), {})()
            error_module.show = lambda: show_error_page(module_name, str(e))
            return error_module
    return None

def show_error_page(module_name, error):
    """Show error page when module fails to load"""
    st.error(f"❌ Failed to load {module_name} page")
    
    # Handle specific common errors
    if "JSON serializable" in error:
        st.error("🔧 **JSON Serialization Error**: This is likely due to pandas data types in Plotly charts")
        st.markdown("""
        **Common causes:**
        - Using pandas dtypes in chart labels
        - Object columns in DataFrames passed to Plotly
        - Mixed data types in chart data
        
        **This has been fixed in the latest version.**
        """)
    else:
        st.code(error)
    
    st.markdown("### 🔧 Troubleshooting Steps")
    st.markdown("""
    1. **Refresh the page** (Ctrl+F5 or Cmd+Shift+R)
    2. **Check Dependencies**: Ensure all required packages are installed
    3. **Verify CausalLLM**: Run `python -c "import causalllm.core"`
    4. **Restart App**: Restart Streamlit if issues persist
    """)

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="CausalLLM Pro - AI-Powered Causal Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load all page modules
home = load_page_module('home')
home_safe = load_page_module('home_safe') 
data_manager = load_page_module('data_manager')
causal_discovery = load_page_module('causal_discovery')
interactive_qa = load_page_module('interactive_qa')
validation_suite = load_page_module('validation_suite')
temporal_analysis = load_page_module('temporal_analysis')
intervention_optimizer = load_page_module('intervention_optimizer')
visualization = load_page_module('visualization')
analytics = load_page_module('analytics')
settings = load_page_module('settings')
test_minimal = load_page_module('test_minimal')

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Main navigation - Use home_safe as fallback if home fails
effective_home = home if home is not None else home_safe

pages = {
    "🏠 Home": {"module": effective_home, "description": "Welcome to CausalLLM Pro"},
    "📊 Data Manager": {"module": data_manager, "description": "Upload, validate, and manage datasets"},
    "🔍 Causal Discovery": {"module": causal_discovery, "description": "Automated causal structure discovery"},
    "💬 Interactive Q&A": {"module": interactive_qa, "description": "Natural language causal analysis"},
    "✅ Validation Suite": {"module": validation_suite, "description": "Comprehensive assumption validation"},
    "⏱️ Temporal Analysis": {"module": temporal_analysis, "description": "Time-series causal modeling"},
    "🎯 Intervention Optimizer": {"module": intervention_optimizer, "description": "Optimal intervention design"},
    "📈 Visualization": {"module": visualization, "description": "Professional causal graph visualization"},
    "📊 Analytics": {"module": analytics, "description": "Usage analytics and insights"},
    "⚙️ Settings": {"module": settings, "description": "Configuration and preferences"}
}

# Only add debug page if we're in debug mode
if st.sidebar.checkbox("🔧 Debug Mode", value=False):
    pages["🧪 Debug Test"] = {"module": test_minimal, "description": "Debug test page"}

# Sidebar navigation
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
        <h1 style='color: white; margin: 0;'>🧠 CausalLLM Pro</h1>
        <p style='color: #e0e6ff; margin: 0; font-size: 0.9rem;'>AI-Powered Causal Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    selected_page = st.selectbox(
        "Navigate to:",
        list(pages.keys()),
        index=0,  # Always default to Home
        help="Select a page to navigate to"
    )
    
    st.markdown("---")
    
    # Quick stats
    if 'session_stats' in st.session_state:
        st.markdown("### 📊 Session Stats")
        stats = st.session_state.session_stats
        st.metric("Analyses Run", stats.get('analyses', 0))
        st.metric("Datasets Loaded", stats.get('datasets', 0))
        st.metric("Success Rate", f"{stats.get('success_rate', 0):.1f}%")
    
    st.markdown("---")
    
    # Help section
    with st.expander("🆘 Help & Support"):
        st.markdown("""
        - **Quick Start**: Begin with Data Manager
        - **Sample Data**: Available in Home page  
        - **Support**: Check troubleshooting guide
        - **GitHub**: [CausalLLM Repository](https://github.com/rdmurugan/causallm)
        """)

# Main content area
st.info(f"📍 {pages[selected_page]['description']}")

# Show module loading status
module_status = {}
for page_name, page_info in pages.items():
    module = page_info["module"]
    if module is None:
        module_status[page_name] = "❌ Failed"
    elif hasattr(module, 'show'):
        module_status[page_name] = "✅ Ready"
    else:
        module_status[page_name] = "⚠️ No show()"

# Debug info
with st.expander("🔍 Module Status (Debug)"):
    for page_name, status in module_status.items():
        st.write(f"{page_name}: {status}")

# Route to selected page
try:
    page_module = pages[selected_page]["module"]
    
    if page_module is None:
        st.error(f"❌ {selected_page} module is not available")
        st.info("This page failed to load. Check the debug information above.")
    elif not hasattr(page_module, 'show'):
        st.error(f"❌ {selected_page} module missing show() function") 
        st.info("This is a development error - the page module is malformed.")
    else:
        # Successfully load the page
        page_module.show()
        
except Exception as e:
    st.error(f"❌ Error loading {selected_page}")
    st.exception(e)
    
    st.markdown("### 🔧 Try These Solutions:")
    st.markdown("""
    1. **Refresh the page** (Ctrl+F5 or Cmd+Shift+R)
    2. **Try a different page** from the navigation
    3. **Check the Debug Test page** to verify basic functionality
    4. **Restart the Streamlit server**
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>CausalLLM Pro v2.0 | Built with ❤️ using Streamlit | 
    <a href='https://github.com/rdmurugan/causallm' target='_blank'>Open Source</a></p>
</div>
""", unsafe_allow_html=True)