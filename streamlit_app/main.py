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
            # Store error info for later display
            error_module = type('ErrorModule', (), {})()
            error_module.show = lambda: show_error_page(module_name, str(e))
            return error_module
    return None

def show_error_page(module_name, error):
    """Show error page when module fails to load"""
    st.error(f"❌ Failed to load {module_name} page")
    st.exception(Exception(error))
    
    st.markdown("### 🔧 Troubleshooting")
    st.markdown(f"""
    The **{module_name}** page failed to load due to an import or initialization error.
    
    **Common causes:**
    - Missing dependencies
    - Import errors in CausalLLM library
    - Configuration issues
    
    **Solutions:**
    1. Check that all required packages are installed
    2. Verify CausalLLM library is working: `python -c "import causalllm.core"`
    3. Check the error details above
    4. Try refreshing the page
    """)
    
    if st.button("🔄 Retry Loading"):
        st.rerun()

# Load all page modules  
test_minimal = load_page_module('test_minimal')
home_safe = load_page_module('home_safe')
home = load_page_module('home')
data_manager = load_page_module('data_manager')
causal_discovery = load_page_module('causal_discovery')
interactive_qa = load_page_module('interactive_qa')
validation_suite = load_page_module('validation_suite')
temporal_analysis = load_page_module('temporal_analysis')
intervention_optimizer = load_page_module('intervention_optimizer')
visualization = load_page_module('visualization')
analytics = load_page_module('analytics')
settings = load_page_module('settings')

# Page configuration
st.set_page_config(
    page_title="CausalLLM Pro - AI-Powered Causal Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Main navigation
pages = {
    "🏠 Home": {"module": home_safe if home is None else home, "description": "Welcome to CausalLLM Pro"},
    "📊 Data Manager": {"module": data_manager, "description": "Upload, validate, and manage datasets"},
    "🔍 Causal Discovery": {"module": causal_discovery, "description": "Automated causal structure discovery"},
    "💬 Interactive Q&A": {"module": interactive_qa, "description": "Natural language causal analysis"},
    "✅ Validation Suite": {"module": validation_suite, "description": "Comprehensive assumption validation"},
    "⏱️ Temporal Analysis": {"module": temporal_analysis, "description": "Time-series causal modeling"},
    "🎯 Intervention Optimizer": {"module": intervention_optimizer, "description": "Optimal intervention design"},
    "📈 Visualization": {"module": visualization, "description": "Professional causal graph visualization"},
    "📊 Analytics": {"module": analytics, "description": "Usage analytics and insights"},
    "⚙️ Settings": {"module": settings, "description": "Configuration and preferences"},
    "🧪 Debug Test": {"module": test_minimal, "description": "Debug test page (for troubleshooting)"}
}

# Sidebar navigation
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h1 style='color: white; margin: 0;'>🧠 CausalLLM Pro</h1>
        <p style='color: #e0e6ff; margin: 0; font-size: 0.9rem;'>AI-Powered Causal Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    selected_page = st.selectbox(
        "Navigate to:",
        list(pages.keys()),
        index=0,  # Default to first page (Home)
        format_func=lambda x: x,
        help="Select a page to navigate to"
    )
    
    st.markdown("---")
    
    # Quick stats (if available)
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
        - **Documentation**: [docs.causallm.ai](https://docs.causallm.ai)
        - **Tutorials**: [tutorials.causallm.ai](https://tutorials.causallm.ai)  
        - **Support**: support@causallm.ai
        - **GitHub**: [github.com/rdmurugan/causallm](https://github.com/rdmurugan/causallm)
        """)

# Display selected page description
st.info(f"📍 {pages[selected_page]['description']}")

# Debug information (can be removed later)
if st.checkbox("🔍 Show Debug Info", value=False):
    st.write("**Current page:**", selected_page)
    st.write("**Module status:**")
    for page_name, page_info in pages.items():
        module = page_info["module"]
        if module is None:
            status = "❌ None"
        elif hasattr(module, 'show'):
            status = "✅ Ready"
        else:
            status = "⚠️ Missing show()"
        st.write(f"- {page_name}: {status}")
    st.write("**Selected module:**", type(pages[selected_page]["module"]))

# Route to selected page
try:
    page_module = pages[selected_page]["module"]
    if page_module is None:
        st.error(f"❌ Module for {selected_page} is not available")
        st.info("This page module failed to load during import")
        
        # Show debug info
        with st.expander("🔍 Debug Information"):
            st.write("**Page loading status:**")
            for page_name, page_info in pages.items():
                status = "✅ Loaded" if page_info["module"] is not None else "❌ Failed"
                st.write(f"- {page_name}: {status}")
    else:
        # Check if the module has the show function
        if hasattr(page_module, 'show'):
            page_module.show()
        else:
            st.error(f"❌ {selected_page} module loaded but missing show() function")
            st.info("This indicates a problem with the page implementation")
            
except Exception as e:
    st.error(f"❌ Error loading page {selected_page}: {str(e)}")
    st.exception(e)  # Show full traceback
    st.info("Please check the implementation of the selected page module.")
    
    # Show debug info
    with st.expander("🔍 Debug Information"):
        st.write("**Page loading status:**")
        for page_name, page_info in pages.items():
            module = page_info["module"]
            if module is None:
                status = "❌ Failed to load"
            elif hasattr(module, 'show'):
                status = "✅ Loaded with show()"
            else:
                status = "⚠️ Loaded but missing show()"
            st.write(f"- {page_name}: {status}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>CausalLLM Pro v2.0 | Built with ❤️ using Streamlit | 
    <a href='https://github.com/rdmurugan/causallm' target='_blank'>Open Source</a></p>
</div>
""", unsafe_allow_html=True)