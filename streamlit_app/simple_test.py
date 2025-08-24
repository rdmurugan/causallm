import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

st.set_page_config(
    page_title="Simple Test - CausalLLM",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 CausalLLM Simple Test")

# Test basic functionality
st.success("✅ Streamlit is working!")

# Test imports
st.markdown("### 📦 Testing Imports")

try:
    import pandas as pd
    import numpy as np
    import plotly.express as px
    st.success("✅ Basic packages (pandas, numpy, plotly) imported successfully")
except Exception as e:
    st.error(f"❌ Basic packages failed: {e}")

try:
    from causalllm.core import CausalLLMCore
    st.success("✅ CausalLLM core imported successfully")
except Exception as e:
    st.error(f"❌ CausalLLM core failed: {e}")

# Test page imports
st.markdown("### 📄 Testing Page Imports")

import importlib.util

def test_page_import(page_name):
    try:
        module_path = Path(__file__).parent / "page_modules" / f"{page_name}.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location(page_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, 'show'):
                return "✅ Loaded with show() function"
            else:
                return "⚠️ Loaded but missing show() function"
        else:
            return "❌ File not found"
    except Exception as e:
        return f"❌ Import error: {str(e)}"

pages_to_test = ['home', 'home_safe', 'data_manager', 'causal_discovery', 'interactive_qa', 'test_minimal']

for page in pages_to_test:
    result = test_page_import(page)
    st.write(f"**{page}**: {result}")

# Test navigation
st.markdown("### 🧭 Navigation Test")

page_choice = st.selectbox(
    "Test page loading:",
    ['home_safe', 'test_minimal', 'home']
)

if st.button("Load Selected Page"):
    try:
        module_path = Path(__file__).parent / "page_modules" / f"{page_choice}.py" 
        spec = importlib.util.spec_from_file_location(page_choice, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        st.success(f"✅ Loading {page_choice} page...")
        st.markdown("---")
        
        # Call the show function
        module.show()
        
    except Exception as e:
        st.error(f"❌ Failed to load {page_choice}: {e}")
        st.exception(e)