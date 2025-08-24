import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

st.title("🔍 Debug Test")

st.write("Testing page imports...")

# Test direct import
try:
    from pages import home
    st.success("✅ Direct import of pages.home works")
    
    if hasattr(home, 'show'):
        st.success("✅ home.show() function exists")
        
        # Try calling it
        try:
            home.show()
            st.success("✅ home.show() executed successfully")
        except Exception as e:
            st.error(f"❌ Error calling home.show(): {e}")
            st.exception(e)
    else:
        st.error("❌ home.show() function missing")
        
except Exception as e:
    st.error(f"❌ Error importing pages.home: {e}")
    st.exception(e)

# Test fallback import
st.write("Testing fallback import mechanism...")

try:
    import importlib.util
    
    pages_dir = Path(__file__).parent / "pages"
    module_path = pages_dir / "data_manager.py"
    
    if module_path.exists():
        spec = importlib.util.spec_from_file_location("data_manager", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        st.success("✅ Fallback import of data_manager works")
        
        if hasattr(module, 'show'):
            st.success("✅ data_manager.show() function exists")
            
            try:
                module.show()
                st.success("✅ data_manager.show() executed successfully")
            except Exception as e:
                st.error(f"❌ Error calling data_manager.show(): {e}")
                st.exception(e)
        else:
            st.error("❌ data_manager.show() function missing")
    else:
        st.error("❌ data_manager.py not found")
        
except Exception as e:
    st.error(f"❌ Error with fallback import: {e}")
    st.exception(e)