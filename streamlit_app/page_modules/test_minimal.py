import streamlit as st

def show():
    st.title("🧪 Minimal Test Page")
    st.success("✅ This page loads successfully!")
    st.write("If you can see this, the page loading mechanism works correctly.")
    
    st.markdown("### Test Basic Functionality")
    st.write("Current timestamp:", st.session_state)
    
    if st.button("Test Button"):
        st.balloons()
        st.success("Button works!")
    
    st.markdown("### Test Chart")
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['a', 'b', 'c']
    )
    
    st.line_chart(df)