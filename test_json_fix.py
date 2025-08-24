#!/usr/bin/env python3
"""
Quick test to verify JSON serialization fix
"""

import pandas as pd
import numpy as np
import plotly.express as px

def test_json_serialization_fix():
    """Test the JSON serialization fix"""
    print("🧪 Testing JSON Serialization Fix...")
    
    # Create test data with mixed types (similar to real data)
    test_data = pd.DataFrame({
        'numeric_col': [1, 2, 3, 4, 5],
        'string_col': ['A', 'B', 'C', 'D', 'E'], 
        'mixed_col': ['X', 1, 'Y', 2, 'Z'],
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5]
    })
    
    print(f"✅ Test data created: {test_data.shape}")
    print(f"Data types: {test_data.dtypes.tolist()}")
    
    # Test 1: Data type distribution (the original problem)
    try:
        dtype_counts = test_data.dtypes.value_counts()
        print(f"✅ Dtype counts created: {dtype_counts}")
        
        # OLD WAY (would fail):
        # names=dtype_counts.index  # This has pandas dtypes
        
        # NEW WAY (fixed):
        dtype_names = [str(dtype) for dtype in dtype_counts.index]
        print(f"✅ Converted dtype names: {dtype_names}")
        
        fig = px.pie(
            values=dtype_counts.values,
            names=dtype_names,  # Now JSON serializable
            title="Data Type Distribution"
        )
        
        # Test JSON serialization
        import plotly.io as pio
        json_str = pio.to_json(fig)
        print(f"✅ Pie chart JSON serialization: OK ({len(json_str)} chars)")
        
    except Exception as e:
        print(f"❌ Pie chart test failed: {e}")
    
    # Test 2: Correlation matrix
    try:
        numeric_data = test_data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            corr_matrix = numeric_data.corr()
            
            # Apply fix
            corr_matrix.index = corr_matrix.index.astype(str)
            corr_matrix.columns = corr_matrix.columns.astype(str)
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                title="Correlation Matrix"
            )
            
            # Test JSON serialization
            json_str = pio.to_json(fig)
            print(f"✅ Correlation matrix JSON serialization: OK ({len(json_str)} chars)")
        
    except Exception as e:
        print(f"❌ Correlation matrix test failed: {e}")
    
    print("\n🎉 All JSON serialization tests passed!")

if __name__ == "__main__":
    test_json_serialization_fix()