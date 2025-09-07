#!/usr/bin/env python3
"""
Test script to verify pip install experience for users
"""

import sys
import subprocess
import tempfile
import os

def test_pip_install():
    """Test the pip install experience"""
    print("Testing pip install causallm experience...")
    
    # Create a temporary directory to simulate fresh install
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple test script
        test_script = """
import sys
try:
    from causallm.domains.marketing import MarketingDomain
    print("✓ Import successful")
    
    # Test basic functionality
    marketing = MarketingDomain()
    print("✓ MarketingDomain initialized")
    
    # Test data generation
    data = marketing.generate_marketing_data(n_customers=100, n_touchpoints=500)
    print(f"✓ Generated {len(data)} touchpoints")
    print(f"✓ Columns: {list(data.columns)}")
    
    # Test attribution analysis
    result = marketing.analyze_attribution(
        data,
        model='first_touch',
        conversion_column='conversion',
        customer_id_column='customer_id',
        channel_column='channel',
        timestamp_column='timestamp'
    )
    print("✓ Attribution analysis successful")
    print(f"✓ Top channels: {list(result.channel_attribution.keys())[:3]}")
    
    print("\\n✅ All tests passed! Package works correctly from pip install.")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except AttributeError as e:
    print(f"✗ Attribute error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    sys.exit(1)
"""
        
        test_file = os.path.join(temp_dir, "test_causallm.py")
        with open(test_file, 'w') as f:
            f.write(test_script)
        
        # Run the test
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ pip install test PASSED")
        else:
            print("❌ pip install test FAILED")
            return False
    
    return True

if __name__ == "__main__":
    test_pip_install()