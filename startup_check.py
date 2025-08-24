#!/usr/bin/env python3
"""
Startup check script for CausalLLM Streamlit app
Verifies all dependencies and imports before starting the app
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'networkx',
        'scipy'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True

def check_causallm_imports():
    """Check if CausalLLM library imports work"""
    print("\n🔍 Checking CausalLLM imports...")
    
    critical_imports = [
        'causalllm.core',
        'causalllm.llm_client', 
        'causalllm.data_manager',
        'causalllm.llm_causal_discovery',
        'causalllm.interactive_causal_qa',
        'causalllm.assumption_checker'
    ]
    
    failed = []
    for import_path in critical_imports:
        try:
            __import__(import_path)
            print(f"✅ {import_path}")
        except ImportError as e:
            print(f"❌ {import_path} - {e}")
            failed.append(import_path)
    
    if failed:
        print(f"\n⚠️  CausalLLM import failures: {len(failed)}")
        print("This may cause some pages to show error messages.")
        return False
        
    return True

def check_streamlit_pages():
    """Check if all Streamlit pages can be loaded"""
    print("\n🔍 Checking Streamlit page files...")
    
    pages_dir = Path(__file__).parent / "streamlit_app" / "page_modules"
    if not pages_dir.exists():
        print(f"❌ Pages directory not found: {pages_dir}")
        return False
    
    required_pages = [
        'home.py',
        'data_manager.py', 
        'causal_discovery.py',
        'interactive_qa.py',
        'validation_suite.py',
        'temporal_analysis.py',
        'intervention_optimizer.py',
        'visualization.py',
        'analytics.py',
        'settings.py'
    ]
    
    missing = []
    for page in required_pages:
        page_path = pages_dir / page
        if page_path.exists():
            print(f"✅ {page}")
        else:
            print(f"❌ {page} - FILE MISSING") 
            missing.append(page)
    
    if missing:
        print(f"\n⚠️  Missing page files: {len(missing)}")
        return False
        
    return True

def main():
    print("🧠 CausalLLM Pro - Startup Check")
    print("=" * 50)
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    sys.path.insert(0, str(script_dir))
    
    checks = [
        ("Dependencies", check_dependencies),
        ("CausalLLM Library", check_causallm_imports),
        ("Streamlit Pages", check_streamlit_pages)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All checks passed! Ready to start CausalLLM Pro")
        print("\nTo start the app, run:")
        print("cd streamlit_app && streamlit run main.py")
    else:
        print("⚠️  Some checks failed. The app may have issues.")
        print("Check the error messages above and fix any problems.")
        print("\nYou can still try to start the app - pages with errors will show troubleshooting info.")
    
    return all_passed

if __name__ == "__main__":
    main()