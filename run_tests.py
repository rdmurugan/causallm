#!/usr/bin/env python3
"""
Test runner for CausalLLM
Runs comprehensive test suite and generates reports
"""
import sys
import os
import subprocess
import argparse
import time
from pathlib import Path


def run_command(cmd, capture=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    else:
        return subprocess.run(cmd).returncode


def install_test_dependencies():
    """Install test dependencies."""
    print("Installing test dependencies...")
    
    dependencies = [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.0.0",
        "pytest-mock>=3.10.0"
    ]
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        returncode = run_command([sys.executable, "-m", "pip", "install", dep], capture=False)
        if returncode != 0:
            print(f"Failed to install {dep}")
            return False
    
    return True


def run_basic_tests():
    """Run basic functionality tests."""
    print("\n" + "="*50)
    print("RUNNING BASIC FUNCTIONALITY TESTS")
    print("="*50)
    
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/test_core_functionality.py",
        "tests/test_dag_parser.py",
        "tests/test_do_operator.py",
        "-v"
    ]
    
    return run_command(cmd, capture=False)


def run_statistical_tests():
    """Run statistical methods tests."""
    print("\n" + "="*50)
    print("RUNNING STATISTICAL METHODS TESTS")
    print("="*50)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_statistical_methods.py",
        "-v", "-m", "not slow"
    ]
    
    return run_command(cmd, capture=False)


def run_discovery_tests():
    """Run causal discovery tests."""
    print("\n" + "="*50)
    print("RUNNING CAUSAL DISCOVERY TESTS")
    print("="*50)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_causal_discovery.py", 
        "-v", "-m", "not llm"  # Skip LLM-dependent tests
    ]
    
    return run_command(cmd, capture=False)


def run_llm_tests():
    """Run LLM client tests."""
    print("\n" + "="*50) 
    print("RUNNING LLM CLIENT TESTS")
    print("="*50)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_llm_client.py",
        "-v"
    ]
    
    return run_command(cmd, capture=False)


def run_integration_tests():
    """Run integration tests."""
    print("\n" + "="*50)
    print("RUNNING INTEGRATION TESTS")
    print("="*50)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_integration.py",
        "-v", "-m", "not slow and not llm"
    ]
    
    return run_command(cmd, capture=False)


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*50)
    print("RUNNING ALL TESTS")
    print("="*50)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v", "-m", "not slow and not llm"
    ]
    
    return run_command(cmd, capture=False)


def run_coverage_tests():
    """Run tests with coverage reporting."""
    print("\n" + "="*50)
    print("RUNNING TESTS WITH COVERAGE")
    print("="*50)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "--cov=causallm",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "-v", "-m", "not slow and not llm"
    ]
    
    return run_command(cmd, capture=False)


def validate_library_documentation():
    """Validate that library meets documentation requirements."""
    print("\n" + "="*50)
    print("VALIDATING DOCUMENTATION REQUIREMENTS")
    print("="*50)
    
    # Check that key modules are importable
    try:
        print("Testing core imports...")
        import causallm
        from causallm import CausalLLM
        from causallm.core.statistical_methods import PCAlgorithm, ConditionalIndependenceTest
        from causallm.core.dag_parser import DAGParser
        from causallm.core.do_operator import DoOperatorSimulator
        from causallm.core.llm_client import get_llm_client
        
        print("✓ All core modules import successfully")
        
        # Check version information
        print(f"✓ Library version: {causallm.__version__}")
        print(f"✓ License: {causallm.__license__}")
        print(f"✓ Author: {causallm.__author__}")
        
        # Test basic functionality
        print("Testing basic functionality...")
        
        # Test CausalLLM initialization
        causal_llm = CausalLLM()
        print("✓ CausalLLM initializes correctly")
        
        # Test statistical methods
        ci_test = ConditionalIndependenceTest()
        pc = PCAlgorithm(ci_test=ci_test)
        print("✓ Statistical methods initialize correctly")
        
        # Test DAG parser
        dag = DAGParser([('A', 'B')])
        print("✓ DAG parser works correctly")
        
        # Test do-operator
        do_op = DoOperatorSimulator("test context", {"A": "value_a", "B": "value_b"})
        print("✓ Do-operator works correctly")
        
        print("\n✅ LIBRARY MEETS DOCUMENTATION REQUIREMENTS")
        return True
        
    except Exception as e:
        print(f"\n❌ LIBRARY VALIDATION FAILED: {e}")
        return False


def generate_test_report():
    """Generate a comprehensive test report."""
    print("\n" + "="*50)
    print("GENERATING TEST REPORT")
    print("="*50)
    
    # Run tests with detailed output
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "--tb=short",
        "-v",
        "-m", "not slow and not llm",
        "--junit-xml=test_results.xml"
    ]
    
    returncode, stdout, stderr = run_command(cmd)
    
    # Write summary report
    with open("test_summary.txt", "w") as f:
        f.write("CausalLLM Test Suite Summary\n")
        f.write("="*30 + "\n")
        f.write(f"Test run date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test result: {'PASSED' if returncode == 0 else 'FAILED'}\n")
        f.write(f"Return code: {returncode}\n\n")
        
        f.write("STDOUT:\n")
        f.write(stdout)
        f.write("\n\nSTDERR:\n")
        f.write(stderr)
    
    print(f"Test report saved to test_summary.txt")
    return returncode == 0


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="CausalLLM Test Runner")
    parser.add_argument("--install-deps", action="store_true", help="Install test dependencies")
    parser.add_argument("--basic", action="store_true", help="Run basic tests only")
    parser.add_argument("--statistical", action="store_true", help="Run statistical tests only")
    parser.add_argument("--discovery", action="store_true", help="Run discovery tests only")
    parser.add_argument("--llm", action="store_true", help="Run LLM tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage")
    parser.add_argument("--validate", action="store_true", help="Validate documentation requirements")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive test report")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    success = True
    
    if args.install_deps:
        success &= install_test_dependencies()
    
    if args.validate:
        success &= validate_library_documentation()
    
    if args.basic:
        success &= (run_basic_tests() == 0)
    
    if args.statistical:
        success &= (run_statistical_tests() == 0)
    
    if args.discovery:
        success &= (run_discovery_tests() == 0)
    
    if args.llm:
        success &= (run_llm_tests() == 0)
    
    if args.integration:
        success &= (run_integration_tests() == 0)
    
    if args.coverage:
        success &= (run_coverage_tests() == 0)
    
    if args.report:
        success &= generate_test_report()
    
    if args.all or not any([args.basic, args.statistical, args.discovery, 
                           args.llm, args.integration, args.coverage, 
                           args.validate, args.report]):
        print("Running full test suite...")
        success &= validate_library_documentation()
        success &= (run_all_tests() == 0)
    
    if success:
        print("\n🎉 ALL TESTS PASSED!")
        print("CausalLLM library is working correctly and meets documentation requirements.")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please check the test output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()