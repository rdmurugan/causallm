# CausalLLM Documentation Index

**Complete documentation guide for CausalLLM - High-Performance Causal Inference Library**

---

## 📋 Documentation Overview

This documentation provides comprehensive coverage of CausalLLM's capabilities, featuring **standardized interfaces**, **centralized configuration management**, and advanced performance optimizations.

### 🎯 Quick Navigation

| Need | Documentation | Description |
|------|---------------|-------------|
| **Get Started** | [README](../README.md) | Main introduction with **CLI & Web Interface** |
| **CLI Interface** ⭐ | [CLI Usage Guide](CLI_USAGE.md) | **Command-line tool for terminal-based analysis** |
| **Web Interface** ⭐ | [Web Interface Guide](WEB_INTERFACE.md) | **Interactive point-and-click analysis** |
| **Configuration** ⭐ | [API Reference - Config](API_REFERENCE.md#configuration-management) | **New centralized configuration system** |
| **Learn the API** | [API Reference](API_REFERENCE.md) | Complete API with **standardized parameters** |
| **Master the Library** | [Complete User Guide](COMPLETE_USER_GUIDE.md) | In-depth guide with **configuration management** |
| **Optimize Performance** | [Performance Guide](PERFORMANCE_GUIDE.md) | **Enhanced async interfaces** and optimization |
| **Domain Expertise** | [Domain Packages](DOMAIN_PACKAGES.md) | **Standardized domain interfaces** |
| **Real Examples** | [Usage Examples](USAGE_EXAMPLES.md) | Real-world use cases with **new parameter names** |
| **Quick Marketing Start** | [Marketing Quick Reference](MARKETING_QUICK_REFERENCE.md) | Fast marketing attribution guide |
| **Advanced Integration** | [MCP Usage Guide](MCP_USAGE.md) | Model Context Protocol integration |
| **Enhancement Summary** ⭐ | [Enhancements Summary](ENHANCEMENTS_SUMMARY.md) | **Complete enhancement overview** |
| **Code Examples** | [Examples Directory](../examples/) | Runnable Python scripts |

### ⭐ **New Features Highlighted**

- **🖥️ Command Line Interface**: Terminal-based causal analysis with `causallm` command
- **🌐 Interactive Web Interface**: Point-and-click analysis with Streamlit (no Python required)
- **📊 Multiple Access Methods**: CLI, Web UI, and Python library for different user preferences
- **🎯 Standardized Interfaces**: Consistent parameter names (`data`, `treatment_variable`, `outcome_variable`) across all components
- **⚙️ Centralized Configuration**: Environment variables, JSON configuration files, and automatic configuration loading
- **🚀 Enhanced Async Support**: Unified async interfaces with identical parameters and configuration-driven optimization
- **📊 Rich Metadata**: Comprehensive analysis metadata, execution tracking, and performance monitoring
- **🔧 Protocol-Based Design**: Type-safe interfaces with standardized method signatures
- **⚡ Performance Optimization**: Configuration-driven performance tuning with intelligent caching and chunking

---

## 🏃‍♂️ Getting Started Path

### 1. **First Time Users** → Start Here
- **[README.md](../README.md)** - Overview with **CLI & Web Interface** introduction
- **[CLI Usage Guide](CLI_USAGE.md)** - **Terminal-based analysis** (no Python required)
- **[Web Interface Guide](WEB_INTERFACE.md)** - **Point-and-click analysis** (no Python required)

### 2. **Choose Your Interface** ⭐
- **[CLI Interface](CLI_USAGE.md)** - Perfect for data scientists and terminal users
- **[Web Interface](WEB_INTERFACE.md)** - Perfect for business users and visual analysis
- **[Python Library](COMPLETE_USER_GUIDE.md)** - Full programmatic control

### 3. **Learn the New Features** ⭐
- **[Enhancements Summary](ENHANCEMENTS_SUMMARY.md)** - Complete overview of **interface standardization** and **configuration management**  
- **[API Reference - Configuration](API_REFERENCE.md#configuration-management)** - **New centralized configuration system**
- **[Complete User Guide - Standardized Interfaces](COMPLETE_USER_GUIDE.md#standardized-interfaces)** - **Consistent parameter naming**

### 4. **Learn the Fundamentals**
- **[Complete User Guide](COMPLETE_USER_GUIDE.md)** - Comprehensive tutorial with **configuration management**
- **[API Reference](API_REFERENCE.md)** - Detailed method documentation with **standardized interfaces**

### 5. **Domain-Specific Applications**  
- **[Domain Packages Guide](DOMAIN_PACKAGES.md)** - Healthcare, Insurance, Marketing with **standardized interfaces**
- **[Marketing Quick Reference](MARKETING_QUICK_REFERENCE.md)** - Fast marketing attribution

### 6. **Advanced Usage**
- **[Performance Guide](PERFORMANCE_GUIDE.md)** - **Enhanced async interfaces** and large dataset handling
- **[MCP Usage Guide](MCP_USAGE.md)** - Integration with Claude Desktop, VS Code

---

## 📚 Documentation Categories

### User Interfaces (No Python Required) ⭐

#### 🖥️ [CLI Usage Guide](CLI_USAGE.md)
**Complete command-line interface documentation**

- **Installation & Setup**: Getting the CLI working
- **Command Reference**: All `causallm` commands with examples
- **Data Formats**: CSV/JSON input and output options
- **Domain Contexts**: Healthcare, marketing, education analysis
- **Advanced Usage**: Batch processing, configuration files
- **Troubleshooting**: Common issues and solutions

**Best for**: Data scientists, analysts, and users comfortable with terminal/command-line tools.

#### 🌐 [Web Interface Guide](WEB_INTERFACE.md) 
**Interactive point-and-click analysis documentation**

- **Interface Overview**: Navigation and main features
- **Data Upload**: Drag-and-drop files, sample datasets
- **Visual Analysis**: Interactive graphs and causal discovery
- **Guided Workflow**: Step-by-step analysis process
- **Export Options**: Download results, visualizations, reports
- **Configuration**: Settings, LLM setup, performance options

**Best for**: Business users, researchers, and anyone who prefers visual interfaces over coding.

### Core Library Documentation

#### 🔧 [API Reference](API_REFERENCE.md)
**Complete technical reference for all classes and methods**

- **Core Classes**: EnhancedCausalLLM, CausalLLM
- **Statistical Methods**: StatisticalCausalInference, EnhancedCausalDiscovery  
- **Domain Packages**: HealthcareDomain, InsuranceDomain, MarketingDomain
- **Performance Classes**: AsyncCausalAnalysis, DataChunker, StreamingDataProcessor
- **Data Models**: Results objects and return types
- **Error Handling**: Exception classes and troubleshooting

**Best for**: Developers who need detailed parameter information, return types, and method signatures.

#### 📖 [Complete User Guide](COMPLETE_USER_GUIDE.md)
**Comprehensive guide covering all features with practical examples**

- **Installation & Setup**: Environment configuration and API keys
- **Core Concepts**: Statistical methods and causal reasoning
- **Method Documentation**: Every API call with parameters and examples
- **Best Practices**: Data preparation and analysis guidelines
- **Troubleshooting**: Common errors and solutions
- **Advanced Features**: Custom LLM configuration and batch processing

**Best for**: Users who want to understand the library thoroughly with practical guidance.

### Performance & Optimization

#### ⚡ [Performance Guide](PERFORMANCE_GUIDE.md)
**Optimize CausalLLM for speed and memory efficiency**

- **Data Chunking**: Handle datasets larger than memory
- **Intelligent Caching**: 20x+ speedup for repeated analyses
- **Vectorized Algorithms**: 10x faster computations with Numba
- **Async Processing**: Parallel execution of independent tasks
- **Memory Management**: Monitor and optimize resource usage
- **Benchmarking Tools**: Built-in performance measurement

**Best for**: Users working with large datasets or needing maximum performance.

### Domain-Specific Guides

#### 🏭 [Domain Packages Guide](DOMAIN_PACKAGES.md)
**Industry-specific components with expert knowledge**

- **Healthcare Domain**: Clinical trials, treatment effectiveness, patient outcomes
- **Insurance Domain**: Risk assessment, premium optimization, claims analysis
- **Marketing Domain**: Campaign attribution, ROI optimization, customer analytics
- **Custom Domains**: How to create your own domain packages
- **Architecture**: Understanding data generators, knowledge bases, and templates

**Best for**: Users working in specific industries who want pre-configured expertise.

#### 📈 [Marketing Quick Reference](MARKETING_QUICK_REFERENCE.md)
**Fast-start guide for marketing attribution**

- **Attribution Models**: First-touch, last-touch, data-driven, Shapley
- **Quick Setup**: Get results in minutes
- **Common Use Cases**: Multi-touch attribution, campaign ROI, cross-device tracking
- **Performance Tips**: Handle large marketing datasets efficiently
- **Troubleshooting**: Common issues and solutions

**Best for**: Marketing professionals who need attribution analysis quickly.

### Examples & Applications

#### 📚 [Usage Examples](USAGE_EXAMPLES.md)
**Real-world use cases across different domains**

- **Healthcare**: Treatment effectiveness and clinical decision support
- **Marketing**: Campaign attribution and budget optimization
- **Finance**: Investment impact and risk analysis  
- **Education**: Learning interventions and student outcomes
- **E-commerce**: Recommendation systems and personalization

**Best for**: Understanding how CausalLLM applies to different business problems.

#### 💡 [Examples Directory](../examples/)
**Runnable Python scripts and tutorials**

- **Quick Start Examples**: Get up and running immediately
- **Domain-Specific Scripts**: Healthcare, marketing, and insurance examples
- **Performance Demonstrations**: Large dataset handling
- **Integration Examples**: MCP server setup and usage

**Best for**: Hands-on learning with executable code.

### Advanced Integration

#### 🔗 [MCP Usage Guide](MCP_USAGE.md)
**Model Context Protocol integration for advanced workflows**

- **Claude Desktop Integration**: Use CausalLLM tools in Claude Desktop
- **VS Code Integration**: CausalLLM within development environment
- **Available Tools**: Counterfactual analysis, treatment effects, causal reasoning
- **Setup Instructions**: Step-by-step configuration
- **Troubleshooting**: Common integration issues

**Best for**: Users who want to integrate CausalLLM with other AI tools and applications.

---

## 🎓 Learning Paths

### Path 1: CLI Quick Start (15 minutes) ⭐
1. Install with `pip install causallm`
2. Follow [CLI Usage Guide](CLI_USAGE.md) quick start
3. Run your first analysis: `causallm discover --data sample.csv --variables "var1,var2,var3"`

### Path 2: Web Interface Quick Start (10 minutes) ⭐  
1. Install with `pip install "causallm[ui]"`
2. Launch with `causallm web`
3. Follow [Web Interface Guide](WEB_INTERFACE.md) getting started workflow

### Path 3: Python Library Quick Start (30 minutes)
1. Read [README.md](../README.md) overview and installation
2. Run a quick example from [Examples Directory](../examples/)
3. Check [Marketing Quick Reference](MARKETING_QUICK_REFERENCE.md) for immediate application

### Path 4: Comprehensive Learning (2-3 hours)
1. Choose your interface: [CLI](CLI_USAGE.md), [Web](WEB_INTERFACE.md), or [Python](COMPLETE_USER_GUIDE.md)
2. Explore relevant [Domain Package](DOMAIN_PACKAGES.md) for your industry
3. Review [Usage Examples](USAGE_EXAMPLES.md) for your use case
4. Reference [API Documentation](API_REFERENCE.md) as needed

### Path 5: Performance Optimization (1 hour)
1. Read [Performance Guide](PERFORMANCE_GUIDE.md) optimization strategies
2. Configure your setup for large datasets
3. Run performance benchmarks on your data
4. Implement caching and async processing

### Path 6: Advanced Integration (1-2 hours)
1. Set up [MCP Integration](MCP_USAGE.md) with Claude Desktop
2. Create custom domain packages using [Domain Packages Guide](DOMAIN_PACKAGES.md)
3. Implement custom statistical methods using [API Reference](API_REFERENCE.md)

---

## 🤝 Getting Help

### Documentation Issues
If you find documentation unclear or incomplete:
- **GitHub Issues**: [Report documentation issues](https://github.com/rdmurugan/causallm/issues) with 'documentation' label
- **GitHub Discussions**: [Ask questions](https://github.com/rdmurugan/causallm/discussions) about usage

### Technical Support
- **Performance Issues**: Tag GitHub issues with 'performance'
- **Domain Questions**: Use 'domain-packages' label
- **API Questions**: Reference specific method names from [API Reference](API_REFERENCE.md)

### Community
- **Email**: durai@infinidatum.net
- **LinkedIn**: [Durai Rajamanickam](https://www.linkedin.com/in/durai-rajamanickam)
- **Discussions**: [GitHub Discussions](https://github.com/rdmurugan/causallm/discussions)

---

## 🔄 Documentation Updates

This documentation is actively maintained and updated. For the latest version:

- **GitHub**: Always reflects the current release
- **Version Info**: Check individual files for last updated dates
- **Changelog**: See [GitHub Releases](https://github.com/rdmurugan/causallm/releases) for updates

---

## 📄 Contributing to Documentation

Help improve CausalLLM documentation:

1. **Report Issues**: Found something unclear? [Create an issue](https://github.com/rdmurugan/causallm/issues)
2. **Suggest Improvements**: [Start a discussion](https://github.com/rdmurugan/causallm/discussions)
3. **Submit Changes**: Fork, improve, and submit pull requests
4. **Add Examples**: Contribute domain-specific examples and use cases

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.

---

**✨ Ready to master causal inference? Start with the [README](../README.md) and follow the learning path that matches your needs!**