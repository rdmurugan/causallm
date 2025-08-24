# 🧠 CausalLLM Pro - Streamlit Application

## Phase 1 Implementation Complete ✅

This directory contains a comprehensive multi-page Streamlit application that transforms the CausalLLM library into a user-friendly UI-based service.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- All dependencies from `requirements.txt`
- Valid LLM API key (OpenAI, Anthropic, etc.)

### Running the Application

```bash
# From the project root directory
./run_app.sh

# Or manually:
cd streamlit_app
streamlit run main.py
```

The application will open in your default browser at `http://localhost:8501`

## 📊 Application Structure

### Core Pages

1. **🏠 Home** (`pages/home.py`)
   - Welcome dashboard with feature overview
   - Sample dataset generation and loading
   - Quick start guide
   - Feature statistics and capabilities

2. **📊 Data Manager** (`pages/data_manager.py`)
   - Multi-format file upload (CSV, Excel, JSON, Parquet)
   - Data quality assessment and validation
   - Variable role assignment (treatment, outcome, confounders)
   - Statistical summaries and visualizations
   - Data export capabilities

3. **🔍 Causal Discovery** (`pages/causal_discovery.py`)
   - AI-powered causal structure discovery
   - Multiple discovery methods (LLM-guided, PC Algorithm, etc.)
   - Interactive graph visualization with NetworkX/Plotly
   - Confidence scoring and validation
   - Comprehensive results analysis

4. **💬 Interactive Q&A** (`pages/interactive_qa.py`)
   - Natural language causal question interface
   - Domain-specific question templates
   - Conversation history and context
   - Q&A analytics and insights
   - Custom question creation

5. **✅ Validation Suite** (`pages/validation_suite.py`)
   - Comprehensive assumption checking
   - Statistical tests and validations
   - Causal argument validation using Bradford Hill criteria
   - Sensitivity analysis guidance
   - Effect size interpretation

6. **⏱️ Temporal Analysis** (`pages/temporal_analysis.py`)
   - Time-series causal modeling
   - Temporal relationship discovery
   - Intervention timing optimization
   - Forecasting and scenario analysis
   - Dynamic causal networks

7. **⚡ Intervention Optimizer** (`pages/intervention_optimizer.py`)
   - Optimal intervention strategy design
   - Effect estimation and comparison
   - Cost-benefit analysis
   - Population-specific recommendations
   - Sensitivity analysis integration

8. **📈 Visualization** (`pages/visualization.py`)
   - Professional causal graph creation
   - Multiple visualization types (DAGs, path diagrams, etc.)
   - Custom diagram builder
   - Export capabilities (PNG, PDF, SVG)
   - Interactive graph manipulation

9. **📊 Analytics** (`pages/analytics.py`)
   - Usage pattern tracking
   - Performance monitoring
   - Analysis trend insights
   - Comprehensive reporting
   - Dashboard customization

10. **⚙️ Settings** (`pages/settings.py`)
    - LLM configuration and API management
    - Analysis parameter defaults
    - UI preferences and themes
    - Data management settings
    - Advanced system configuration

### Navigation & Architecture

- **Main App** (`main.py`): Central router with sidebar navigation
- **Pages Module** (`pages/__init__.py`): Organized page imports
- **Shared State**: Session-based data and configuration management
- **Error Handling**: Comprehensive error management and user feedback

## 🎯 Key Features

### 🤖 AI Integration
- Multiple LLM provider support (OpenAI, Anthropic, etc.)
- Intelligent causal reasoning and discovery
- Natural language query processing
- Automated report generation

### 📊 Comprehensive Analysis
- Full causal inference pipeline
- 10+ statistical methods and tests
- Assumption validation and checking
- Effect size interpretation
- Sensitivity analysis

### 🎨 Professional UI
- Modern, responsive design
- Interactive visualizations with Plotly
- Intuitive navigation and workflows
- Comprehensive documentation
- Export capabilities

### ⚡ Performance
- Efficient data handling
- Caching and optimization
- Background processing support
- Real-time updates
- Session persistence

## 📈 Usage Patterns

### Typical Workflow
1. **Data Upload**: Load datasets via Data Manager
2. **Variable Assignment**: Define treatment, outcome, and confounder roles
3. **Causal Discovery**: Discover causal relationships using AI
4. **Validation**: Validate assumptions and check robustness
5. **Optimization**: Design optimal intervention strategies
6. **Visualization**: Create publication-ready graphs
7. **Analysis**: Review analytics and generate reports

### Advanced Features
- **Temporal Analysis**: Time-series causal modeling
- **Interactive Q&A**: Natural language queries
- **Custom Visualization**: Build specialized diagrams
- **Analytics Dashboard**: Monitor usage and performance

## 🔧 Configuration

### LLM Setup
Configure your LLM provider in Settings:
- OpenAI: Requires API key
- Anthropic: Requires API key
- Local models: Configure endpoint

### Data Settings
- File size limits and formats
- Privacy and anonymization
- Backup and retention policies
- Export preferences

### UI Customization
- Themes (light/dark)
- Layout preferences
- Chart styling
- Accessibility options

## 📊 Analytics & Monitoring

### Built-in Analytics
- Session tracking and usage patterns
- Analysis success rates
- Performance metrics
- Error monitoring
- Feature usage statistics

### Reporting
- Automated report generation
- Custom dashboard creation
- Export capabilities (PDF, Excel, CSV)
- Scheduled reports

## 🚀 Phase 2 Roadmap

The current implementation represents **Phase 1** of the SaaS transformation. Future phases include:

### Phase 2: FastAPI Backend
- RESTful API architecture
- Database integration
- User authentication
- Advanced caching

### Phase 3: SaaS Features
- Multi-tenancy
- User management
- Subscription handling
- Advanced analytics

### Phase 4: Production Deployment
- Cloud infrastructure
- Auto-scaling
- Monitoring and logging
- CI/CD pipelines

### Phase 5: Enterprise Features
- Team collaboration
- Advanced security
- Custom integrations
- White-label solutions

## 🛠️ Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **NetworkX**: Graph analysis
- **CausalLLM**: Core causal inference engine

### Architecture
- **Modular Design**: Each page is a separate module
- **State Management**: Streamlit session state for persistence
- **Error Handling**: Comprehensive exception management
- **Responsive UI**: Adapts to different screen sizes

### Performance Optimizations
- **Caching**: Strategic use of Streamlit caching
- **Lazy Loading**: Components load on demand
- **Efficient Data Handling**: Optimized pandas operations
- **Background Processing**: Non-blocking operations

## 🎯 Getting Started Tips

### First-Time Users
1. Start with the **Home** page to understand capabilities
2. Load sample data from the **Data Manager**
3. Try **Causal Discovery** with default settings
4. Explore **Interactive Q&A** for natural language queries
5. Review **Analytics** to understand your usage

### Advanced Users
1. Configure LLM settings for optimal performance
2. Customize analysis defaults for your domain
3. Use **Temporal Analysis** for time-series data
4. Leverage **Intervention Optimizer** for decision making
5. Create custom visualizations in **Visualization** studio

### Power Users
1. Set up automated reporting in **Analytics**
2. Configure advanced settings for your workflow
3. Use the full validation suite for robust analysis
4. Export results in multiple formats
5. Integrate with external tools via data exports

## 📞 Support & Documentation

- **In-App Help**: Available in the sidebar of each page
- **Interactive Tooltips**: Hover over elements for guidance
- **Sample Data**: Built-in datasets for testing
- **Error Messages**: Detailed error information and suggestions

---

**CausalLLM Pro v2.0** | Built with ❤️ using Streamlit | [GitHub Repository](https://github.com/rdmurugan/causallm)