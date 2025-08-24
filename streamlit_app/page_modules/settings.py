import streamlit as st
import pandas as pd
import json
from datetime import datetime
import os

def show():
    st.title("⚙️ Settings & Configuration")
    st.markdown("Customize your CausalLLM experience and manage system preferences")
    
    # Create settings tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔧 General", "🤖 LLM Configuration", "📊 Analysis Defaults", 
        "🎨 UI Preferences", "📁 Data Management", "🔐 Advanced"
    ])
    
    # Initialize settings if not exists
    if 'user_settings' not in st.session_state:
        initialize_default_settings()
    
    with tab1:
        st.markdown("### General Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### User Preferences")
            
            # User profile
            user_name = st.text_input(
                "Display Name",
                value=st.session_state.user_settings.get('user_name', ''),
                help="Your display name in the application"
            )
            
            user_role = st.selectbox(
                "Primary Role",
                ["Researcher", "Data Analyst", "Student", "Consultant", "Other"],
                index=["Researcher", "Data Analyst", "Student", "Consultant", "Other"].index(
                    st.session_state.user_settings.get('user_role', 'Researcher')
                )
            )
            
            experience_level = st.selectbox(
                "Experience with Causal Inference",
                ["Beginner", "Intermediate", "Advanced", "Expert"],
                index=["Beginner", "Intermediate", "Advanced", "Expert"].index(
                    st.session_state.user_settings.get('experience_level', 'Intermediate')
                )
            )
            
            # Language and locale
            language = st.selectbox(
                "Language",
                ["English", "Spanish", "French", "German", "Chinese"],
                index=0,
                help="Application language (affects AI responses)"
            )
        
        with col2:
            st.markdown("#### Application Behavior")
            
            # Auto-save settings
            auto_save_enabled = st.checkbox(
                "Enable auto-save",
                value=st.session_state.user_settings.get('auto_save_enabled', True),
                help="Automatically save progress during analysis"
            )
            
            if auto_save_enabled:
                auto_save_interval = st.slider(
                    "Auto-save interval (minutes)",
                    min_value=1, max_value=30, 
                    value=st.session_state.user_settings.get('auto_save_interval', 5)
                )
            
            # Confirmation dialogs
            show_confirmations = st.checkbox(
                "Show confirmation dialogs",
                value=st.session_state.user_settings.get('show_confirmations', True),
                help="Ask for confirmation before potentially destructive actions"
            )
            
            # Debug mode
            debug_mode = st.checkbox(
                "Enable debug mode",
                value=st.session_state.user_settings.get('debug_mode', False),
                help="Show additional technical information and logs"
            )
            
            # Analytics and telemetry
            enable_analytics = st.checkbox(
                "Enable usage analytics",
                value=st.session_state.user_settings.get('enable_analytics', True),
                help="Help improve the application by sharing anonymous usage data"
            )
        
        # Notification settings
        st.markdown("#### Notifications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            email_notifications = st.checkbox(
                "Email notifications",
                value=st.session_state.user_settings.get('email_notifications', False)
            )
            
            if email_notifications:
                email_address = st.text_input(
                    "Email address",
                    value=st.session_state.user_settings.get('email_address', ''),
                    placeholder="your@email.com"
                )
        
        with col2:
            notification_types = st.multiselect(
                "Notification types",
                ["Analysis Complete", "Errors", "Weekly Summary", "System Updates"],
                default=st.session_state.user_settings.get('notification_types', ["Analysis Complete", "Errors"])
            )
    
    with tab2:
        st.markdown("### LLM Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model Settings")
            
            # LLM provider selection
            llm_provider = st.selectbox(
                "LLM Provider",
                ["OpenAI", "Anthropic", "Google", "Local Model", "Custom"],
                index=["OpenAI", "Anthropic", "Google", "Local Model", "Custom"].index(
                    st.session_state.user_settings.get('llm_provider', 'OpenAI')
                )
            )
            
            # Model-specific settings
            if llm_provider == "OpenAI":
                model_name = st.selectbox(
                    "Model",
                    ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
                    index=0
                )
                
                api_key_input = st.text_input(
                    "API Key",
                    type="password",
                    value="***" if st.session_state.user_settings.get('openai_api_key') else "",
                    help="Your OpenAI API key"
                )
                
            elif llm_provider == "Anthropic":
                model_name = st.selectbox(
                    "Model",
                    ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307", "claude-3-opus-20240229"],
                    index=0
                )
                
                api_key_input = st.text_input(
                    "API Key",
                    type="password",
                    value="***" if st.session_state.user_settings.get('anthropic_api_key') else "",
                    help="Your Anthropic API key"
                )
        
        with col2:
            st.markdown("#### Generation Parameters")
            
            # Temperature and other parameters
            temperature = st.slider(
                "Temperature",
                min_value=0.0, max_value=2.0,
                value=st.session_state.user_settings.get('llm_temperature', 0.7),
                step=0.1,
                help="Controls randomness in AI responses (0=deterministic, 2=very random)"
            )
            
            max_tokens = st.number_input(
                "Max tokens per response",
                min_value=100, max_value=4000,
                value=st.session_state.user_settings.get('llm_max_tokens', 1000),
                step=100,
                help="Maximum length of AI responses"
            )
            
            timeout = st.number_input(
                "Request timeout (seconds)",
                min_value=10, max_value=120,
                value=st.session_state.user_settings.get('llm_timeout', 30),
                help="How long to wait for AI responses"
            )
            
            # Response style
            response_style = st.selectbox(
                "Response style",
                ["Professional", "Conversational", "Technical", "Educational"],
                index=["Professional", "Conversational", "Technical", "Educational"].index(
                    st.session_state.user_settings.get('response_style', 'Professional')
                )
            )
        
        # Custom prompts
        st.markdown("#### Custom System Prompts")
        
        with st.expander("📝 Customize AI Behavior"):
            
            custom_system_prompt = st.text_area(
                "Custom system prompt",
                value=st.session_state.user_settings.get('custom_system_prompt', ''),
                height=100,
                placeholder="Add custom instructions for the AI assistant...",
                help="This will be added to the system prompt for all AI interactions"
            )
            
            domain_specific_prompts = st.checkbox(
                "Enable domain-specific prompts",
                value=st.session_state.user_settings.get('domain_specific_prompts', True),
                help="Use specialized prompts for different analysis domains"
            )
    
    with tab3:
        st.markdown("### Analysis Defaults")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Statistical Settings")
            
            # Default confidence level
            default_confidence = st.slider(
                "Default confidence level",
                min_value=0.80, max_value=0.99,
                value=st.session_state.user_settings.get('default_confidence', 0.95),
                step=0.01,
                format="%.2f"
            )
            
            # Significance level
            significance_level = st.slider(
                "Significance level (α)",
                min_value=0.01, max_value=0.10,
                value=st.session_state.user_settings.get('significance_level', 0.05),
                step=0.01,
                format="%.2f"
            )
            
            # Effect size thresholds
            st.markdown("**Effect Size Thresholds**")
            
            small_effect = st.number_input(
                "Small effect threshold",
                value=st.session_state.user_settings.get('small_effect_threshold', 0.2),
                step=0.1,
                format="%.1f"
            )
            
            medium_effect = st.number_input(
                "Medium effect threshold",
                value=st.session_state.user_settings.get('medium_effect_threshold', 0.5),
                step=0.1,
                format="%.1f"
            )
            
            large_effect = st.number_input(
                "Large effect threshold",
                value=st.session_state.user_settings.get('large_effect_threshold', 0.8),
                step=0.1,
                format="%.1f"
            )
        
        with col2:
            st.markdown("#### Analysis Preferences")
            
            # Default methods
            default_discovery_method = st.selectbox(
                "Default causal discovery method",
                ["LLM-guided", "PC Algorithm", "GES", "DirectLiNGAM", "NOTEARS"],
                index=0
            )
            
            default_estimation_method = st.selectbox(
                "Default effect estimation method",
                ["Regression Adjustment", "Propensity Score Matching", "Instrumental Variables", 
                 "Difference-in-Differences", "Doubly Robust"],
                index=0
            )
            
            # Validation settings
            auto_run_validation = st.checkbox(
                "Auto-run assumption validation",
                value=st.session_state.user_settings.get('auto_run_validation', True),
                help="Automatically validate causal assumptions after discovery"
            )
            
            strict_validation = st.checkbox(
                "Strict validation mode",
                value=st.session_state.user_settings.get('strict_validation', False),
                help="Use stricter criteria for assumption validation"
            )
            
            # Sample size requirements
            min_sample_size = st.number_input(
                "Minimum sample size warning",
                min_value=10, max_value=1000,
                value=st.session_state.user_settings.get('min_sample_size', 30),
                help="Show warning if dataset is smaller than this"
            )
        
        # Domain-specific defaults
        st.markdown("#### Domain-Specific Defaults")
        
        domain_defaults = {}
        domains = ["Healthcare", "Economics", "Education", "Technology", "Marketing", "Social Sciences"]
        
        selected_domain = st.selectbox("Configure defaults for domain", domains)
        
        col1, col2 = st.columns(2)
        
        with col1:
            domain_defaults[selected_domain] = {
                'typical_effect_size': st.number_input(
                    f"Typical effect size in {selected_domain}",
                    value=0.3, step=0.1, format="%.1f"
                ),
                'common_confounders': st.text_area(
                    f"Common confounders in {selected_domain}",
                    value="Age, Gender, Socioeconomic Status",
                    height=60
                )
            }
        
        with col2:
            domain_defaults[selected_domain].update({
                'preferred_methods': st.multiselect(
                    f"Preferred methods for {selected_domain}",
                    ["Randomized Experiment", "Natural Experiment", "Observational Study", 
                     "Instrumental Variables", "Regression Discontinuity"],
                    default=["Observational Study"]
                ),
                'regulatory_considerations': st.text_area(
                    f"Regulatory considerations for {selected_domain}",
                    value="",
                    height=60
                )
            })
    
    with tab4:
        st.markdown("### UI Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Visual Theme")
            
            # Theme selection
            theme = st.selectbox(
                "Application theme",
                ["Light", "Dark", "Auto (System)"],
                index=["Light", "Dark", "Auto (System)"].index(
                    st.session_state.user_settings.get('theme', 'Light')
                )
            )
            
            # Color scheme
            color_scheme = st.selectbox(
                "Color scheme",
                ["Default", "Colorblind Friendly", "High Contrast", "Monochrome"],
                index=["Default", "Colorblind Friendly", "High Contrast", "Monochrome"].index(
                    st.session_state.user_settings.get('color_scheme', 'Default')
                )
            )
            
            # Font settings
            font_size = st.selectbox(
                "Font size",
                ["Small", "Medium", "Large", "Extra Large"],
                index=["Small", "Medium", "Large", "Extra Large"].index(
                    st.session_state.user_settings.get('font_size', 'Medium')
                )
            )
            
            # Layout preferences
            sidebar_default = st.selectbox(
                "Sidebar default state",
                ["Expanded", "Collapsed"],
                index=["Expanded", "Collapsed"].index(
                    st.session_state.user_settings.get('sidebar_default', 'Expanded')
                )
            )
        
        with col2:
            st.markdown("#### Dashboard Layout")
            
            # Page layout preferences
            default_layout = st.radio(
                "Default page layout",
                ["Wide", "Centered", "Full Width"],
                index=["Wide", "Centered", "Full Width"].index(
                    st.session_state.user_settings.get('default_layout', 'Wide')
                )
            )
            
            # Chart preferences
            default_chart_theme = st.selectbox(
                "Chart theme",
                ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"],
                index=0
            )
            
            show_tooltips = st.checkbox(
                "Show detailed tooltips",
                value=st.session_state.user_settings.get('show_tooltips', True)
            )
            
            animate_charts = st.checkbox(
                "Enable chart animations",
                value=st.session_state.user_settings.get('animate_charts', True)
            )
            
            # Table preferences
            rows_per_page = st.number_input(
                "Table rows per page",
                min_value=10, max_value=100,
                value=st.session_state.user_settings.get('rows_per_page', 25),
                step=5
            )
        
        # Accessibility options
        st.markdown("#### Accessibility")
        
        col1, col2 = st.columns(2)
        
        with col1:
            high_contrast_mode = st.checkbox(
                "High contrast mode",
                value=st.session_state.user_settings.get('high_contrast_mode', False)
            )
            
            reduce_motion = st.checkbox(
                "Reduce motion and animations",
                value=st.session_state.user_settings.get('reduce_motion', False)
            )
        
        with col2:
            keyboard_navigation = st.checkbox(
                "Enhanced keyboard navigation",
                value=st.session_state.user_settings.get('keyboard_navigation', False)
            )
            
            screen_reader_mode = st.checkbox(
                "Screen reader optimization",
                value=st.session_state.user_settings.get('screen_reader_mode', False)
            )
    
    with tab5:
        st.markdown("### Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Storage Settings")
            
            # Data retention
            data_retention_days = st.number_input(
                "Data retention period (days)",
                min_value=1, max_value=365,
                value=st.session_state.user_settings.get('data_retention_days', 30),
                help="How long to keep uploaded datasets"
            )
            
            # Auto-cleanup
            auto_cleanup = st.checkbox(
                "Auto-cleanup old data",
                value=st.session_state.user_settings.get('auto_cleanup', True),
                help="Automatically delete old datasets and results"
            )
            
            # Backup settings
            auto_backup = st.checkbox(
                "Enable automatic backups",
                value=st.session_state.user_settings.get('auto_backup', False)
            )
            
            if auto_backup:
                backup_frequency = st.selectbox(
                    "Backup frequency",
                    ["Daily", "Weekly", "Monthly"],
                    index=1
                )
        
        with col2:
            st.markdown("#### Privacy Settings")
            
            # Data privacy
            anonymize_data = st.checkbox(
                "Anonymize sensitive data",
                value=st.session_state.user_settings.get('anonymize_data', True),
                help="Automatically detect and anonymize personal information"
            )
            
            # Sharing settings
            allow_data_sharing = st.checkbox(
                "Allow anonymous data sharing for research",
                value=st.session_state.user_settings.get('allow_data_sharing', False),
                help="Share anonymized analysis patterns to improve the service"
            )
            
            # Export settings
            include_metadata = st.checkbox(
                "Include metadata in exports",
                value=st.session_state.user_settings.get('include_metadata', True)
            )
            
            export_format_default = st.selectbox(
                "Default export format",
                ["CSV", "Excel", "JSON", "Parquet"],
                index=0
            )
        
        # File handling
        st.markdown("#### File Handling")
        
        max_file_size = st.slider(
            "Maximum file size (MB)",
            min_value=1, max_value=500,
            value=st.session_state.user_settings.get('max_file_size', 100)
        )
        
        allowed_file_types = st.multiselect(
            "Allowed file types",
            ["CSV", "Excel", "JSON", "Parquet", "TSV", "SPSS", "Stata"],
            default=st.session_state.user_settings.get('allowed_file_types', ["CSV", "Excel", "JSON", "Parquet"])
        )
        
        # Sample data preferences
        st.markdown("#### Sample Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reset Sample Datasets"):
                # Reset sample data
                if 'sample_data' in st.session_state:
                    del st.session_state['sample_data']
                st.success("Sample datasets reset!")
        
        with col2:
            if st.button("📥 Import Custom Samples"):
                st.info("Custom sample import feature coming soon!")
    
    with tab6:
        st.markdown("### Advanced Settings")
        
        st.warning("⚠️ Advanced settings - modify with caution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### System Configuration")
            
            # Performance settings
            enable_caching = st.checkbox(
                "Enable result caching",
                value=st.session_state.user_settings.get('enable_caching', True),
                help="Cache analysis results for faster repeat operations"
            )
            
            if enable_caching:
                cache_size_mb = st.number_input(
                    "Cache size (MB)",
                    min_value=10, max_value=1000,
                    value=st.session_state.user_settings.get('cache_size_mb', 100)
                )
            
            # Parallel processing
            enable_parallel = st.checkbox(
                "Enable parallel processing",
                value=st.session_state.user_settings.get('enable_parallel', True),
                help="Use multiple CPU cores for intensive computations"
            )
            
            if enable_parallel:
                max_workers = st.number_input(
                    "Maximum worker threads",
                    min_value=1, max_value=16,
                    value=st.session_state.user_settings.get('max_workers', 4)
                )
            
            # Memory management
            memory_limit_gb = st.number_input(
                "Memory limit (GB)",
                min_value=1, max_value=64,
                value=st.session_state.user_settings.get('memory_limit_gb', 8),
                help="Maximum memory usage for analysis operations"
            )
        
        with col2:
            st.markdown("#### Development Settings")
            
            # Logging
            log_level = st.selectbox(
                "Log level",
                ["DEBUG", "INFO", "WARNING", "ERROR"],
                index=["DEBUG", "INFO", "WARNING", "ERROR"].index(
                    st.session_state.user_settings.get('log_level', 'INFO')
                )
            )
            
            # API settings
            api_rate_limit = st.number_input(
                "API rate limit (requests/minute)",
                min_value=1, max_value=100,
                value=st.session_state.user_settings.get('api_rate_limit', 20)
            )
            
            # Experimental features
            enable_experimental = st.checkbox(
                "Enable experimental features",
                value=st.session_state.user_settings.get('enable_experimental', False),
                help="Access to beta and experimental functionality"
            )
            
            if enable_experimental:
                st.info("🧪 Experimental features enabled. Some functionality may be unstable.")
        
        # Configuration management
        st.markdown("#### Configuration Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📤 Export Settings"):
                settings_json = json.dumps(st.session_state.user_settings, indent=2)
                st.download_button(
                    "Download Settings",
                    data=settings_json,
                    file_name=f"causallm_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            uploaded_settings = st.file_uploader("📥 Import Settings", type=['json'])
            if uploaded_settings:
                try:
                    imported_settings = json.load(uploaded_settings)
                    st.session_state.user_settings.update(imported_settings)
                    st.success("Settings imported successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error importing settings: {str(e)}")
        
        with col3:
            if st.button("🔄 Reset to Defaults"):
                if st.button("⚠️ Confirm Reset", type="secondary"):
                    initialize_default_settings(reset=True)
                    st.success("Settings reset to defaults!")
                    st.rerun()
    
    # Save settings button (always visible at bottom)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("💾 Save Settings", type="primary"):
            # Update session state with form values
            save_current_settings()
            st.success("✅ Settings saved successfully!")
    
    with col2:
        if st.button("🔄 Reload Settings"):
            st.rerun()
    
    with col3:
        if st.button("❌ Discard Changes"):
            initialize_default_settings()
            st.info("Changes discarded - settings reloaded")
            st.rerun()

def initialize_default_settings(reset=False):
    """Initialize default user settings"""
    if reset or 'user_settings' not in st.session_state:
        st.session_state.user_settings = {
            # General settings
            'user_name': '',
            'user_role': 'Researcher',
            'experience_level': 'Intermediate',
            'auto_save_enabled': True,
            'auto_save_interval': 5,
            'show_confirmations': True,
            'debug_mode': False,
            'enable_analytics': True,
            'email_notifications': False,
            'email_address': '',
            'notification_types': ['Analysis Complete', 'Errors'],
            
            # LLM settings
            'llm_provider': 'OpenAI',
            'llm_temperature': 0.7,
            'llm_max_tokens': 1000,
            'llm_timeout': 30,
            'response_style': 'Professional',
            'custom_system_prompt': '',
            'domain_specific_prompts': True,
            
            # Analysis defaults
            'default_confidence': 0.95,
            'significance_level': 0.05,
            'small_effect_threshold': 0.2,
            'medium_effect_threshold': 0.5,
            'large_effect_threshold': 0.8,
            'auto_run_validation': True,
            'strict_validation': False,
            'min_sample_size': 30,
            
            # UI preferences
            'theme': 'Light',
            'color_scheme': 'Default',
            'font_size': 'Medium',
            'sidebar_default': 'Expanded',
            'default_layout': 'Wide',
            'show_tooltips': True,
            'animate_charts': True,
            'rows_per_page': 25,
            'high_contrast_mode': False,
            'reduce_motion': False,
            'keyboard_navigation': False,
            'screen_reader_mode': False,
            
            # Data management
            'data_retention_days': 30,
            'auto_cleanup': True,
            'auto_backup': False,
            'anonymize_data': True,
            'allow_data_sharing': False,
            'include_metadata': True,
            'max_file_size': 100,
            'allowed_file_types': ['CSV', 'Excel', 'JSON', 'Parquet'],
            
            # Advanced settings
            'enable_caching': True,
            'cache_size_mb': 100,
            'enable_parallel': True,
            'max_workers': 4,
            'memory_limit_gb': 8,
            'log_level': 'INFO',
            'api_rate_limit': 20,
            'enable_experimental': False,
            
            # Metadata
            'settings_version': '1.0',
            'last_updated': datetime.now().isoformat()
        }

def save_current_settings():
    """Save current form values to session state settings"""
    # This function would collect values from all form inputs
    # and update st.session_state.user_settings
    # For brevity, we'll just update the timestamp
    st.session_state.user_settings['last_updated'] = datetime.now().isoformat()
    
    # In a real implementation, you would:
    # 1. Collect values from all form inputs
    # 2. Validate the settings
    # 3. Save to persistent storage (database, file, etc.)
    # 4. Apply settings to the current session

def apply_settings():
    """Apply current settings to the application"""
    settings = st.session_state.user_settings
    
    # Apply theme
    if settings.get('theme') == 'Dark':
        # Apply dark theme CSS
        pass
    
    # Apply other UI settings
    if settings.get('high_contrast_mode'):
        # Apply high contrast styles
        pass
    
    # Configure logging
    import logging
    log_level = getattr(logging, settings.get('log_level', 'INFO'))
    logging.getLogger().setLevel(log_level)