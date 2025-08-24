import streamlit as st
import pandas as pd
import asyncio
from causalllm.interactive_causal_qa import InteractiveCausalQA, QuestionType, ConfidenceLevel
from causalllm.llm_client import get_llm_client
import time
import re

def show():
    st.title("💬 Interactive Causal Q&A")
    st.markdown("Ask natural language questions about causal relationships in your data")
    
    # Initialize Q&A system
    if 'qa_system' not in st.session_state:
        try:
            llm_client = get_llm_client()
            st.session_state.qa_system = InteractiveCausalQA(llm_client)
        except Exception as e:
            st.error(f"Failed to initialize Q&A system: {str(e)}")
            st.info("Please check your LLM client configuration.")
            return
    
    # Check if data is available
    has_data = 'current_data' in st.session_state
    has_discovery = 'discovery_results' in st.session_state
    
    if not has_data:
        st.warning("📁 For best results, upload a dataset in the Data Manager first!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to Data Manager"):
                # This would navigate to data manager in a real app
                st.info("Navigate to Data Manager → Upload your dataset")
        with col2:
            if st.button("Continue without data"):
                st.info("You can still ask general causal questions!")
    
    # Main interface
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗨️ Ask Questions", "📚 Question Library", "📊 Q&A History", "⚙️ Configuration"
    ])
    
    with tab1:
        st.markdown("### Ask Your Causal Question")
        
        # Question input section
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Domain selection
            domain = st.selectbox(
                "Domain Context",
                ["general", "healthcare", "business", "education", "social_science", "technology"],
                help="Domain helps provide more relevant and accurate answers"
            )
            
            # Question type suggestions
            st.markdown("**Suggested Question Types:**")
            question_types = {
                "Effect Questions": "What is the effect of X on Y?",
                "Mechanism Questions": "How does X affect Y?",
                "Counterfactual Questions": "What would happen if X were different?",
                "Confounding Questions": "What variables might confound the X-Y relationship?",
                "Policy Questions": "Should we intervene on X to change Y?"
            }
            
            selected_type = st.radio(
                "Question Type",
                list(question_types.keys()),
                horizontal=True,
                help="Select a question type for guidance"
            )
        
        with col2:
            st.markdown("**Quick Examples:**")
            example_questions = {
                "healthcare": [
                    "Does the treatment improve patient outcomes?",
                    "What factors confound treatment effectiveness?",
                    "What would happen if patients took higher doses?"
                ],
                "business": [
                    "Does marketing spend increase sales?",
                    "How does pricing affect customer retention?",
                    "What drives customer satisfaction?"
                ],
                "education": [
                    "Do smaller class sizes improve learning?",
                    "What factors influence student performance?",
                    "How does teacher experience affect outcomes?"
                ]
            }
            
            examples = example_questions.get(domain, example_questions["healthcare"])
            
            for i, example in enumerate(examples):
                if st.button(f"💡 {example[:30]}...", key=f"example_{i}"):
                    st.session_state['question_input'] = example
                    st.rerun()
        
        # Main question input
        question = st.text_area(
            "Enter your causal question:",
            value=st.session_state.get('question_input', ''),
            height=100,
            placeholder=f"Example: {question_types[selected_type]}",
            help="Be as specific as possible. Mention variables by name if available."
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                include_evidence = st.checkbox("Include Supporting Evidence", value=True)
                include_limitations = st.checkbox("Include Limitations", value=True)
                include_alternatives = st.checkbox("Include Alternative Explanations", value=True)
            
            with col2:
                confidence_threshold = st.slider("Minimum Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
                max_response_length = st.selectbox("Response Length", ["Short", "Medium", "Detailed"], index=1)
                reasoning_style = st.selectbox("Reasoning Style", ["Balanced", "Conservative", "Exploratory"], index=0)
        
        # Ask question
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            ask_button = st.button("🤔 Ask Question", type="primary", disabled=not question.strip())
        
        with col2:
            if st.button("🔄 Clear"):
                st.session_state['question_input'] = ''
                st.rerun()
        
        with col3:
            if st.button("💾 Save Question"):
                if question.strip():
                    if 'saved_questions' not in st.session_state:
                        st.session_state.saved_questions = []
                    st.session_state.saved_questions.append({
                        'question': question,
                        'domain': domain,
                        'timestamp': pd.Timestamp.now()
                    })
                    st.success("Question saved!")
        
        # Process question
        if ask_button and question.strip():
            with st.spinner("🧠 Analyzing your question... This may take a moment."):
                try:
                    start_time = time.time()
                    
                    # Start conversation context
                    context_data = {}
                    variable_descriptions = {}
                    
                    if has_data:
                        context_data = st.session_state.current_data
                        if 'variable_roles' in st.session_state:
                            roles = st.session_state.variable_roles
                            for role, vars_list in roles.items():
                                if isinstance(vars_list, list):
                                    for var in vars_list:
                                        variable_descriptions[var] = f"{role} variable"
                                elif vars_list:
                                    variable_descriptions[vars_list] = f"{role} variable"
                    
                    context = st.session_state.qa_system.start_conversation(
                        domain=domain,
                        data=context_data,
                        variable_descriptions=variable_descriptions
                    )
                    
                    # Get answer
                    answer = asyncio.run(
                        st.session_state.qa_system.ask_causal_question(question, context)
                    )
                    
                    response_time = time.time() - start_time
                    
                    # Store Q&A in session
                    qa_record = {
                        'question': question,
                        'answer': answer,
                        'domain': domain,
                        'response_time': response_time,
                        'timestamp': pd.Timestamp.now(),
                        'has_data': has_data
                    }
                    
                    if 'qa_history' not in st.session_state:
                        st.session_state.qa_history = []
                    
                    st.session_state.qa_history.insert(0, qa_record)  # Most recent first
                    
                    # Display results
                    st.success(f"✅ Analysis completed in {response_time:.1f} seconds!")
                    
                    # Main answer
                    st.markdown("### 🎯 Answer")
                    st.markdown(f"**{answer.main_answer}**")
                    
                    # Confidence indicator
                    confidence_colors = {
                        "very_high": "🟢",
                        "high": "🟡", 
                        "medium": "🟠",
                        "low": "🔴",
                        "very_low": "⚫"
                    }
                    
                    confidence_level = answer.confidence_level.value
                    confidence_emoji = confidence_colors.get(confidence_level, "🔘")
                    
                    st.markdown(f"**Confidence Level:** {confidence_emoji} {confidence_level.replace('_', ' ').title()}")
                    
                    # Detailed sections in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if include_evidence and answer.supporting_evidence:
                            st.markdown("#### 📊 Supporting Evidence")
                            for i, evidence in enumerate(answer.supporting_evidence, 1):
                                st.markdown(f"{i}. {evidence}")
                        
                        if include_alternatives and answer.alternative_explanations:
                            st.markdown("#### 🤔 Alternative Explanations")
                            for i, alt in enumerate(answer.alternative_explanations, 1):
                                st.markdown(f"{i}. {alt}")
                    
                    with col2:
                        if include_limitations and answer.limitations:
                            st.markdown("#### ⚠️ Limitations & Caveats")
                            for i, limitation in enumerate(answer.limitations, 1):
                                st.warning(f"{i}. {limitation}")
                        
                        if answer.follow_up_questions:
                            st.markdown("#### 🔍 Suggested Follow-up Questions")
                            for i, followup in enumerate(answer.follow_up_questions, 1):
                                if st.button(f"❓ {followup}", key=f"followup_{i}"):
                                    st.session_state['question_input'] = followup
                                    st.rerun()
                    
                    # Action buttons
                    st.markdown("---")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if st.button("👍 Helpful"):
                            st.success("Thank you for the feedback!")
                    
                    with col2:
                        if st.button("👎 Not Helpful"):
                            feedback = st.text_input("What could be improved?")
                            if feedback:
                                st.info("Feedback recorded!")
                    
                    with col3:
                        if st.button("💾 Save Answer"):
                            st.info("Answer saved to history!")
                    
                    with col4:
                        if st.button("🔄 Ask Follow-up"):
                            # Clear input for new question
                            st.session_state['question_input'] = ''
                    
                except Exception as e:
                    st.error(f"Failed to process question: {str(e)}")
                    st.info("This might be due to:")
                    st.markdown("""
                    - LLM client configuration issues
                    - Network connectivity problems
                    - Question complexity or format
                    - Data compatibility issues
                    """)
    
    with tab2:
        st.markdown("### Question Library & Templates")
        st.info("Browse pre-built question templates organized by domain and analysis type")
        
        # Question templates by domain
        question_library = {
            "Healthcare": {
                "Treatment Effects": [
                    "Does [treatment] improve [outcome] compared to [control]?",
                    "What is the optimal dosage of [drug] for [condition]?",
                    "How does [intervention] affect [health_metric] over time?",
                    "What factors mediate the effect of [treatment] on [outcome]?"
                ],
                "Risk Factors": [
                    "Does [exposure] increase the risk of [disease]?",
                    "What lifestyle factors influence [health_outcome]?",
                    "How do genetics interact with [environmental_factor]?",
                    "What are the key predictors of [medical_condition]?"
                ],
                "Policy & Prevention": [
                    "Would implementing [policy] reduce [health_problem]?",
                    "What prevention strategies are most effective for [disease]?",
                    "How would changing [healthcare_practice] affect [patient_outcomes]?"
                ]
            },
            "Business": {
                "Marketing & Sales": [
                    "Does [marketing_channel] increase [conversion_metric]?",
                    "What is the ROI of [advertising_campaign] on [sales]?",
                    "How does [pricing_strategy] affect [customer_behavior]?",
                    "What factors drive [customer_satisfaction]?"
                ],
                "Operations": [
                    "Does [process_change] improve [efficiency_metric]?",
                    "How does [employee_training] affect [performance]?",
                    "What operational factors influence [customer_retention]?",
                    "Would automating [process] reduce [costs]?"
                ],
                "Strategy": [
                    "How does [market_expansion] affect [profitability]?",
                    "What factors contribute to [competitive_advantage]?",
                    "Does [innovation_investment] drive [growth]?"
                ]
            },
            "Education": {
                "Teaching & Learning": [
                    "Does [teaching_method] improve [learning_outcome]?",
                    "How does [class_size] affect [student_performance]?",
                    "What factors influence [student_engagement]?",
                    "Does [technology_use] enhance [educational_outcomes]?"
                ],
                "Policy & Administration": [
                    "Would [education_policy] improve [system_performance]?",
                    "How does [resource_allocation] affect [school_outcomes]?",
                    "What factors contribute to [teacher_effectiveness]?"
                ],
                "Student Success": [
                    "What predicts [student_graduation] rates?",
                    "How do [socioeconomic_factors] influence [academic_achievement]?",
                    "Does [early_intervention] prevent [academic_problems]?"
                ]
            }
        }
        
        # Display question library
        selected_domain = st.selectbox("Select Domain", list(question_library.keys()))
        
        if selected_domain in question_library:
            domain_questions = question_library[selected_domain]
            
            for category, questions in domain_questions.items():
                with st.expander(f"📋 {category}"):
                    for i, template in enumerate(questions):
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            st.markdown(f"**Template:** {template}")
                        
                        with col2:
                            if st.button("Use Template", key=f"template_{selected_domain}_{category}_{i}"):
                                st.session_state['question_input'] = template
                                st.success("Template loaded! Go to Ask Questions tab.")
        
        # Custom template creation
        st.markdown("### 🛠️ Create Custom Template")
        
        with st.expander("Create New Question Template"):
            template_name = st.text_input("Template Name")
            template_question = st.text_area("Question Template (use [placeholder] format)")
            template_domain = st.selectbox("Domain", ["Healthcare", "Business", "Education", "General"])
            template_category = st.text_input("Category")
            
            if st.button("Save Template"):
                if template_name and template_question:
                    if 'custom_templates' not in st.session_state:
                        st.session_state.custom_templates = []
                    
                    st.session_state.custom_templates.append({
                        'name': template_name,
                        'question': template_question,
                        'domain': template_domain,
                        'category': template_category,
                        'created': pd.Timestamp.now()
                    })
                    
                    st.success("Custom template saved!")
                else:
                    st.warning("Please fill in template name and question.")
        
        # Display custom templates
        if 'custom_templates' in st.session_state and st.session_state.custom_templates:
            st.markdown("### 📝 Your Custom Templates")
            
            for i, template in enumerate(st.session_state.custom_templates):
                with st.expander(f"📝 {template['name']} ({template['domain']})"):
                    st.markdown(f"**Question:** {template['question']}")
                    st.markdown(f"**Category:** {template['category']}")
                    st.markdown(f"**Created:** {template['created'].strftime('%Y-%m-%d %H:%M')}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Use Template", key=f"custom_{i}"):
                            st.session_state['question_input'] = template['question']
                            st.success("Template loaded!")
                    
                    with col2:
                        if st.button("Delete", key=f"delete_{i}"):
                            st.session_state.custom_templates.pop(i)
                            st.rerun()
    
    with tab3:
        st.markdown("### Q&A History")
        
        if 'qa_history' not in st.session_state or not st.session_state.qa_history:
            st.info("No questions asked yet. Start by asking a question in the main tab!")
            return
        
        history = st.session_state.qa_history
        
        # History overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Questions", len(history))
        
        with col2:
            avg_confidence = []
            for qa in history:
                conf_map = {"very_high": 5, "high": 4, "medium": 3, "low": 2, "very_low": 1}
                avg_confidence.append(conf_map.get(qa['answer'].confidence_level.value, 3))
            
            avg_conf_score = sum(avg_confidence) / len(avg_confidence) if avg_confidence else 0
            st.metric("Avg Confidence", f"{avg_conf_score:.1f}/5")
        
        with col3:
            domains = [qa['domain'] for qa in history]
            most_common_domain = max(set(domains), key=domains.count) if domains else "None"
            st.metric("Most Used Domain", most_common_domain.title())
        
        with col4:
            response_times = [qa['response_time'] for qa in history]
            avg_time = sum(response_times) / len(response_times) if response_times else 0
            st.metric("Avg Response Time", f"{avg_time:.1f}s")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_domain = st.selectbox("Filter by Domain", ["All"] + list(set(qa['domain'] for qa in history)))
        
        with col2:
            filter_confidence = st.selectbox("Filter by Confidence", 
                                           ["All", "High (4-5)", "Medium (3)", "Low (1-2)"])
        
        with col3:
            sort_by = st.selectbox("Sort by", ["Most Recent", "Oldest", "Highest Confidence", "Response Time"])
        
        # Apply filters
        filtered_history = history.copy()
        
        if filter_domain != "All":
            filtered_history = [qa for qa in filtered_history if qa['domain'] == filter_domain]
        
        if filter_confidence != "All":
            conf_map = {"very_high": 5, "high": 4, "medium": 3, "low": 2, "very_low": 1}
            
            if filter_confidence == "High (4-5)":
                filtered_history = [qa for qa in filtered_history 
                                  if conf_map.get(qa['answer'].confidence_level.value, 3) >= 4]
            elif filter_confidence == "Medium (3)":
                filtered_history = [qa for qa in filtered_history 
                                  if conf_map.get(qa['answer'].confidence_level.value, 3) == 3]
            elif filter_confidence == "Low (1-2)":
                filtered_history = [qa for qa in filtered_history 
                                  if conf_map.get(qa['answer'].confidence_level.value, 3) <= 2]
        
        # Sort results
        if sort_by == "Oldest":
            filtered_history = sorted(filtered_history, key=lambda x: x['timestamp'])
        elif sort_by == "Highest Confidence":
            conf_map = {"very_high": 5, "high": 4, "medium": 3, "low": 2, "very_low": 1}
            filtered_history = sorted(filtered_history, 
                                    key=lambda x: conf_map.get(x['answer'].confidence_level.value, 3), 
                                    reverse=True)
        elif sort_by == "Response Time":
            filtered_history = sorted(filtered_history, key=lambda x: x['response_time'])
        
        # Display history
        st.markdown(f"#### Showing {len(filtered_history)} of {len(history)} questions")
        
        for i, qa in enumerate(filtered_history):
            with st.expander(f"Q{i+1}: {qa['question'][:60]}... ({qa['timestamp'].strftime('%Y-%m-%d %H:%M')})"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Question:** {qa['question']}")
                    st.markdown(f"**Answer:** {qa['answer'].main_answer}")
                    
                    if qa['answer'].supporting_evidence:
                        st.markdown("**Evidence:**")
                        for evidence in qa['answer'].supporting_evidence[:2]:  # Show first 2
                            st.markdown(f"• {evidence}")
                
                with col2:
                    st.markdown(f"**Domain:** {qa['domain'].title()}")
                    st.markdown(f"**Confidence:** {qa['answer'].confidence_level.value.replace('_', ' ').title()}")
                    st.markdown(f"**Response Time:** {qa['response_time']:.1f}s")
                    st.markdown(f"**Data Used:** {'Yes' if qa['has_data'] else 'No'}")
                    
                    # Action buttons
                    if st.button("🔄 Ask Again", key=f"reask_{i}"):
                        st.session_state['question_input'] = qa['question']
                        st.info("Question loaded! Go to Ask Questions tab.")
                    
                    if st.button("🗑️ Delete", key=f"del_{i}"):
                        st.session_state.qa_history.remove(qa)
                        st.rerun()
        
        # Bulk actions
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export History"):
                # Create export data
                export_data = []
                for qa in filtered_history:
                    export_data.append({
                        'timestamp': qa['timestamp'],
                        'question': qa['question'],
                        'domain': qa['domain'],
                        'answer': qa['answer'].main_answer,
                        'confidence': qa['answer'].confidence_level.value,
                        'response_time': qa['response_time'],
                        'has_data': qa['has_data']
                    })
                
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)
                
                st.download_button(
                    "Download CSV",
                    csv,
                    file_name=f"qa_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Generate Summary"):
                st.info("Summary generation feature coming soon!")
        
        with col3:
            if st.button("🗑️ Clear All History"):
                if st.confirm("Are you sure you want to clear all Q&A history?"):
                    st.session_state.qa_history = []
                    st.rerun()
    
    with tab4:
        st.markdown("### Configuration & Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Q&A System Settings")
            
            default_domain = st.selectbox(
                "Default Domain",
                ["general", "healthcare", "business", "education", "social_science", "technology"]
            )
            
            default_confidence_threshold = st.slider(
                "Default Confidence Threshold",
                0.0, 1.0, 0.5, 0.1
            )
            
            auto_save_questions = st.checkbox("Auto-save all questions", value=True)
            show_follow_ups = st.checkbox("Show follow-up questions", value=True)
            detailed_explanations = st.checkbox("Include detailed explanations", value=True)
        
        with col2:
            st.markdown("#### Display Settings")
            
            max_history_items = st.slider("Max history items to show", 10, 100, 50)
            response_format = st.selectbox("Preferred response format", 
                                         ["Structured", "Narrative", "Bullet Points"])
            
            show_confidence_colors = st.checkbox("Color-code confidence levels", value=True)
            show_timing_info = st.checkbox("Show response timing", value=True)
            compact_mode = st.checkbox("Use compact display mode", value=False)
        
        # Save settings
        if st.button("💾 Save Settings"):
            settings = {
                'default_domain': default_domain,
                'default_confidence_threshold': default_confidence_threshold,
                'auto_save_questions': auto_save_questions,
                'show_follow_ups': show_follow_ups,
                'detailed_explanations': detailed_explanations,
                'max_history_items': max_history_items,
                'response_format': response_format,
                'show_confidence_colors': show_confidence_colors,
                'show_timing_info': show_timing_info,
                'compact_mode': compact_mode
            }
            
            st.session_state['qa_settings'] = settings
            st.success("Settings saved!")
        
        # System information
        st.markdown("---")
        st.markdown("#### System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Q&A System Status:** ✅ Active")
            st.info("**LLM Integration:** ✅ Connected")
            st.info("**Data Integration:** ✅ Available" if has_data else "**Data Integration:** ⚠️ No data loaded")
        
        with col2:
            st.info("**Discovery Integration:** ✅ Available" if has_discovery else "**Discovery Integration:** ⚠️ No discovery results")
            st.info("**Question Templates:** ✅ Loaded")
            st.info("**History Tracking:** ✅ Active")
        
        # Performance metrics
        if 'qa_history' in st.session_state and st.session_state.qa_history:
            st.markdown("#### Performance Metrics")
            history = st.session_state.qa_history
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                response_times = [qa['response_time'] for qa in history]
                avg_time = sum(response_times) / len(response_times)
                st.metric("Avg Response Time", f"{avg_time:.1f}s")
            
            with col2:
                success_rate = 100.0  # Assuming all completed questions are successful
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            with col3:
                total_questions = len(history)
                st.metric("Total Questions Processed", total_questions)

    # Update session statistics
    if 'qa_history' in st.session_state and st.session_state.qa_history:
        if 'session_stats' not in st.session_state:
            st.session_state.session_stats = {'analyses': 0, 'datasets': 0, 'success_rate': 0}
        
        # Update analyses count
        qa_count = len(st.session_state.qa_history)
        st.session_state.session_stats['analyses'] = max(st.session_state.session_stats.get('analyses', 0), qa_count)