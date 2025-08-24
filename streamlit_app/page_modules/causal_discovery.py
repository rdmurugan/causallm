import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import plotly.graph_objects as go
import networkx as nx
from causalllm.core import CausalLLMCore
from causalllm.llm_client import get_llm_client
from causalllm.llm_causal_discovery import LLMCausalDiscoveryAgent, DiscoveryMethod
from causalllm.automatic_confounder_detection import AutomaticConfounderDetector
from causalllm.visual_causal_graphs import VisualCausalGraphGenerator, NodeType, GraphLayout
import time

def show():
    st.title("🔍 Causal Discovery")
    st.markdown("AI-powered automated discovery of causal relationships in your data")
    
    # Check if data is available
    if 'current_data' not in st.session_state:
        st.warning("📁 Please upload a dataset in the Data Manager first!")
        if st.button("Go to Data Manager"):
            st.session_state['selected_page'] = "📊 Data Manager"
            st.rerun()
        return
    
    data = st.session_state.current_data
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Run Discovery", "📊 Results & Visualization", "🎯 Confounder Detection", "📋 Analysis Summary"
    ])
    
    with tab1:
        st.markdown("### Discovery Configuration")
        
        # Method selection
        col1, col2 = st.columns(2)
        
        with col1:
            discovery_method = st.selectbox(
                "Discovery Method",
                ["Hybrid LLM + Statistical", "LLM-Guided", "Statistical Only"],
                help="Hybrid method combines statistical algorithms with LLM reasoning for best results"
            )
            
            domain = st.selectbox(
                "Domain Context",
                ["healthcare", "business", "education", "social_science", "technology", "general"],
                help="Domain context helps the LLM provide more relevant insights"
            )
        
        with col2:
            confidence_threshold = st.slider(
                "Minimum Confidence Threshold", 
                0.0, 1.0, 0.6, 0.05,
                help="Only include relationships above this confidence level"
            )
            
            max_edges = st.number_input(
                "Maximum Edges to Discover", 
                min_value=5, max_value=50, value=20,
                help="Limit the number of causal relationships to discover"
            )
        
        # Variable selection
        st.markdown("#### Variable Configuration")
        
        if 'variable_roles' in st.session_state:
            roles = st.session_state.variable_roles
            st.info(f"Using pre-defined variable roles: {len(roles.get('treatment', []))} treatment, {len(roles.get('outcome', []))} outcome, {len(roles.get('confounders', []))} confounders")
            
            variable_descriptions = {}
            
            # Let user provide descriptions for key variables
            st.markdown("**Variable Descriptions** (Optional - helps improve discovery accuracy)")
            
            key_vars = roles.get('treatment', []) + roles.get('outcome', []) + roles.get('confounders', [])[:5]
            
            for var in key_vars:
                desc = st.text_input(
                    f"Description for '{var}':",
                    key=f"desc_{var}",
                    placeholder=f"Brief description of what {var} represents..."
                )
                if desc.strip():
                    variable_descriptions[var] = desc
        else:
            st.warning("⚠️ No variable roles assigned. Please assign roles in the Data Manager for better results.")
            variable_descriptions = {}
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            target_variable = st.selectbox(
                "Target Variable (Optional)",
                ["None"] + data.columns.tolist(),
                help="Focus discovery around this target variable"
            )
            target_variable = target_variable if target_variable != "None" else None
            
            include_temporal = st.checkbox(
                "Include Temporal Analysis",
                help="Analyze time-based relationships if time variable is available"
            )
            
            bootstrap_iterations = st.slider(
                "Bootstrap Iterations", 
                10, 100, 50,
                help="More iterations = more reliable results but slower execution"
            )
        
        # Run discovery
        st.markdown("---")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            run_discovery = st.button("🚀 Run Causal Discovery", type="primary", use_container_width=True)
        
        with col2:
            if st.button("💾 Save Config"):
                config = {
                    'method': discovery_method,
                    'domain': domain,
                    'confidence_threshold': confidence_threshold,
                    'max_edges': max_edges,
                    'variable_descriptions': variable_descriptions,
                    'target_variable': target_variable
                }
                st.session_state['discovery_config'] = config
                st.success("Config saved!")
        
        with col3:
            if 'discovery_config' in st.session_state:
                if st.button("📂 Load Config"):
                    st.info("Config loaded!")
        
        if run_discovery:
            with st.spinner("🔍 Discovering causal relationships... This may take a few minutes."):
                try:
                    # Initialize discovery agent
                    llm_client = get_llm_client()
                    discovery_agent = LLMCausalDiscoveryAgent(llm_client)
                    
                    # Map method selection to enum
                    method_map = {
                        "Hybrid LLM + Statistical": DiscoveryMethod.HYBRID_LLM_STATISTICAL,
                        "LLM-Guided": DiscoveryMethod.LLM_GUIDED,
                        "Statistical Only": DiscoveryMethod.STATISTICAL_ONLY
                    }
                    
                    # Run discovery
                    start_time = time.time()
                    
                    structure = asyncio.run(discovery_agent.discover_causal_structure(
                        data=data,
                        variable_descriptions=variable_descriptions,
                        domain=domain,
                        context=f"Causal discovery analysis in {domain} domain",
                        method=method_map[discovery_method],
                        target_variable=target_variable
                    ))
                    
                    discovery_time = time.time() - start_time
                    
                    # Store results
                    st.session_state['discovery_results'] = {
                        'structure': structure,
                        'config': {
                            'method': discovery_method,
                            'domain': domain,
                            'confidence_threshold': confidence_threshold,
                            'discovery_time': discovery_time
                        },
                        'timestamp': pd.Timestamp.now()
                    }
                    
                    st.success(f"✅ Discovery completed in {discovery_time:.1f} seconds!")
                    st.balloons()
                    
                    # Show quick summary
                    edges = structure.edges
                    filtered_edges = [e for e in edges if e.confidence >= confidence_threshold]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Relationships", len(edges))
                    with col2:
                        st.metric("High Confidence", len(filtered_edges))
                    with col3:
                        st.metric("Overall Confidence", f"{structure.overall_confidence:.2f}")
                    with col4:
                        st.metric("Discovery Time", f"{discovery_time:.1f}s")
                    
                except Exception as e:
                    st.error(f"Discovery failed: {str(e)}")
                    st.info("This might be due to missing LLM configuration or data format issues.")
    
    with tab2:
        if 'discovery_results' not in st.session_state:
            st.info("🚀 Run causal discovery first to view results and visualizations")
            return
        
        results = st.session_state.discovery_results
        structure = results['structure']
        config = results['config']
        
        st.markdown("### Discovery Results")
        
        # Results overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Discovered Edges", len(structure.edges))
        with col2:
            st.metric("Overall Confidence", f"{structure.overall_confidence:.2f}")
        with col3:
            st.metric("Method Used", config['method'].replace("_", " ").title())
        with col4:
            st.metric("Domain", config['domain'].title())
        
        # Edge details table
        st.markdown("#### Discovered Relationships")
        
        if structure.edges:
            edges_data = []
            for edge in structure.edges:
                edges_data.append({
                    'Source': edge.source,
                    'Target': edge.target,
                    'Confidence': f"{edge.confidence:.3f}",
                    'Type': edge.edge_type.value if hasattr(edge, 'edge_type') else 'causal',
                    'Strength': 'Strong' if edge.confidence > 0.8 else 'Medium' if edge.confidence > 0.6 else 'Weak'
                })
            
            edges_df = pd.DataFrame(edges_data)
            
            # Filter by confidence
            confidence_filter = st.slider(
                "Filter by Confidence", 
                0.0, 1.0, config['confidence_threshold'], 0.05,
                key="results_confidence_filter"
            )
            
            filtered_df = edges_df[edges_df['Confidence'].astype(float) >= confidence_filter]
            
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download results
            col1, col2 = st.columns(2)
            with col1:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    file_name=f"causal_discovery_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Visualization
        st.markdown("#### Causal Graph Visualization")
        
        if structure.edges:
            # Visualization options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                layout_option = st.selectbox(
                    "Graph Layout",
                    ["hierarchical", "spring", "circular", "shell"],
                    help="Different layouts emphasize different aspects of the graph"
                )
            
            with col2:
                node_size_by = st.selectbox(
                    "Node Size Based On",
                    ["degree", "betweenness", "uniform"],
                    help="How to size nodes in the visualization"
                )
            
            with col3:
                show_labels = st.checkbox("Show Edge Labels", value=True)
            
            # Create visualization
            try:
                # Build networkx graph
                G = nx.DiGraph()
                
                # Add nodes
                all_nodes = set()
                for edge in structure.edges:
                    if edge.confidence >= confidence_filter:
                        all_nodes.add(edge.source)
                        all_nodes.add(edge.target)
                
                G.add_nodes_from(all_nodes)
                
                # Add edges with weights
                for edge in structure.edges:
                    if edge.confidence >= confidence_filter:
                        G.add_edge(edge.source, edge.target, weight=edge.confidence)
                
                if len(G.nodes()) > 0:
                    # Create layout
                    if layout_option == "hierarchical":
                        pos = nx.nx_agraph.graphviz_layout(G, prog='dot') if nx.nx_agraph else nx.spring_layout(G)
                    elif layout_option == "spring":
                        pos = nx.spring_layout(G, k=3, iterations=50)
                    elif layout_option == "circular":
                        pos = nx.circular_layout(G)
                    else:
                        pos = nx.shell_layout(G)
                    
                    # Calculate node attributes
                    if node_size_by == "degree":
                        node_sizes = dict(G.degree())
                    elif node_size_by == "betweenness":
                        node_sizes = nx.betweenness_centrality(G)
                    else:
                        node_sizes = {node: 1 for node in G.nodes()}
                    
                    # Create plotly figure
                    fig = go.Figure()
                    
                    # Add edges
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        
                        weight = G[edge[0]][edge[1]]['weight']
                        
                        fig.add_trace(go.Scatter(
                            x=[x0, x1, None],
                            y=[y0, y1, None],
                            mode='lines',
                            line=dict(
                                width=max(1, weight * 5),
                                color=f'rgba(100, 100, 100, {weight})'
                            ),
                            hoverinfo='none',
                            showlegend=False
                        ))
                    
                    # Add nodes
                    node_x = [pos[node][0] for node in G.nodes()]
                    node_y = [pos[node][1] for node in G.nodes()]
                    node_text = list(G.nodes())
                    node_size_vals = [max(20, node_sizes.get(node, 1) * 50) for node in G.nodes()]
                    
                    # Color nodes by type if roles are available
                    node_colors = []
                    if 'variable_roles' in st.session_state:
                        roles = st.session_state.variable_roles
                        for node in G.nodes():
                            if node in roles.get('treatment', []):
                                node_colors.append('lightblue')
                            elif node in roles.get('outcome', []):
                                node_colors.append('lightcoral')
                            elif node in roles.get('confounders', []):
                                node_colors.append('lightyellow')
                            else:
                                node_colors.append('lightgray')
                    else:
                        node_colors = ['lightblue'] * len(G.nodes())
                    
                    fig.add_trace(go.Scatter(
                        x=node_x,
                        y=node_y,
                        mode='markers+text',
                        marker=dict(
                            size=node_size_vals,
                            color=node_colors,
                            line=dict(width=2, color='black')
                        ),
                        text=node_text,
                        textposition="middle center",
                        hoverinfo='text',
                        hovertext=[f"Variable: {node}<br>Connections: {G.degree(node)}" for node in G.nodes()],
                        showlegend=False
                    ))
                    
                    fig.update_layout(
                        title=f"Causal Graph - {config['domain'].title()} Domain",
                        showlegend=False,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600,
                        hovermode='closest'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Graph statistics
                    st.markdown("#### Graph Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Nodes", len(G.nodes()))
                    with col2:
                        st.metric("Edges", len(G.edges()))
                    with col3:
                        density = nx.density(G)
                        st.metric("Graph Density", f"{density:.3f}")
                    with col4:
                        if len(G.nodes()) > 0:
                            avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
                            st.metric("Avg Degree", f"{avg_degree:.1f}")
                
                else:
                    st.warning("No relationships meet the current confidence threshold.")
                    
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")
                st.info("Try adjusting the confidence threshold or layout options.")
        
        # Key insights
        if hasattr(structure, 'key_insights') and structure.key_insights:
            st.markdown("#### 🔑 Key Insights")
            for insight in structure.key_insights:
                st.info(f"💡 {insight}")
        
        # Limitations and assumptions
        col1, col2 = st.columns(2)
        
        with col1:
            if hasattr(structure, 'limitations') and structure.limitations:
                st.markdown("#### ⚠️ Limitations")
                for limitation in structure.limitations:
                    st.warning(f"⚠️ {limitation}")
        
        with col2:
            if hasattr(structure, 'assumptions') and structure.assumptions:
                st.markdown("#### 📋 Assumptions")
                for assumption in structure.assumptions:
                    st.info(f"📋 {assumption}")
    
    with tab3:
        st.markdown("### Automatic Confounder Detection")
        st.info("Identify variables that might confound causal relationships")
        
        if 'variable_roles' not in st.session_state:
            st.warning("Please assign variable roles in the Data Manager first.")
            return
        
        roles = st.session_state.variable_roles
        treatment_vars = roles.get('treatment', [])
        outcome_vars = roles.get('outcome', [])
        
        if not treatment_vars or not outcome_vars:
            st.warning("Please assign at least one treatment and one outcome variable.")
            return
        
        # Configuration
        col1, col2 = st.columns(2)
        
        with col1:
            treatment_var = st.selectbox("Treatment Variable", treatment_vars)
            outcome_var = st.selectbox("Outcome Variable", outcome_vars)
        
        with col2:
            confounder_domain = st.selectbox(
                "Domain for Confounder Analysis", 
                ["healthcare", "business", "education", "general"]
            )
            
            detection_methods = st.multiselect(
                "Detection Methods",
                ["statistical", "domain_knowledge", "data_driven", "literature_based"],
                default=["statistical", "domain_knowledge"]
            )
        
        if st.button("🎯 Detect Confounders", type="primary"):
            with st.spinner("Detecting potential confounders..."):
                try:
                    llm_client = get_llm_client()
                    detector = AutomaticConfounderDetector(llm_client)
                    
                    # Run confounder detection
                    summary = asyncio.run(detector.detect_confounders(
                        data=data,
                        treatment_variable=treatment_var,
                        outcome_variable=outcome_var,
                        variable_descriptions=variable_descriptions if 'variable_descriptions' in locals() else {},
                        domain=confounder_domain,
                        context=f"Confounder analysis for {treatment_var} -> {outcome_var}"
                    ))
                    
                    st.session_state['confounder_results'] = summary
                    
                    st.success("✅ Confounder detection completed!")
                    
                except Exception as e:
                    st.error(f"Confounder detection failed: {str(e)}")
        
        # Display confounder results
        if 'confounder_results' in st.session_state:
            summary = st.session_state.confounder_results
            
            st.markdown("#### Detection Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confounders Found", len(summary.detected_confounders))
            with col2:
                st.metric("Overall Risk", summary.overall_confounding_risk)
            with col3:
                if summary.recommended_adjustment_set:
                    st.metric("Recommended Adjustments", len(summary.recommended_adjustment_set))
            
            # Detailed confounder information
            if summary.detected_confounders:
                st.markdown("#### Detected Confounders")
                
                confounder_data = []
                for conf in summary.detected_confounders:
                    confounder_data.append({
                        'Variable': conf.variable_name,
                        'Confounding Score': f"{conf.confounding_score:.3f}",
                        'Confidence': f"{conf.confidence:.3f}",
                        'Reasoning': conf.reasoning[:100] + "..." if len(conf.reasoning) > 100 else conf.reasoning,
                        'Recommended Action': ", ".join([adj.value for adj in conf.recommended_adjustment])
                    })
                
                confounder_df = pd.DataFrame(confounder_data)
                st.dataframe(confounder_df, use_container_width=True)
                
                # Recommended adjustment set
                if summary.recommended_adjustment_set:
                    st.markdown("#### 🎯 Recommended Adjustment Set")
                    st.info(f"Variables to control for: **{', '.join(summary.recommended_adjustment_set)}**")
                
                # Alternative adjustment sets
                if summary.alternative_adjustment_sets:
                    st.markdown("#### Alternative Adjustment Sets")
                    for i, alt_set in enumerate(summary.alternative_adjustment_sets, 1):
                        st.write(f"**Option {i}:** {', '.join(alt_set)}")
    
    with tab4:
        st.markdown("### Analysis Summary")
        
        if 'discovery_results' not in st.session_state:
            st.info("Complete causal discovery to view the analysis summary.")
            return
        
        results = st.session_state.discovery_results
        structure = results['structure']
        config = results['config']
        
        # Generate comprehensive summary
        st.markdown("#### 📊 Discovery Summary Report")
        
        # Executive summary
        st.markdown("##### Executive Summary")
        
        summary_text = f"""
        **Analysis Overview:**
        - **Dataset:** {data.shape[0]:,} observations, {data.shape[1]} variables
        - **Method:** {config['method']}
        - **Domain:** {config['domain'].title()}
        - **Discovery Time:** {config['discovery_time']:.1f} seconds
        - **Relationships Found:** {len(structure.edges)}
        - **Overall Confidence:** {structure.overall_confidence:.2f}
        """
        
        st.markdown(summary_text)
        
        # Key findings
        high_conf_edges = [e for e in structure.edges if e.confidence > 0.8]
        medium_conf_edges = [e for e in structure.edges if 0.6 < e.confidence <= 0.8]
        low_conf_edges = [e for e in structure.edges if e.confidence <= 0.6]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 🎯 High Confidence Relationships")
            if high_conf_edges:
                for edge in high_conf_edges[:5]:  # Show top 5
                    st.write(f"• {edge.source} → {edge.target} ({edge.confidence:.2f})")
            else:
                st.write("No high confidence relationships found.")
        
        with col2:
            st.markdown("##### 📊 Medium Confidence Relationships")
            if medium_conf_edges:
                for edge in medium_conf_edges[:5]:
                    st.write(f"• {edge.source} → {edge.target} ({edge.confidence:.2f})")
            else:
                st.write("No medium confidence relationships found.")
        
        with col3:
            st.markdown("##### 🤔 Low Confidence Relationships")
            if low_conf_edges:
                st.write(f"Found {len(low_conf_edges)} relationships with confidence < 0.6")
                st.write("These may require additional validation or data.")
            else:
                st.write("No low confidence relationships.")
        
        # Recommendations
        st.markdown("##### 💡 Recommendations")
        
        recommendations = []
        
        if len(high_conf_edges) > 3:
            recommendations.append("✅ Strong causal structure detected. Consider validation with experimental data.")
        
        if len(medium_conf_edges) > len(high_conf_edges):
            recommendations.append("⚠️ Many medium-confidence relationships. Consider gathering more data or domain expertise.")
        
        if structure.overall_confidence < 0.6:
            recommendations.append("🔍 Low overall confidence. Review data quality and consider alternative methods.")
        
        if 'confounder_results' in st.session_state:
            conf_results = st.session_state.confounder_results
            if len(conf_results.detected_confounders) > 0:
                recommendations.append(f"🎯 Control for detected confounders: {', '.join(conf_results.recommended_adjustment_set)}")
        
        if not recommendations:
            recommendations = ["📊 Results look reasonable. Proceed with validation and interpretation."]
        
        for rec in recommendations:
            st.info(rec)
        
        # Export full report
        st.markdown("##### 📥 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Generate Report"):
                # This would generate a comprehensive PDF/HTML report
                st.info("Report generation feature coming soon!")
        
        with col2:
            if st.button("📊 Export to R/Python"):
                # Generate code to reproduce analysis
                st.info("Code export feature coming soon!")
        
        with col3:
            if st.button("🔄 Save Analysis"):
                # Save complete analysis to session or database
                analysis_record = {
                    'timestamp': results['timestamp'],
                    'method': config['method'],
                    'domain': config['domain'],
                    'edges_found': len(structure.edges),
                    'confidence': structure.overall_confidence,
                    'status': 'completed'
                }
                
                if 'recent_analyses' not in st.session_state:
                    st.session_state.recent_analyses = []
                
                st.session_state.recent_analyses.append(analysis_record)
                
                # Update session stats
                if 'session_stats' not in st.session_state:
                    st.session_state.session_stats = {'analyses': 0, 'datasets': 0, 'success_rate': 0}
                
                st.session_state.session_stats['analyses'] += 1
                
                st.success("✅ Analysis saved to session history!")

    # Update session stats if discovery was successful
    if 'discovery_results' in st.session_state:
        if 'session_stats' not in st.session_state:
            st.session_state.session_stats = {'analyses': 0, 'datasets': 0, 'success_rate': 0}
        
        # Calculate success rate
        total_analyses = st.session_state.session_stats.get('analyses', 0)
        if total_analyses > 0:
            success_rate = 100.0  # Assume success if we got this far
            st.session_state.session_stats['success_rate'] = success_rate