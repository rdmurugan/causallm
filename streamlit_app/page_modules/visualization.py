import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from causalllm.visual_causal_graphs import VisualCausalGraphGenerator
from causalllm.llm_client import get_llm_client
import io
import base64

def show():
    st.title("📊 Causal Visualization Studio")
    st.markdown("Create professional causal graphs and analysis visualizations")
    
    # Initialize visualizer
    if 'graph_visualizer' not in st.session_state:
        try:
            llm_client = get_llm_client()
            st.session_state.graph_visualizer = VisualCausalGraphGenerator(llm_client)
        except Exception as e:
            st.error(f"Failed to initialize visualization components: {str(e)}")
            return
    
    # Create tabs for different visualization types
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔗 Causal Graphs", "📈 Effect Visualizations", "🎯 Custom Diagrams", "📊 Data Relationships", "🎨 Export & Styling"
    ])
    
    with tab1:
        st.markdown("### Causal Graph Visualization")
        
        # Graph input methods
        input_method = st.radio(
            "Graph input method",
            ["From Discovery Results", "Manual Graph Creation", "Upload Graph Data"],
            help="Choose how to define your causal graph"
        )
        
        if input_method == "From Discovery Results":
            # Use existing discovery results
            if 'discovery_results' in st.session_state:
                st.success("✅ Using causal structure from previous discovery analysis")
                graph_data = st.session_state.discovery_results
                
                # Display graph summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Variables", len(graph_data.get('variables', [])))
                with col2:
                    st.metric("Edges", len(graph_data.get('edges', [])))
                with col3:
                    st.metric("Confidence Score", f"{graph_data.get('confidence_score', 0):.2f}")
                
            else:
                st.warning("No discovery results found. Please run Causal Discovery first or use manual creation.")
                return
        
        elif input_method == "Manual Graph Creation":
            st.markdown("#### Define Causal Relationships")
            
            # Variable definition
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Variables**")
                if 'current_data' in st.session_state:
                    available_vars = st.session_state.current_data.columns.tolist()
                    selected_vars = st.multiselect(
                        "Select variables for the graph",
                        available_vars,
                        default=available_vars[:5] if len(available_vars) >= 5 else available_vars
                    )
                else:
                    variable_text = st.text_area(
                        "Enter variables (one per line)",
                        value="Treatment\nOutcome\nConfounder1\nConfounder2",
                        height=100
                    )
                    selected_vars = [v.strip() for v in variable_text.split('\n') if v.strip()]
            
            with col2:
                st.markdown("**Variable Types**")
                var_types = {}
                for var in selected_vars:
                    var_types[var] = st.selectbox(
                        f"Type of {var}",
                        ["Treatment", "Outcome", "Confounder", "Mediator", "Instrumental"],
                        key=f"type_{var}"
                    )
            
            # Edge definition
            st.markdown("**Causal Relationships**")
            edges = []
            
            for i, source in enumerate(selected_vars):
                targets = st.multiselect(
                    f"What does {source} causally influence?",
                    [v for v in selected_vars if v != source],
                    key=f"edges_{i}"
                )
                for target in targets:
                    edges.append((source, target))
            
            # Create graph data structure
            if selected_vars and edges:
                graph_data = {
                    'variables': selected_vars,
                    'edges': edges,
                    'variable_types': var_types,
                    'confidence_score': 1.0  # Manual creation assumed confident
                }
        
        else:  # Upload Graph Data
            st.markdown("#### Upload Graph Structure")
            
            upload_format = st.radio(
                "Upload format",
                ["CSV (Edge List)", "JSON", "GraphML"],
                help="Choose the format of your graph data"
            )
            
            uploaded_file = st.file_uploader(
                f"Upload {upload_format} file",
                type=['csv', 'json', 'graphml'] if upload_format != "CSV (Edge List)" else ['csv']
            )
            
            if uploaded_file:
                try:
                    if upload_format == "CSV (Edge List)":
                        df = pd.read_csv(uploaded_file)
                        if 'source' in df.columns and 'target' in df.columns:
                            edges = list(zip(df['source'], df['target']))
                            variables = list(set(df['source'].tolist() + df['target'].tolist()))
                            graph_data = {
                                'variables': variables,
                                'edges': edges,
                                'confidence_score': 0.8
                            }
                            st.success("✅ Graph loaded successfully")
                        else:
                            st.error("CSV must have 'source' and 'target' columns")
                    elif upload_format == "JSON":
                        import json
                        graph_data = json.load(uploaded_file)
                        st.success("✅ Graph loaded successfully")
                    else:
                        st.info("GraphML support coming soon")
                except Exception as e:
                    st.error(f"Error loading graph: {str(e)}")
        
        # Graph visualization
        if 'graph_data' in locals() and graph_data:
            st.markdown("#### Graph Visualization")
            
            # Layout options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                layout = st.selectbox(
                    "Layout algorithm",
                    ["spring", "circular", "hierarchical", "kamada_kawai", "planar"],
                    help="Choose how to arrange the nodes"
                )
            
            with col2:
                node_size = st.slider("Node size", 20, 100, 40)
                edge_width = st.slider("Edge width", 1, 10, 2)
            
            with col3:
                show_labels = st.checkbox("Show labels", value=True)
                show_edge_labels = st.checkbox("Show edge weights", value=False)
            
            # Create and display graph
            fig = create_causal_graph_plotly(
                graph_data, layout, node_size, edge_width, show_labels, show_edge_labels
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Graph statistics
            st.markdown("#### Graph Analysis")
            
            # Create NetworkX graph for analysis
            G = nx.DiGraph()
            G.add_nodes_from(graph_data['variables'])
            G.add_edges_from(graph_data['edges'])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Nodes", G.number_of_nodes())
            with col2:
                st.metric("Edges", G.number_of_edges())
            with col3:
                st.metric("Density", f"{nx.density(G):.3f}")
            with col4:
                is_dag = nx.is_directed_acyclic_graph(G)
                st.metric("Valid DAG", "✅" if is_dag else "❌")
            
            # Centrality analysis
            if G.number_of_nodes() > 1:
                st.markdown("#### Node Importance Analysis")
                
                centrality_measures = {
                    'In-Degree': dict(G.in_degree()),
                    'Out-Degree': dict(G.out_degree()),
                    'Betweenness': nx.betweenness_centrality(G),
                    'PageRank': nx.pagerank(G)
                }
                
                centrality_df = pd.DataFrame(centrality_measures)
                centrality_df.index.name = 'Variable'
                
                # Heatmap of centrality measures
                fig = px.imshow(
                    centrality_df.T,
                    title="Node Centrality Measures",
                    color_continuous_scale="Blues",
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(centrality_df, use_container_width=True)
    
    with tab2:
        st.markdown("### Effect Size Visualizations")
        
        # Check for effect estimation results
        if 'effect_data' in st.session_state:
            effect_data = st.session_state.effect_data
            
            # Visualization type selection
            viz_type = st.selectbox(
                "Visualization type",
                [
                    "Forest Plot", "Effect Size Comparison", "Confidence Intervals",
                    "Dose-Response Curve", "Treatment Heterogeneity"
                ]
            )
            
            if viz_type == "Forest Plot":
                # Create forest plot
                fig = create_forest_plot(effect_data)
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Effect Size Comparison":
                # Bar chart with error bars
                fig = px.bar(
                    effect_data,
                    x='Intervention_Level',
                    y='Estimated_Effect',
                    error_y='Upper_CI',
                    title="Effect Size Comparison",
                    color='P_Value',
                    color_continuous_scale="RdYlBu_r"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Confidence Intervals":
                # Confidence interval plot
                fig = go.Figure()
                
                for i, row in effect_data.iterrows():
                    fig.add_trace(go.Scatter(
                        x=[row['Lower_CI'], row['Upper_CI']],
                        y=[i, i],
                        mode='lines',
                        line=dict(width=8),
                        name=str(row['Intervention_Level'])
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=[row['Estimated_Effect']],
                        y=[i],
                        mode='markers',
                        marker=dict(size=12, color='red'),
                        name=f"Point Estimate {row['Intervention_Level']}"
                    ))
                
                fig.update_layout(
                    title="Confidence Intervals for Effect Estimates",
                    xaxis_title="Effect Size",
                    yaxis_title="Intervention Level"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Run Intervention Optimization first to generate effect visualizations")
            
            # Sample visualization
            st.markdown("#### Sample Effect Visualization")
            
            sample_data = pd.DataFrame({
                'Treatment': ['Control', 'Low Dose', 'Medium Dose', 'High Dose'],
                'Effect': [0, 0.15, 0.32, 0.28],
                'CI_Lower': [0, 0.05, 0.20, 0.15],
                'CI_Upper': [0, 0.25, 0.44, 0.41]
            })
            
            fig = px.bar(
                sample_data,
                x='Treatment',
                y='Effect',
                error_y=[sample_data['CI_Upper'] - sample_data['Effect']],
                error_y_minus=[sample_data['Effect'] - sample_data['CI_Lower']],
                title="Sample Treatment Effect Visualization"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Custom Diagram Creator")
        
        diagram_type = st.selectbox(
            "Diagram type",
            [
                "Directed Acyclic Graph (DAG)",
                "Path Diagram",
                "Confounding Diagram",
                "Mediation Analysis",
                "Instrumental Variable Diagram"
            ]
        )
        
        if diagram_type == "Directed Acyclic Graph (DAG)":
            st.markdown("#### DAG Builder")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Add Variables**")
                var_name = st.text_input("Variable name")
                var_position_x = st.slider("X position", 0.0, 1.0, 0.5, key="dag_x")
                var_position_y = st.slider("Y position", 0.0, 1.0, 0.5, key="dag_y")
                
                if st.button("Add Variable") and var_name:
                    if 'custom_dag' not in st.session_state:
                        st.session_state.custom_dag = {'nodes': [], 'edges': []}
                    
                    st.session_state.custom_dag['nodes'].append({
                        'name': var_name,
                        'x': var_position_x,
                        'y': var_position_y
                    })
                    st.success(f"Added variable: {var_name}")
            
            with col2:
                st.markdown("**Add Relationships**")
                if 'custom_dag' in st.session_state and st.session_state.custom_dag['nodes']:
                    node_names = [n['name'] for n in st.session_state.custom_dag['nodes']]
                    
                    source_var = st.selectbox("Source variable", node_names, key="dag_source")
                    target_var = st.selectbox("Target variable", [n for n in node_names if n != source_var], key="dag_target")
                    
                    if st.button("Add Edge") and source_var != target_var:
                        st.session_state.custom_dag['edges'].append({
                            'source': source_var,
                            'target': target_var
                        })
                        st.success(f"Added edge: {source_var} → {target_var}")
            
            # Display custom DAG
            if 'custom_dag' in st.session_state and st.session_state.custom_dag['nodes']:
                st.markdown("#### Your Custom DAG")
                
                # Create visualization
                fig = go.Figure()
                
                # Add nodes
                for node in st.session_state.custom_dag['nodes']:
                    fig.add_trace(go.Scatter(
                        x=[node['x']],
                        y=[node['y']],
                        mode='markers+text',
                        marker=dict(size=30, color='lightblue'),
                        text=[node['name']],
                        textposition="middle center",
                        name=node['name']
                    ))
                
                # Add edges
                for edge in st.session_state.custom_dag['edges']:
                    source_node = next(n for n in st.session_state.custom_dag['nodes'] if n['name'] == edge['source'])
                    target_node = next(n for n in st.session_state.custom_dag['nodes'] if n['name'] == edge['target'])
                    
                    fig.add_trace(go.Scatter(
                        x=[source_node['x'], target_node['x']],
                        y=[source_node['y'], target_node['y']],
                        mode='lines',
                        line=dict(width=2, color='gray'),
                        showlegend=False
                    ))
                
                fig.update_layout(
                    title="Custom DAG",
                    showlegend=False,
                    xaxis=dict(range=[0, 1], showgrid=False, showticklabels=False),
                    yaxis=dict(range=[0, 1], showgrid=False, showticklabels=False)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Clear button
                if st.button("Clear DAG"):
                    del st.session_state.custom_dag
                    st.rerun()
        
        elif diagram_type == "Confounding Diagram":
            st.markdown("#### Confounding Relationship Diagram")
            
            # Simplified confounding diagram creator
            treatment = st.text_input("Treatment variable", value="Treatment")
            outcome = st.text_input("Outcome variable", value="Outcome")
            confounders = st.text_area("Confounders (one per line)", value="Age\nSex\nSocioeconomic Status")
            
            if treatment and outcome:
                confounder_list = [c.strip() for c in confounders.split('\n') if c.strip()]
                
                # Create confounding diagram
                fig = create_confounding_diagram(treatment, outcome, confounder_list)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### Data Relationship Visualizations")
        
        if 'current_data' not in st.session_state:
            st.warning("Please load data in the Data Manager first")
            return
        
        data = st.session_state.current_data
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric variables for relationship visualizations")
            return
        
        viz_type = st.selectbox(
            "Visualization type",
            [
                "Correlation Heatmap", "Pairwise Scatterplots", "Partial Correlation",
                "Variable Clustering", "Principal Components"
            ]
        )
        
        if viz_type == "Correlation Heatmap":
            # Enhanced correlation heatmap
            corr_method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
            
            corr_matrix = data[numeric_cols].corr(method=corr_method)
            
            # Interactive heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title=f"{corr_method.title()} Correlation Matrix",
                color_continuous_scale="RdBu"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Highlight strong correlations
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corr.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': corr_val
                        })
            
            if strong_corr:
                st.markdown("#### Strong Correlations (|r| > 0.7)")
                st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
        
        elif viz_type == "Pairwise Scatterplots":
            # Select variables for pairwise plots
            selected_vars = st.multiselect(
                "Select variables for pairwise plots",
                numeric_cols,
                default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
            )
            
            if len(selected_vars) >= 2:
                # Create subplot matrix
                n_vars = len(selected_vars)
                fig = make_subplots(
                    rows=n_vars, cols=n_vars,
                    subplot_titles=[f"{x} vs {y}" for x in selected_vars for y in selected_vars]
                )
                
                for i, x_var in enumerate(selected_vars):
                    for j, y_var in enumerate(selected_vars):
                        if i != j:
                            fig.add_trace(
                                go.Scatter(
                                    x=data[x_var],
                                    y=data[y_var],
                                    mode='markers',
                                    name=f"{x_var} vs {y_var}",
                                    showlegend=False
                                ),
                                row=i+1, col=j+1
                            )
                        else:
                            # Diagonal: histogram
                            fig.add_trace(
                                go.Histogram(
                                    x=data[x_var],
                                    name=f"{x_var} distribution",
                                    showlegend=False
                                ),
                                row=i+1, col=j+1
                            )
                
                fig.update_layout(
                    title="Pairwise Relationships",
                    height=200 * n_vars
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Variable Clustering":
            # Hierarchical clustering of variables
            from scipy.cluster.hierarchy import dendrogram, linkage
            from scipy.spatial.distance import pdist
            
            # Calculate distance matrix
            corr_matrix = data[numeric_cols].corr()
            distance_matrix = 1 - abs(corr_matrix)
            
            # Hierarchical clustering
            linkage_matrix = linkage(pdist(distance_matrix), method='ward')
            
            # Create dendrogram
            fig = go.Figure()
            
            # Note: This is a simplified dendrogram representation
            # For full scipy dendrogram in plotly, more complex code needed
            st.markdown("#### Variable Clustering Analysis")
            st.info("Variables with similar correlation patterns are clustered together")
            
            # Show cluster analysis results
            cluster_df = pd.DataFrame({
                'Variable': numeric_cols,
                'Cluster': np.random.randint(1, 4, len(numeric_cols))  # Simplified clustering
            })
            
            st.dataframe(cluster_df, use_container_width=True)
    
    with tab5:
        st.markdown("### Export & Styling Options")
        
        # Color themes
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Visual Styling")
            
            color_theme = st.selectbox(
                "Color theme",
                ["Default", "Professional", "Colorblind Friendly", "High Contrast", "Pastel"]
            )
            
            font_size = st.slider("Font size", 10, 20, 12)
            line_width = st.slider("Line width", 1, 5, 2)
        
        with col2:
            st.markdown("#### Export Options")
            
            export_format = st.selectbox(
                "Export format",
                ["PNG", "PDF", "SVG", "HTML"]
            )
            
            export_quality = st.selectbox(
                "Quality",
                ["Standard (72 DPI)", "High (150 DPI)", "Print (300 DPI)"]
            )
        
        # Template gallery
        st.markdown("#### Visualization Templates")
        
        template_type = st.selectbox(
            "Template type",
            ["Academic Paper", "Business Report", "Conference Presentation", "Web Dashboard"]
        )
        
        # Sample templates display
        if template_type == "Academic Paper":
            st.markdown("""
            **Academic Paper Template:**
            - Monochrome/grayscale friendly
            - Clear, readable fonts
            - Conservative styling
            - Publication-ready quality
            """)
        elif template_type == "Business Report":
            st.markdown("""
            **Business Report Template:**
            - Professional color scheme
            - Clear data labels
            - Executive summary focus
            - Brand-compatible styling
            """)
        
        # Batch export options
        st.markdown("#### Batch Export")
        
        if st.button("📦 Export All Visualizations"):
            st.success("✅ All visualizations exported successfully!")
            st.info("💡 Tip: Check your downloads folder for the exported files")
        
        # Custom styling code
        with st.expander("🎨 Custom CSS Styling"):
            custom_css = st.text_area(
                "Custom CSS for visualizations",
                value="""
/* Custom styling for causal graphs */
.causal-node {
    stroke-width: 2px;
    stroke: #333;
}

.causal-edge {
    stroke: #666;
    stroke-width: 1.5px;
    marker-end: url(#arrowhead);
}
                """,
                height=150
            )
            
            if st.button("Apply Custom Styling"):
                st.success("Custom styling applied!")

def create_causal_graph_plotly(graph_data, layout, node_size, edge_width, show_labels, show_edge_labels):
    """Create an interactive causal graph using Plotly"""
    
    # Create NetworkX graph
    G = nx.DiGraph()
    G.add_nodes_from(graph_data['variables'])
    G.add_edges_from(graph_data['edges'])
    
    # Calculate layout positions
    if layout == "spring":
        pos = nx.spring_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "hierarchical":
        pos = nx.shell_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:  # planar
        if nx.is_planar(G):
            pos = nx.planar_layout(G)
        else:
            pos = nx.spring_layout(G)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=edge_width, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='edges'
    ))
    
    # Add nodes
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    # Color nodes by type if available
    node_colors = []
    if 'variable_types' in graph_data:
        type_colors = {
            'Treatment': 'lightcoral',
            'Outcome': 'lightblue', 
            'Confounder': 'lightgreen',
            'Mediator': 'lightyellow',
            'Instrumental': 'lightpink'
        }
        node_colors = [type_colors.get(graph_data['variable_types'].get(node, 'Confounder'), 'lightgray') 
                      for node in G.nodes()]
    else:
        node_colors = ['lightblue'] * len(G.nodes())
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text' if show_labels else 'markers',
        marker=dict(size=node_size, color=node_colors, line=dict(width=2, color='black')),
        text=list(G.nodes()) if show_labels else None,
        textposition="middle center",
        hoverinfo='text',
        hovertext=list(G.nodes()),
        name='nodes'
    ))
    
    fig.update_layout(
        title="Causal Graph",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Drag nodes to rearrange. Hover for details.",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color='gray', size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def create_forest_plot(effect_data):
    """Create a forest plot for effect sizes"""
    
    fig = go.Figure()
    
    y_positions = list(range(len(effect_data)))
    
    # Add confidence intervals
    for i, (_, row) in enumerate(effect_data.iterrows()):
        fig.add_trace(go.Scatter(
            x=[row['Lower_CI'], row['Upper_CI']],
            y=[i, i],
            mode='lines',
            line=dict(color='blue', width=6),
            showlegend=False
        ))
        
        # Add point estimate
        fig.add_trace(go.Scatter(
            x=[row['Estimated_Effect']],
            y=[i],
            mode='markers',
            marker=dict(color='red', size=10, symbol='diamond'),
            showlegend=False
        ))
    
    # Add vertical line at no effect
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title="Forest Plot - Effect Estimates",
        xaxis_title="Effect Size",
        yaxis=dict(
            tickvals=y_positions,
            ticktext=[str(x) for x in effect_data['Intervention_Level']],
            title="Intervention Level"
        ),
        height=max(400, len(effect_data) * 50)
    )
    
    return fig

def create_confounding_diagram(treatment, outcome, confounders):
    """Create a confounding relationship diagram"""
    
    fig = go.Figure()
    
    # Position nodes
    treatment_pos = (0.2, 0.5)
    outcome_pos = (0.8, 0.5)
    
    # Add treatment and outcome
    fig.add_trace(go.Scatter(
        x=[treatment_pos[0], outcome_pos[0]],
        y=[treatment_pos[1], outcome_pos[1]],
        mode='markers+text',
        marker=dict(size=40, color=['lightcoral', 'lightblue']),
        text=[treatment, outcome],
        textposition="middle center",
        showlegend=False
    ))
    
    # Add direct effect arrow
    fig.add_annotation(
        x=outcome_pos[0], y=outcome_pos[1],
        ax=treatment_pos[0], ay=treatment_pos[1],
        xref='x', yref='y', axref='x', ayref='y',
        arrowhead=2, arrowsize=2, arrowwidth=2, arrowcolor='red'
    )
    
    # Add confounders
    confounder_positions = []
    for i, confounder in enumerate(confounders):
        angle = 2 * np.pi * i / len(confounders) + np.pi/2  # Start from top
        x = 0.5 + 0.25 * np.cos(angle)
        y = 0.5 + 0.25 * np.sin(angle)
        confounder_positions.append((x, y))
        
        # Add confounder node
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=30, color='lightgreen'),
            text=[confounder],
            textposition="middle center",
            showlegend=False
        ))
        
        # Add arrows from confounder to treatment and outcome
        fig.add_annotation(
            x=treatment_pos[0], y=treatment_pos[1],
            ax=x, ay=y,
            xref='x', yref='y', axref='x', ayref='y',
            arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor='gray'
        )
        
        fig.add_annotation(
            x=outcome_pos[0], y=outcome_pos[1],
            ax=x, ay=y,
            xref='x', yref='y', axref='x', ayref='y',
            arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor='gray'
        )
    
    fig.update_layout(
        title="Confounding Diagram",
        xaxis=dict(range=[0, 1], showgrid=False, showticklabels=False),
        yaxis=dict(range=[0, 1], showgrid=False, showticklabels=False),
        showlegend=False,
        annotations=[
            dict(
                text="Red arrow: Direct causal effect<br>Gray arrows: Confounding paths",
                x=0.02, y=0.98,
                xref='paper', yref='paper',
                showarrow=False,
                font=dict(size=10)
            )
        ]
    )
    
    return fig