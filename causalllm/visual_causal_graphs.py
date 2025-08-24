"""
Visual Causal Graph Generation System

This module provides comprehensive visualization capabilities for causal graphs,
including interactive and static visualizations, different layout algorithms,
and customizable styling for different types of causal relationships.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from causalllm.logging import get_logger


class GraphLayout(Enum):
    """Graph layout algorithms."""
    HIERARCHICAL = "hierarchical"
    CIRCULAR = "circular"
    SPRING = "spring"
    SHELL = "shell"
    SPECTRAL = "spectral"
    PLANAR = "planar"
    KAMADA_KAWAI = "kamada_kawai"


class NodeType(Enum):
    """Types of nodes in causal graphs."""
    TREATMENT = "treatment"
    OUTCOME = "outcome"
    CONFOUNDER = "confounder"
    MEDIATOR = "mediator"
    COLLIDER = "collider"
    INSTRUMENTAL = "instrumental"
    COVARIATE = "covariate"
    UNOBSERVED = "unobserved"


class EdgeType(Enum):
    """Types of edges in causal graphs."""
    CAUSAL = "causal"
    CONFOUNDING = "confounding"
    SELECTION = "selection"
    MEASUREMENT = "measurement"
    UNCERTAIN = "uncertain"


class VisualizationType(Enum):
    """Types of visualizations."""
    STATIC_MATPLOTLIB = "static_matplotlib"
    INTERACTIVE_PLOTLY = "interactive_plotly"
    NETWORK_DIAGRAM = "network_diagram"
    DAG_DIAGRAM = "dag_diagram"
    CONFOUNDING_DIAGRAM = "confounding_diagram"


@dataclass
class NodeStyle:
    """Styling for graph nodes."""
    
    color: str = "#1f77b4"
    size: float = 500
    shape: str = "circle"  # circle, square, diamond, triangle
    border_color: str = "black"
    border_width: float = 2
    label_color: str = "black"
    label_size: float = 12
    alpha: float = 1.0


@dataclass
class EdgeStyle:
    """Styling for graph edges."""
    
    color: str = "black"
    width: float = 2
    style: str = "solid"  # solid, dashed, dotted
    arrow_size: float = 20
    alpha: float = 1.0
    curve: float = 0  # For curved edges


@dataclass
class GraphTheme:
    """Complete visual theme for causal graphs."""
    
    name: str
    node_styles: Dict[str, NodeStyle]
    edge_styles: Dict[str, EdgeStyle]
    background_color: str = "white"
    text_color: str = "black"
    title_size: float = 16
    legend_position: str = "upper right"


class VisualCausalGraphGenerator:
    """Generator for visual causal graphs with multiple output formats."""
    
    def __init__(self):
        self.logger = get_logger("causalllm.visual_causal_graphs")
        
        # Initialize themes
        self.themes = self._initialize_themes()
        self.current_theme = self.themes["default"]
        
        # Layout algorithms
        self.layout_functions = {
            GraphLayout.HIERARCHICAL: self._hierarchical_layout,
            GraphLayout.CIRCULAR: nx.circular_layout,
            GraphLayout.SPRING: nx.spring_layout,
            GraphLayout.SHELL: nx.shell_layout,
            GraphLayout.SPECTRAL: nx.spectral_layout,
            GraphLayout.KAMADA_KAWAI: nx.kamada_kawai_layout
        }
    
    def _initialize_themes(self) -> Dict[str, GraphTheme]:
        """Initialize built-in visual themes."""
        
        themes = {}
        
        # Default theme
        themes["default"] = GraphTheme(
            name="Default",
            node_styles={
                NodeType.TREATMENT.value: NodeStyle("#2ca02c", 600, "square"),  # Green square
                NodeType.OUTCOME.value: NodeStyle("#d62728", 600, "diamond"),   # Red diamond
                NodeType.CONFOUNDER.value: NodeStyle("#ff7f0e", 500, "circle"), # Orange circle
                NodeType.MEDIATOR.value: NodeStyle("#1f77b4", 500, "circle"),   # Blue circle
                NodeType.COLLIDER.value: NodeStyle("#9467bd", 500, "triangle"), # Purple triangle
                NodeType.INSTRUMENTAL.value: NodeStyle("#8c564b", 450, "circle"), # Brown circle
                NodeType.COVARIATE.value: NodeStyle("#e377c2", 400, "circle"),  # Pink circle
                NodeType.UNOBSERVED.value: NodeStyle("#7f7f7f", 500, "circle", alpha=0.7)  # Gray circle
            },
            edge_styles={
                EdgeType.CAUSAL.value: EdgeStyle("black", 3, "solid"),
                EdgeType.CONFOUNDING.value: EdgeStyle("red", 2, "dashed"),
                EdgeType.SELECTION.value: EdgeStyle("blue", 2, "dotted"),
                EdgeType.UNCERTAIN.value: EdgeStyle("gray", 1, "dashed", alpha=0.7)
            }
        )
        
        # Academic theme (black and white for publications)
        themes["academic"] = GraphTheme(
            name="Academic",
            node_styles={
                NodeType.TREATMENT.value: NodeStyle("white", 600, "square", "black", 3),
                NodeType.OUTCOME.value: NodeStyle("black", 600, "diamond", "black", 3),
                NodeType.CONFOUNDER.value: NodeStyle("lightgray", 500, "circle", "black", 2),
                NodeType.MEDIATOR.value: NodeStyle("white", 500, "circle", "black", 2),
                NodeType.COLLIDER.value: NodeStyle("gray", 500, "triangle", "black", 2),
                NodeType.INSTRUMENTAL.value: NodeStyle("white", 450, "circle", "black", 1),
                NodeType.COVARIATE.value: NodeStyle("lightgray", 400, "circle", "black", 1),
                NodeType.UNOBSERVED.value: NodeStyle("white", 500, "circle", "gray", 2, alpha=0.7)
            },
            edge_styles={
                EdgeType.CAUSAL.value: EdgeStyle("black", 3, "solid"),
                EdgeType.CONFOUNDING.value: EdgeStyle("black", 2, "dashed"),
                EdgeType.SELECTION.value: EdgeStyle("gray", 2, "dotted"),
                EdgeType.UNCERTAIN.value: EdgeStyle("gray", 1, "dashed", alpha=0.5)
            },
            background_color="white",
            text_color="black"
        )
        
        # Healthcare theme
        themes["healthcare"] = GraphTheme(
            name="Healthcare",
            node_styles={
                NodeType.TREATMENT.value: NodeStyle("#0077be", 600, "square"),   # Medical blue
                NodeType.OUTCOME.value: NodeStyle("#dc143c", 600, "diamond"),    # Crimson red
                NodeType.CONFOUNDER.value: NodeStyle("#ff8c00", 500, "circle"),  # Dark orange
                NodeType.MEDIATOR.value: NodeStyle("#32cd32", 500, "circle"),    # Lime green
                NodeType.COLLIDER.value: NodeStyle("#9932cc", 500, "triangle"),  # Dark orchid
                NodeType.INSTRUMENTAL.value: NodeStyle("#8b4513", 450, "circle"), # Saddle brown
                NodeType.COVARIATE.value: NodeStyle("#20b2aa", 400, "circle"),   # Light sea green
                NodeType.UNOBSERVED.value: NodeStyle("#696969", 500, "circle", alpha=0.7)  # Dim gray
            },
            edge_styles={
                EdgeType.CAUSAL.value: EdgeStyle("#0077be", 3, "solid"),
                EdgeType.CONFOUNDING.value: EdgeStyle("#dc143c", 2, "dashed"),
                EdgeType.SELECTION.value: EdgeStyle("#ff8c00", 2, "dotted"),
                EdgeType.UNCERTAIN.value: EdgeStyle("#696969", 1, "dashed", alpha=0.7)
            },
            background_color="#f8f9fa"
        )
        
        return themes
    
    def create_causal_graph_visualization(self,
                                        graph: nx.Graph,
                                        node_types: Optional[Dict[str, str]] = None,
                                        edge_types: Optional[Dict[Tuple[str, str], str]] = None,
                                        layout: GraphLayout = GraphLayout.SPRING,
                                        theme: str = "default",
                                        title: str = "Causal Graph",
                                        output_format: VisualizationType = VisualizationType.STATIC_MATPLOTLIB,
                                        save_path: Optional[str] = None,
                                        **kwargs) -> Any:
        """
        Create a comprehensive causal graph visualization.
        
        Args:
            graph: NetworkX graph to visualize
            node_types: Dictionary mapping node names to types
            edge_types: Dictionary mapping edge tuples to types
            layout: Layout algorithm to use
            theme: Visual theme name
            title: Graph title
            output_format: Type of visualization to create
            save_path: Optional path to save the visualization
            **kwargs: Additional arguments for specific visualization types
            
        Returns:
            Visualization object (matplotlib figure or plotly figure)
        """
        self.logger.info(f"Creating causal graph visualization with {output_format.value} format")
        
        # Set theme
        if theme in self.themes:
            self.current_theme = self.themes[theme]
        
        # Assign default node types if not provided
        if node_types is None:
            node_types = self._infer_node_types(graph)
        
        # Assign default edge types if not provided
        if edge_types is None:
            edge_types = self._infer_edge_types(graph, node_types)
        
        # Create visualization based on format
        if output_format == VisualizationType.STATIC_MATPLOTLIB:
            fig = self._create_matplotlib_visualization(
                graph, node_types, edge_types, layout, title, **kwargs
            )
        elif output_format == VisualizationType.INTERACTIVE_PLOTLY:
            if not PLOTLY_AVAILABLE:
                self.logger.warning("Plotly not available, falling back to matplotlib")
                fig = self._create_matplotlib_visualization(
                    graph, node_types, edge_types, layout, title, **kwargs
                )
            else:
                fig = self._create_plotly_visualization(
                    graph, node_types, edge_types, layout, title, **kwargs
                )
        elif output_format == VisualizationType.NETWORK_DIAGRAM:
            fig = self._create_network_diagram(
                graph, node_types, edge_types, layout, title, **kwargs
            )
        else:
            # Default to matplotlib
            fig = self._create_matplotlib_visualization(
                graph, node_types, edge_types, layout, title, **kwargs
            )
        
        # Save if path provided
        if save_path:
            self._save_visualization(fig, save_path, output_format)
        
        self.logger.info("Causal graph visualization created successfully")
        return fig
    
    def _infer_node_types(self, graph: nx.Graph) -> Dict[str, str]:
        """Infer node types from graph structure and attributes."""
        
        node_types = {}
        
        for node in graph.nodes():
            # Check if node has type attribute
            if 'causal_role' in graph.nodes[node]:
                role = graph.nodes[node]['causal_role']
                if role in [nt.value for nt in NodeType]:
                    node_types[node] = role
                else:
                    node_types[node] = NodeType.COVARIATE.value
            else:
                # Infer from graph structure
                in_degree = graph.in_degree(node) if isinstance(graph, nx.DiGraph) else len(list(graph.neighbors(node)))
                out_degree = graph.out_degree(node) if isinstance(graph, nx.DiGraph) else len(list(graph.neighbors(node)))
                
                if in_degree == 0 and out_degree > 0:
                    node_types[node] = NodeType.TREATMENT.value
                elif out_degree == 0 and in_degree > 0:
                    node_types[node] = NodeType.OUTCOME.value
                elif in_degree > 1 and out_degree == 0:
                    node_types[node] = NodeType.COLLIDER.value
                elif in_degree > 0 and out_degree > 0:
                    node_types[node] = NodeType.MEDIATOR.value
                else:
                    node_types[node] = NodeType.COVARIATE.value
        
        return node_types
    
    def _infer_edge_types(self, 
                         graph: nx.Graph, 
                         node_types: Dict[str, str]) -> Dict[Tuple[str, str], str]:
        """Infer edge types from node types and graph attributes."""
        
        edge_types = {}
        
        for edge in graph.edges():
            source, target = edge
            
            # Check if edge has type attribute
            if 'edge_type' in graph.edges[edge]:
                edge_types[edge] = graph.edges[edge]['edge_type']
            else:
                # Infer from node types
                source_type = node_types.get(source, NodeType.COVARIATE.value)
                target_type = node_types.get(target, NodeType.COVARIATE.value)
                
                if (source_type == NodeType.CONFOUNDER.value and 
                    target_type in [NodeType.TREATMENT.value, NodeType.OUTCOME.value]):
                    edge_types[edge] = EdgeType.CONFOUNDING.value
                elif (source_type == NodeType.TREATMENT.value and 
                      target_type == NodeType.OUTCOME.value):
                    edge_types[edge] = EdgeType.CAUSAL.value
                else:
                    edge_types[edge] = EdgeType.CAUSAL.value
        
        return edge_types
    
    def _hierarchical_layout(self, graph: nx.Graph) -> Dict[str, Tuple[float, float]]:
        """Create hierarchical layout based on causal ordering."""
        
        pos = {}
        
        if isinstance(graph, nx.DiGraph):
            # Use topological sort for DAGs
            try:
                layers = list(nx.topological_generations(graph))
                
                for layer_idx, layer in enumerate(layers):
                    y = len(layers) - layer_idx - 1  # Top to bottom
                    
                    for node_idx, node in enumerate(layer):
                        x = node_idx - len(layer) / 2 + 0.5
                        pos[node] = (x, y)
                
                return pos
                
            except nx.NetworkXError:
                # Graph has cycles, fall back to spring layout
                return nx.spring_layout(graph)
        else:
            # For undirected graphs, use spring layout
            return nx.spring_layout(graph)
    
    def _create_matplotlib_visualization(self,
                                       graph: nx.Graph,
                                       node_types: Dict[str, str],
                                       edge_types: Dict[Tuple[str, str], str],
                                       layout: GraphLayout,
                                       title: str,
                                       **kwargs) -> plt.Figure:
        """Create matplotlib visualization."""
        
        # Set up figure
        fig, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize', (12, 8)))
        fig.patch.set_facecolor(self.current_theme.background_color)
        ax.set_facecolor(self.current_theme.background_color)
        
        # Calculate layout
        layout_func = self.layout_functions.get(layout, nx.spring_layout)
        pos = layout_func(graph)
        
        # Draw edges by type
        edge_types_present = set(edge_types.values())
        
        for edge_type in edge_types_present:
            edges_of_type = [edge for edge, etype in edge_types.items() if etype == edge_type]
            
            if edges_of_type:
                edge_style = self.current_theme.edge_styles.get(
                    edge_type, self.current_theme.edge_styles[EdgeType.CAUSAL.value]
                )
                
                # Convert style string to matplotlib linestyle
                linestyle_map = {
                    "solid": "-",
                    "dashed": "--",
                    "dotted": ":",
                    "dashdot": "-."
                }
                linestyle = linestyle_map.get(edge_style.style, "-")
                
                if isinstance(graph, nx.DiGraph):
                    nx.draw_networkx_edges(
                        graph, pos, edgelist=edges_of_type,
                        edge_color=edge_style.color,
                        width=edge_style.width,
                        alpha=edge_style.alpha,
                        style=linestyle,
                        arrowsize=edge_style.arrow_size,
                        ax=ax
                    )
                else:
                    nx.draw_networkx_edges(
                        graph, pos, edgelist=edges_of_type,
                        edge_color=edge_style.color,
                        width=edge_style.width,
                        alpha=edge_style.alpha,
                        style=linestyle,
                        ax=ax
                    )
        
        # Draw nodes by type
        node_types_present = set(node_types.values())
        
        for node_type in node_types_present:
            nodes_of_type = [node for node, ntype in node_types.items() if ntype == node_type]
            
            if nodes_of_type:
                node_style = self.current_theme.node_styles.get(
                    node_type, self.current_theme.node_styles[NodeType.COVARIATE.value]
                )
                
                # Convert shape string to matplotlib node shape
                shape_map = {
                    "circle": "o",
                    "square": "s",
                    "diamond": "D",
                    "triangle": "^"
                }
                node_shape = shape_map.get(node_style.shape, "o")
                
                nx.draw_networkx_nodes(
                    graph, pos, nodelist=nodes_of_type,
                    node_color=node_style.color,
                    node_size=node_style.size,
                    node_shape=node_shape,
                    edgecolors=node_style.border_color,
                    linewidths=node_style.border_width,
                    alpha=node_style.alpha,
                    ax=ax
                )
        
        # Draw labels
        nx.draw_networkx_labels(
            graph, pos,
            font_size=12,
            font_color=self.current_theme.text_color,
            ax=ax
        )
        
        # Create legend
        self._add_matplotlib_legend(ax, node_types_present, edge_types_present)
        
        # Set title and clean up axes
        ax.set_title(title, fontsize=self.current_theme.title_size, 
                    color=self.current_theme.text_color, pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def _create_plotly_visualization(self,
                                   graph: nx.Graph,
                                   node_types: Dict[str, str],
                                   edge_types: Dict[Tuple[str, str], str],
                                   layout: GraphLayout,
                                   title: str,
                                   **kwargs) -> go.Figure:
        """Create interactive Plotly visualization."""
        
        # Calculate layout
        layout_func = self.layout_functions.get(layout, nx.spring_layout)
        pos = layout_func(graph)
        
        # Prepare data
        node_trace = []
        edge_trace = []
        
        # Create edge traces
        edge_types_present = set(edge_types.values())
        
        for edge_type in edge_types_present:
            edge_style = self.current_theme.edge_styles.get(
                edge_type, self.current_theme.edge_styles[EdgeType.CAUSAL.value]
            )
            
            edges_of_type = [edge for edge, etype in edge_types.items() if etype == edge_type]
            
            edge_x = []
            edge_y = []
            
            for edge in edges_of_type:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace.append(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=edge_style.width, color=edge_style.color),
                hoverinfo='none',
                mode='lines',
                name=f'{edge_type.replace("_", " ").title()} Edges'
            ))
        
        # Create node traces
        node_types_present = set(node_types.values())
        
        for node_type in node_types_present:
            nodes_of_type = [node for node, ntype in node_types.items() if ntype == node_type]
            
            if nodes_of_type:
                node_style = self.current_theme.node_styles.get(
                    node_type, self.current_theme.node_styles[NodeType.COVARIATE.value]
                )
                
                node_x = [pos[node][0] for node in nodes_of_type]
                node_y = [pos[node][1] for node in nodes_of_type]
                
                # Convert shape
                symbol_map = {
                    "circle": "circle",
                    "square": "square",
                    "diamond": "diamond",
                    "triangle": "triangle-up"
                }
                symbol = symbol_map.get(node_style.shape, "circle")
                
                node_trace.append(go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=nodes_of_type,
                    textposition="middle center",
                    hoverinfo='text',
                    hovertext=[f'{node} ({node_type.replace("_", " ")})' for node in nodes_of_type],
                    marker=dict(
                        size=node_style.size // 20,  # Scale down for Plotly
                        color=node_style.color,
                        symbol=symbol,
                        line=dict(width=node_style.border_width, color=node_style.border_color)
                    ),
                    name=f'{node_type.replace("_", " ").title()} Nodes'
                ))
        
        # Create figure
        fig = go.Figure(data=edge_trace + node_trace,
                       layout=go.Layout(
                           title=title,
                           titlefont_size=16,
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Interactive Causal Graph - Click and drag to explore",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="#888", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor=self.current_theme.background_color,
                           paper_bgcolor=self.current_theme.background_color
                       ))
        
        return fig
    
    def _create_network_diagram(self,
                              graph: nx.Graph,
                              node_types: Dict[str, str],
                              edge_types: Dict[Tuple[str, str], str],
                              layout: GraphLayout,
                              title: str,
                              **kwargs) -> plt.Figure:
        """Create specialized network diagram with enhanced annotations."""
        
        fig, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize', (14, 10)))
        fig.patch.set_facecolor(self.current_theme.background_color)
        
        # Calculate layout
        layout_func = self.layout_functions.get(layout, nx.spring_layout)
        pos = layout_func(graph)
        
        # Draw with enhanced styling
        self._draw_enhanced_network(ax, graph, pos, node_types, edge_types)
        
        # Add detailed annotations
        self._add_network_annotations(ax, graph, pos, node_types)
        
        # Create comprehensive legend
        self._add_comprehensive_legend(fig, ax, node_types, edge_types)
        
        ax.set_title(title, fontsize=18, color=self.current_theme.text_color, pad=30)
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def _draw_enhanced_network(self, 
                             ax, 
                             graph: nx.Graph, 
                             pos: Dict[str, Tuple[float, float]], 
                             node_types: Dict[str, str], 
                             edge_types: Dict[Tuple[str, str], str]):
        """Draw network with enhanced visual elements."""
        
        # Draw edges with varying styles
        for edge, edge_type in edge_types.items():
            edge_style = self.current_theme.edge_styles.get(
                edge_type, self.current_theme.edge_styles[EdgeType.CAUSAL.value]
            )
            
            x1, y1 = pos[edge[0]]
            x2, y2 = pos[edge[1]]
            
            # Create connection
            if isinstance(graph, nx.DiGraph):
                # Directed edge with arrow
                arrow = ConnectionPatch(
                    (x1, y1), (x2, y2), "data", "data",
                    arrowstyle="->", shrinkA=15, shrinkB=15,
                    mutation_scale=edge_style.arrow_size,
                    fc=edge_style.color, ec=edge_style.color,
                    linewidth=edge_style.width,
                    alpha=edge_style.alpha,
                    linestyle=edge_style.style
                )
                ax.add_patch(arrow)
            else:
                # Undirected edge
                ax.plot([x1, x2], [y1, y2], 
                       color=edge_style.color,
                       linewidth=edge_style.width,
                       alpha=edge_style.alpha,
                       linestyle=edge_style.style)
        
        # Draw nodes with custom shapes
        for node, node_type in node_types.items():
            node_style = self.current_theme.node_styles.get(
                node_type, self.current_theme.node_styles[NodeType.COVARIATE.value]
            )
            
            x, y = pos[node]
            size = node_style.size / 100  # Scale for patch size
            
            if node_style.shape == "circle":
                patch = plt.Circle((x, y), size, 
                                 color=node_style.color, 
                                 alpha=node_style.alpha,
                                 ec=node_style.border_color,
                                 linewidth=node_style.border_width)
            elif node_style.shape == "square":
                patch = plt.Rectangle((x - size, y - size), 2*size, 2*size,
                                    color=node_style.color,
                                    alpha=node_style.alpha,
                                    ec=node_style.border_color,
                                    linewidth=node_style.border_width)
            elif node_style.shape == "diamond":
                diamond_points = np.array([[x, y + size], [x + size, y], 
                                         [x, y - size], [x - size, y]])
                patch = plt.Polygon(diamond_points,
                                  color=node_style.color,
                                  alpha=node_style.alpha,
                                  ec=node_style.border_color,
                                  linewidth=node_style.border_width)
            elif node_style.shape == "triangle":
                triangle_points = np.array([[x, y + size], [x - size, y - size], [x + size, y - size]])
                patch = plt.Polygon(triangle_points,
                                  color=node_style.color,
                                  alpha=node_style.alpha,
                                  ec=node_style.border_color,
                                  linewidth=node_style.border_width)
            else:
                # Default to circle
                patch = plt.Circle((x, y), size, 
                                 color=node_style.color, 
                                 alpha=node_style.alpha)
            
            ax.add_patch(patch)
            
            # Add label
            ax.text(x, y, node, 
                   ha='center', va='center',
                   fontsize=node_style.label_size,
                   color=node_style.label_color,
                   fontweight='bold')
    
    def _add_network_annotations(self, 
                               ax, 
                               graph: nx.Graph, 
                               pos: Dict[str, Tuple[float, float]], 
                               node_types: Dict[str, str]):
        """Add detailed annotations to the network."""
        
        # Add degree information
        max_x = max(x for x, y in pos.values())
        min_x = min(x for x, y in pos.values())
        max_y = max(y for x, y in pos.values())
        
        # Network statistics
        stats_text = f"Nodes: {graph.number_of_nodes()}\nEdges: {graph.number_of_edges()}"
        if isinstance(graph, nx.DiGraph):
            stats_text += f"\nDAG: {'Yes' if nx.is_directed_acyclic_graph(graph) else 'No'}"
        
        ax.text(max_x + 0.1, max_y, stats_text,
               fontsize=10, ha='left', va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def _add_matplotlib_legend(self, ax, node_types_present: set, edge_types_present: set):
        """Add legend to matplotlib plot."""
        
        legend_elements = []
        
        # Node type legend
        for node_type in node_types_present:
            node_style = self.current_theme.node_styles.get(
                node_type, self.current_theme.node_styles[NodeType.COVARIATE.value]
            )
            
            marker_map = {
                "circle": "o",
                "square": "s", 
                "diamond": "D",
                "triangle": "^"
            }
            marker = marker_map.get(node_style.shape, "o")
            
            legend_elements.append(
                plt.Line2D([0], [0], marker=marker, color='w', 
                          markerfacecolor=node_style.color,
                          markersize=10, label=node_type.replace('_', ' ').title())
            )
        
        # Edge type legend
        for edge_type in edge_types_present:
            edge_style = self.current_theme.edge_styles.get(
                edge_type, self.current_theme.edge_styles[EdgeType.CAUSAL.value]
            )
            
            linestyle_map = {
                "solid": "-",
                "dashed": "--", 
                "dotted": ":",
                "dashdot": "-."
            }
            linestyle = linestyle_map.get(edge_style.style, "-")
            
            legend_elements.append(
                plt.Line2D([0], [0], color=edge_style.color,
                          linewidth=edge_style.width, linestyle=linestyle,
                          label=f"{edge_type.replace('_', ' ').title()} Edge")
            )
        
        ax.legend(handles=legend_elements, loc=self.current_theme.legend_position,
                 frameon=True, fancybox=True, shadow=True)
    
    def _add_comprehensive_legend(self, fig, ax, node_types: Dict[str, str], edge_types: Dict[str, str]):
        """Add comprehensive legend with detailed explanations."""
        
        # Create separate legend axis
        legend_ax = fig.add_axes([0.85, 0.1, 0.15, 0.8])
        legend_ax.axis('off')
        
        y_pos = 0.95
        legend_ax.text(0.5, y_pos, 'Legend', ha='center', va='top', 
                      fontsize=14, fontweight='bold', transform=legend_ax.transAxes)
        y_pos -= 0.08
        
        # Node types
        legend_ax.text(0.05, y_pos, 'Node Types:', ha='left', va='top',
                      fontsize=12, fontweight='bold', transform=legend_ax.transAxes)
        y_pos -= 0.05
        
        for node_type in set(node_types.values()):
            node_style = self.current_theme.node_styles.get(
                node_type, self.current_theme.node_styles[NodeType.COVARIATE.value]
            )
            
            # Draw small node
            legend_ax.scatter(0.1, y_pos, s=50, c=node_style.color, 
                            marker='o', transform=legend_ax.transAxes)
            
            # Add description
            legend_ax.text(0.15, y_pos, node_type.replace('_', ' ').title(),
                          ha='left', va='center', fontsize=10, 
                          transform=legend_ax.transAxes)
            y_pos -= 0.06
        
        y_pos -= 0.03
        
        # Edge types
        legend_ax.text(0.05, y_pos, 'Edge Types:', ha='left', va='top',
                      fontsize=12, fontweight='bold', transform=legend_ax.transAxes)
        y_pos -= 0.05
        
        for edge_type in set(edge_types.values()):
            edge_style = self.current_theme.edge_styles.get(
                edge_type, self.current_theme.edge_styles[EdgeType.CAUSAL.value]
            )
            
            # Draw small line
            legend_ax.plot([0.05, 0.12], [y_pos, y_pos], 
                         color=edge_style.color, linewidth=2,
                         linestyle=edge_style.style, transform=legend_ax.transAxes)
            
            # Add description
            legend_ax.text(0.15, y_pos, edge_type.replace('_', ' ').title(),
                          ha='left', va='center', fontsize=10,
                          transform=legend_ax.transAxes)
            y_pos -= 0.06
    
    def _save_visualization(self, 
                          fig: Any, 
                          save_path: str, 
                          output_format: VisualizationType):
        """Save visualization to file."""
        
        save_path = Path(save_path)
        
        if output_format == VisualizationType.INTERACTIVE_PLOTLY and PLOTLY_AVAILABLE:
            if save_path.suffix.lower() == '.html':
                fig.write_html(str(save_path))
            else:
                fig.write_image(str(save_path))
        else:
            # Matplotlib figure
            fig.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor=self.current_theme.background_color)
        
        self.logger.info(f"Visualization saved to {save_path}")
    
    def create_causal_diagram_from_data(self,
                                      data: pd.DataFrame,
                                      discovered_edges: List[Tuple[str, str]],
                                      treatment_vars: List[str],
                                      outcome_vars: List[str],
                                      confounder_vars: Optional[List[str]] = None,
                                      title: str = "Discovered Causal Structure") -> plt.Figure:
        """
        Create causal diagram from discovered causal relationships.
        
        Args:
            data: Original dataset
            discovered_edges: List of discovered causal relationships
            treatment_vars: Treatment/intervention variables
            outcome_vars: Outcome variables
            confounder_vars: Optional list of confounders
            title: Diagram title
            
        Returns:
            Matplotlib figure with causal diagram
        """
        
        # Create graph
        graph = nx.DiGraph()
        
        # Add nodes
        all_vars = set()
        for source, target in discovered_edges:
            all_vars.add(source)
            all_vars.add(target)
        
        graph.add_nodes_from(all_vars)
        graph.add_edges_from(discovered_edges)
        
        # Assign node types
        node_types = {}
        for var in all_vars:
            if var in treatment_vars:
                node_types[var] = NodeType.TREATMENT.value
            elif var in outcome_vars:
                node_types[var] = NodeType.OUTCOME.value
            elif confounder_vars and var in confounder_vars:
                node_types[var] = NodeType.CONFOUNDER.value
            else:
                node_types[var] = NodeType.COVARIATE.value
        
        # Create visualization
        return self.create_causal_graph_visualization(
            graph=graph,
            node_types=node_types,
            layout=GraphLayout.HIERARCHICAL,
            theme="default",
            title=title,
            output_format=VisualizationType.STATIC_MATPLOTLIB
        )
    
    def export_interactive_html(self, 
                              graph: nx.Graph, 
                              output_path: str,
                              **kwargs) -> str:
        """Export interactive HTML visualization."""
        
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive HTML export")
        
        fig = self.create_causal_graph_visualization(
            graph=graph,
            output_format=VisualizationType.INTERACTIVE_PLOTLY,
            **kwargs
        )
        
        fig.write_html(output_path)
        return output_path


# Convenience functions
def create_graph_visualizer() -> VisualCausalGraphGenerator:
    """Create a visual causal graph generator."""
    return VisualCausalGraphGenerator()


def quick_visualize_graph(graph: nx.Graph, 
                         title: str = "Causal Graph",
                         save_path: Optional[str] = None) -> plt.Figure:
    """Quick function to visualize a causal graph."""
    
    visualizer = create_graph_visualizer()
    return visualizer.create_causal_graph_visualization(
        graph=graph,
        title=title,
        save_path=save_path
    )


def create_causal_diagram(treatment_vars: List[str],
                         outcome_vars: List[str], 
                         confounders: List[str],
                         edges: List[Tuple[str, str]],
                         title: str = "Causal Diagram") -> plt.Figure:
    """Create a causal diagram from variable lists and edges."""
    
    # Build graph
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    
    # Assign types
    node_types = {}
    for node in graph.nodes():
        if node in treatment_vars:
            node_types[node] = NodeType.TREATMENT.value
        elif node in outcome_vars:
            node_types[node] = NodeType.OUTCOME.value
        elif node in confounders:
            node_types[node] = NodeType.CONFOUNDER.value
        else:
            node_types[node] = NodeType.COVARIATE.value
    
    # Visualize
    visualizer = create_graph_visualizer()
    return visualizer.create_causal_graph_visualization(
        graph=graph,
        node_types=node_types,
        layout=GraphLayout.HIERARCHICAL,
        title=title
    )