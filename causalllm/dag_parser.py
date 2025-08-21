from typing import Dict, List, Optional
import networkx as nx
import matplotlib.pyplot as plt

class DAGParser:
    def __init__(self, edges: List[tuple]) -> None:
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(edges)

    def get_ancestors(self, node: str) -> List[str]:
        return list(nx.ancestors(self.graph, node))

    def get_descendants(self, node: str) -> List[str]:
        return list(nx.descendants(self.graph, node))

    def to_prompt(self, task: str) -> str:
        return f"Task: {task}\nGraph structure: {list(self.graph.edges())}"

    def visualize(self, path: str = "dag.png") -> None:
        pos = nx.spring_layout(self.graph)
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_color="skyblue",
            node_size=2000,
            font_size=10,
            font_weight="bold",
            arrows=True,
        )
        plt.savefig(path)
        plt.close()
