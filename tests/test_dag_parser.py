from causalllm.dag_parser import DAGParser

def test_parse_edges():
    edges = [("X", "Y"), ("Y", "Z")]
    parser = DAGParser(edges)
    assert parser.graph.has_edge("X", "Y")
    assert parser.graph.has_edge("Y", "Z")
    assert not parser.graph.has_edge("Z", "X")
