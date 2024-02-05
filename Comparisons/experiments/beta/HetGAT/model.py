import torch
import torch.nn as nn


class NodeClassificationScorer(nn.Module):
    def __init__(self, in_features, n_classes) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=n_classes)
        self.softmax = nn.Softmax(dim=-1)

    def apply_nodes(self, nodes):
        score = self.softmax(self.linear(nodes["x"]))
        return {"score": score}
    
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata["x"] = x
            for ntype in edge_subgraph.ntypes:
                edge_subgraph.apply_nodes(self.apply_nodes, ntype=ntype)
            return edge_subgraph.ndata["score"]

class EdgeRegresionScorer(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, 1)
    
    def apply_edges(self, edges):
        score = self.linear(edges.src["x"] * edges.dst["x"])
        return {"score": score}

    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata["x"] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(self.apply_edges, etype=etype)
            return edge_subgraph.edata["score"]

class EdgeClassificationScorer(nn.Module):
    def __init__(self, num_classes, in_features, dropout=0.2) -> None:
        super().__init__()
        self.linear = nn.Linear(2*in_features, num_classes)
        self.softmax = nn.Softmax(dim=-1)
    
    def apply_edges(self, edges):
        score = self.linear(torch.cat([edges.src["x"], edges.dst["x"]], dim=-1))
        return {"score": self.softmax(score)}

    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata["x"] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(self.apply_edges, etype=etype)
            return edge_subgraph.edata["score"]

class LinkPredictionScorer(nn.Module):
    """
    Use cat + linear + sigmoid as the score function
    """
    def __init__(self, in_features) -> None:
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(2 * in_features, 1)

    def apply_edges(self, edges):
        score = self.linear(torch.cat([edges.src["x"], edges.dst["x"]], dim=-1))
        return {"score": self.sigmoid(score)}

    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata["x"] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(self.apply_edges, etype=etype)
            return edge_subgraph.edata["score"]
        
class LinkPredictionScorer_V2(nn.Module):
    """
    Use dot product + sigmoid as the score function
    """
    def __init__(self, in_features) -> None:
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def apply_edges(self, edges):
        score = self.linear(edges.src["x"] * edges.dst["x"])
        return {"score": self.sigmoid(score)}

    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata["x"] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(self.apply_edges, etype=etype)
            return edge_subgraph.edata["score"]
  