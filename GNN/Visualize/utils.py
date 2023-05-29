import json

import torch
import dgl
from dgl import load_graphs

class GraphLoader():
    '''
    The Scheme of the heterogenous graph:
        ("contributor", "contributor_belong_to_org", "organization")
        ("contributor", "contributor_commit_repo", "repository")
        ("contributor", "contributor_follow_contributor", "contributor")
        ("contributor", "contributor_propose_issue", "issue")
        ("contributor", "contributor_propose_pr", "pr")
        ("contributor", "contributor_star_repo", "repository")
        ("contributor", "contributor_watch_repo", "repository")
        ("issue", "issue_belong_to_repo", "repository")
        ("pr", "pr_belong_to_repo", "repository")
    '''
    def __init__(self, graph_path) -> None:
        self.graph_path = graph_path
        pass
    
    @staticmethod
    def print_hg_info(hg):
        print("################# Basic Information of The Graph #################")
        for etype in hg.canonical_etypes:
            print("Edge", etype, hg.number_of_edges(etype))
        for ntype in hg.ntypes:
            print("Node", ntype, hg.number_of_nodes(ntype))
        print("Total number of nodes", hg.number_of_nodes())
        print("Total number of edges", hg.number_of_edges())
        print("################# End of the Graph Information  #################")

    def load_graph(self, device=torch.device("cpu")):
        hgs, _ = load_graphs(self.graph_path)
        hg = hgs[0]
        GraphLoader.print_hg_info(hg=hg)

        edge2ids = {}
        for c_etype in hg.canonical_etypes:
            edge2ids[c_etype] = len(edge2ids)
        print(edge2ids)

        node_feats = {}
        for nt in hg.ntypes:
            node_feats[nt] = hg.nodes[nt].data.pop("feat").to(device)
        return hg, node_feats, edge2ids


class NodeID():
    def __init__(self, repo_idx_file, pr_idx_file, issue_idx_file, contributor_idx_file) -> None:
        self.repo_idx_file = repo_idx_file
        self.pr_idx_file = pr_idx_file
        self.issue_idx_file = issue_idx_file
        self.contributor_idx_file = contributor_idx_file

        with open(self.repo_idx_file, "r") as f:
            self.repo_idx = json.load(f)
            self.idx_repo = {v: k for k, v in self.repo_idx.items()}
        with open(self.pr_idx_file, "r") as f:
            self.pr_idx = json.load(f)
            self.idx_pr = {v: k for k, v in self.pr_idx.items()}
        with open(self.issue_idx_file, "r") as f:
            self.issue_idx = json.load(f)
            self.idx_issue = {v: k for k, v in self.issue_idx.items()}
        with open(self.contributor_idx_file, "r") as f:
            self.contributor_idx = json.load(f)
            self.idx_contributor = {v: k for k, v in self.contributor_idx.items()}
    
    def get_name_by_id(self, id, id_type):
        if id_type == "repo":
            print(self.idx_repo[id])
            return self.idx_repo[id]
        elif id_type == "pr":
            return self.idx_pr[id]
        elif id_type == "issue":
            return self.idx_issue[id]
        elif id_type == "contributor":
            return self.idx_contributor[id]
        else:
            raise ValueError("Wrong id type: {}".format(id_type))
    
    def get_id_by_name(self, name, id_type):
        if id_type == "repo":
            return self.repo_idx[name]
        elif id_type == "pr":
            return self.pr_idx[name]
        elif id_type == "issue":
            return self.issue_idx[name]
        elif id_type == "contributor":
            return self.contributor_idx[name]
        else:
            raise ValueError("Wrong id type: {}".format(id_type))