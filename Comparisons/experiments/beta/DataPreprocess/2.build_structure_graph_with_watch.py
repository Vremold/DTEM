import os
import math
import json
import pickle

import dgl
import torch
from dgl.data.utils import save_graphs


""" 
    这个类也只在此文件中使用这一次. 
    刚才第一步中 1.load_crawled_data 做了数据处理, 
    接下来构建图. 

    (还没有针对节点的 embedding 做训练)

        GNN/DataPreprocess/full_graph/content => Comparisons/data/beta/structure_graph_without_watch.bin
"""

class GraphBuilder(object):

    def __init__(self, src_graph_dir) -> None:
        self.src_graph_dir = src_graph_dir

    def load_watch_and_star_rels(self, hg_srcs:dict, reverse=False, contain_reverse=False):
        watch_rels = set()
        with open(os.path.join(self.src_graph_dir, "contributor_watch_repo.txt"), "r", encoding="utf-8") as inf:
            for line in inf:
                s, d, _ = line.strip().split('\t')
                watch_rels.add((s, d))
                
        star_rels = set()
        with open(os.path.join(self.src_graph_dir, "contributor_star_repo.txt"), "r", encoding="utf-8") as inf:
            for line in inf:
                s, d, _ = line.strip().split('\t')
                star_rels.add((s, d))
    
        # contributor - star - repository
        src = []
        dst = []
        for item in star_rels:
            src.append(int(item[0]))
            dst.append(int(item[1]))
        hg_srcs[("contributor", "contributor_star_repo", "repository")] = (torch.tensor(src), torch.tensor(dst))
        
        # contributor - watch - repository
        src = []
        dst = []
        for item in watch_rels:
            src.append(int(item[0]))
            dst.append(int(item[1]))
        hg_srcs[("contributor", "contributor_watch_repo", "repository")] = (torch.tensor(src), torch.tensor(dst))
    
    def load_contributor_belong_to_org_rels(self, hg_srcs:dict, reverse=False, contain_reverse=False):
        # contributor - organization
        src = []
        dst = []
        with open(os.path.join(self.src_graph_dir, "contributor_belong_to_org.txt"), "r", encoding="utf-8") as inf:
            for line in inf:
                s, d, _ = line.strip().split("\t")
                src.append(int(s))
                dst.append(int(d))
        hg_srcs[("contributor", "contributor_belong_to_org", "organization")] = (torch.tensor(src), torch.tensor(dst))

    def load_contributor_commit_repo_rels(self, hg_srcs:dict, reverse=True, contain_reverse=False):
        # contributor - repo
        src = []
        dst = []
        weights = []
        contributor_weights = {}
        with open(os.path.join(self.src_graph_dir, "contributor_commit_repo.txt"), "r", encoding="utf-8") as inf:
            for line in inf:
                s, d, w = line.strip().split("\t")
                weights.append(math.log(int(w)))
                src.append(int(s))
                dst.append(int(d))
                contributor_weights[int(s)] = contributor_weights.get(int(s), 0) + int(w)

        # if reverse:
        hg_srcs[("repository", "repo_committed_by_contributor", "contributor")] = (torch.tensor(dst), torch.tensor(src))
        return weights

    def load_contributor_follow_contributor_rels(self, hg_srcs:dict, reverse=False, contain_reverse=False):
        # contributor - contributor
        src = []
        dst = []
        with open(os.path.join(self.src_graph_dir, "contributor_follow_contributor.txt"), "r", encoding="utf-8") as inf:
            for line in inf:
                s, d, _ = line.strip().split("\t")
                src.append(int(s))
                dst.append(int(d))
        hg_srcs[("contributor", "contributor_follow_contributor", "contributor")] = (torch.tensor(src), torch.tensor(dst))

    def load_contributor_propose_issue_rels(self, hg_srcs:dict, reverse=False, contain_reverse=False):
        src = []
        dst = []
        with open(os.path.join(self.src_graph_dir, "contributor_propose_issue.txt"), "r", encoding="utf-8") as inf:
            for line in inf:
                s, d, _ = line.strip().split("\t")
                src.append(int(s))
                dst.append(int(d))
        hg_srcs[("contributor", "contributor_propose_issue", "issue")] = (torch.tensor(src), torch.tensor(dst))
    
    def load_contributor_propose_pr_rels(self, hg_srcs:dict, reverse=False, contain_reverse=False):
        # contributor - pr
        src = []
        dst = []
        with open(os.path.join(self.src_graph_dir, "contributor_propose_pr.txt"), "r", encoding="utf-8") as inf:
            for line in inf:
                s, d, _ = line.strip().split("\t")
                src.append(int(s))
                dst.append(int(d))
        hg_srcs[("contributor", "contributor_propose_pr", "pr")] = (torch.tensor(src), torch.tensor(dst))
    
    def load_issue_belong_to_repo_rels(self, hg_srcs:dict, reverse=False, contain_reverse=False):
        # issue - repo
        src = []
        dst = []
        with open(os.path.join(self.src_graph_dir, "issue_belong_to_repo.txt"), "r", encoding="utf-8") as inf:
            for line in inf:
                s, d, _ = line.strip().split("\t")
                src.append(int(s))
                dst.append(int(d))
        hg_srcs[("issue", "issue_belong_to_repo", "repository")] = (torch.tensor(src), torch.tensor(dst))

    def load_pr_belong_to_repo_rels(self, hg_srcs:dict, reverse=False, contain_reverse=False):
        # pr - repo
        src = []
        dst = []
        weights = []
        with open(os.path.join(self.src_graph_dir, "pr_belong_to_repo.txt"), "r", encoding="utf-8") as inf:
            for line in inf:
                s, d, p_state = line.strip().split("\t")
                pr_state = int(p_state)
                if pr_state >= 2:
                    weights.append(1)
                else:
                    weights.append(0)
                src.append(int(s))
                dst.append(int(d))
        hg_srcs[("pr", "pr_belong_to_repo", "repository")] = (torch.tensor(src), torch.tensor(dst))
        return weights

    def build_graph(self, dst_graph_path):
        """ 
            Set hg_srcs, which is used to build `dgl` graph. 

            1. invoke methods to set 8 normal kinds of edges; 
            2. specify weights for 2 special kinds of edges;
        """
        hg_srcs = dict()

        self.load_watch_and_star_rels(hg_srcs)
        repo_commmited_by_contriutor_weight = self.load_contributor_commit_repo_rels(hg_srcs)
        self.load_contributor_follow_contributor_rels(hg_srcs)
        self.load_contributor_propose_issue_rels(hg_srcs)
        self.load_contributor_propose_pr_rels(hg_srcs)
        self.load_issue_belong_to_repo_rels(hg_srcs)
        pr_state = self.load_pr_belong_to_repo_rels(hg_srcs)

        hg = dgl.heterograph(hg_srcs)  # TODO THIS MAY BE IMPORTANT. TRY TO MODIFY THIS. 

        edge_label = 0
        for et in hg.etypes:
            hg.edges[et].data["reltype"] = torch.LongTensor([edge_label] * hg.number_of_edges(et))
            edge_label += 1

        # Set weight of special edge
        hg.edges[("repository", "repo_committed_by_contributor", "contributor")].data["cr_weight"] = torch.FloatTensor(repo_commmited_by_contriutor_weight)
        hg.edges[("pr", "pr_belong_to_repo", "repository")].data["pr_label"] = torch.LongTensor(pr_state)
        
        save_graphs(dst_graph_path, [hg])

if __name__ == "__main__":
    gb = GraphBuilder(src_graph_dir="GNN/DataPreprocess/full_graph/content")
    gb.build_graph(dst_graph_path="Comparisons/data/beta/structure_graph_with_watch.bin")