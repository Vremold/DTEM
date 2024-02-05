import os
import json
import sys

import numpy as np

class Graph():
    def __init__(self, commit_rels_f, follow_rels_f, star_rels_f, watch_rels_f) -> None:
        self.commit_rels = {}
        self.following_rels = {}
        self.follower_rels = {}
        self.star_rels = {}
        self.watch_rels = {}
        with open(commit_rels_f, "r", encoding="utf-8") as inf:
            for line in inf:
                s, d, w = line.strip().split("\t")
                s, d = int(s), int(d)
                if s not in self.commit_rels:
                    self.commit_rels[s] = []
                self.commit_rels[s].append(d)
        
        with open(follow_rels_f, "r", encoding="utf-8") as inf:
            for line in inf:
                s, d, _ = line.strip().split("\t")
                s, d = int(s), int(d)
                if d not in self.follower_rels:
                    self.follower_rels[d] = []
                self.follower_rels[d].append(s)
                if s not in self.following_rels:
                    self.following_rels[s] = []
                self.following_rels[s].append(d)

        with open(star_rels_f, "r", encoding="utf-8") as inf:
            for line in inf:
                s, d, _ = line.strip().split("\t")
                s, d = int(s), int(d)
                if s not in self.star_rels:
                    self.star_rels[s] = []
                self.star_rels[s].append(d)

        with open(watch_rels_f, "r", encoding="utf-8") as inf:
            for line in inf:
                s, d, _ = line.strip().split("\t")
                s, d = int(s), int(d)
                if s not in self.watch_rels:
                    self.watch_rels[s] = []
                self.watch_rels[s].append(d)
    
    def get_repo_by_contributor(self, contributor_idx):
        repo_idxs = set()
        repo_idxs.update(self.commit_rels.get(contributor_idx, []))
        repo_idxs.update(self.star_rels.get(contributor_idx, []))
        repo_idxs.update(self.watch_rels.get(contributor_idx, []))
        for follower in self.follower_rels.get(contributor_idx, []):
            repo_idxs.update(self.commit_rels.get(follower, []))
        for following in self.following_rels.get(contributor_idx, []):
            repo_idxs.update(self.commit_rels.get(following, []))
        return list(repo_idxs)

if __name__ == "__main__":
    sample_path = ("../data/test.json", "../data/valid.json")
    dst_path = "./data/dataset_valid_test.json"
    with open(sample_path[0], "r", encoding="utf-8") as inf:
        samples = json.load(inf)
    with open(sample_path[1], "r", encoding="utf-8") as inf:
        samples.extend(json.load(inf))
    
    g = Graph(
        commit_rels_f="../../../GNN/DataPreprocess/full_graph/content/contributor_commit_repo.txt",
        follow_rels_f="../../../GNN/DataPreprocess/full_graph/content/contributor_follow_contributor.txt",
        star_rels_f="../../../GNN/DataPreprocess/full_graph/content/contributor_star_repo.txt",
        watch_rels_f="../../../GNN/DataPreprocess/full_graph/content/contributor_watch_repo.txt",
    )

    # {contributor: [positive_repos]}
    contributor_labels = {}
    for sample in samples:
        contributor_idx, pos_repo_idx, neg_repo_idx = sample
        if contributor_idx not in contributor_labels:
            contributor_labels[contributor_idx] = []
        contributor_labels[contributor_idx].append(pos_repo_idx)

    samples = []
    for contributor_idx in contributor_labels:
        search_scope = g.get_repo_by_contributor(contributor_idx)
        samples.append([contributor_idx, search_scope, contributor_labels[contributor_idx]])
    
    with open(dst_path, "w", encoding="utf-8") as ouf:
        json.dump(samples, ouf, indent=4, ensure_ascii=False)