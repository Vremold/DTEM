import os
import json
import sys

import numpy as np

class Graph():
    def __init__(self, commit_rels_f, follow_rels_f, star_rels_f, watch_rels_f) -> None:
        self.commit_rels = {}
        self.reverse_commit_rels = {}
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
                if d not in self.reverse_commit_rels:
                    self.reverse_commit_rels[d] = []
                self.commit_rels[s].append(d)
                self.reverse_commit_rels[d].append(s)

    def get_contributor_by_repo(self, repo_idx):
        return self.reverse_commit_rels.get(repo_idx, [])


if __name__ == "__main__":
    sample_path = ("../data/test.json", "../data/valid.json")
    pr_reviewers_path = "../data/pr_reviewers.json"
    dst_path = "./data/dataset_valid_test.json"
    with open(sample_path[0], "r", encoding="utf-8") as inf:
        samples = json.load(inf)
    with open(sample_path[1], "r", encoding="utf-8") as inf:
        samples.extend(json.load(inf))
    with open(pr_reviewers_path, "r", encoding="utf-8") as inf:
        pr_reviewers = json.load(inf)
    

    g = Graph(
        commit_rels_f="../../../GNN/DataPreprocess/full_graph/content/contributor_commit_repo.txt",
        follow_rels_f="../../../GNN/DataPreprocess/full_graph/content/contributor_follow_contributor.txt",
        star_rels_f="../../../GNN/DataPreprocess/full_graph/content/contributor_star_repo.txt",
        watch_rels_f="../../../GNN/DataPreprocess/full_graph/content/contributor_watch_repo.txt",
    )

    pr_labels = {}
    for sample in samples:
        repo_idx, pr_idx, _, _ = sample
        if (repo_idx, pr_idx) not in pr_labels:
            key = f"{repo_idx}#{pr_idx}"
            pr_labels[(repo_idx, pr_idx)] = pr_reviewers[key]

    samples = []
    for repo_idx, pr_idx in pr_labels:
        search_scope = g.get_contributor_by_repo(repo_idx)
        samples.append([repo_idx, pr_idx, search_scope, pr_labels[(repo_idx, pr_idx)]])
    
    with open(dst_path, "w", encoding="utf-8") as ouf:
        json.dump(samples, ouf, indent=4, ensure_ascii=False)