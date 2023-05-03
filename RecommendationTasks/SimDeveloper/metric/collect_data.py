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
    
    def get_repo_by_contributor(self, contributor_idx):
        repo_idxs = set()
        repo_idxs.update(self.commit_rels.get(contributor_idx, []))
        repo_idxs.update(self.star_rels.get(contributor_idx, []))
        repo_idxs.update(self.watch_rels.get(contributor_idx, []))
        for follower in self.follower_rels.get(contributor_idx, []):
            repo_idxs.update(self.commit_rels.get(follower, []))
            repo_idxs.update(self.star_rels.get(follower, []))
            repo_idxs.update(self.watch_rels.get(follower, []))
        for following in self.following_rels.get(contributor_idx, []):
            repo_idxs.update(self.commit_rels.get(following, []))
            repo_idxs.update(self.star_rels.get(following, []))
            repo_idxs.update(self.watch_rels.get(following, []))
        return list(repo_idxs)
    
    def get_contributor_by_repo(self, repo_idx):
        return self.reverse_commit_rels.get(repo_idx, [])

    def get_contributor_by_contributor(self, contributor_idx):
        contributor_idxs = set()
        # 贡献同一仓库的开发者们
        for repo in self.commit_rels.get(contributor_idx, []):
            contributor_idxs.update(self.reverse_commit_rels.get(repo, []))
        return list(contributor_idxs)


if __name__ == "__main__":
    sample_path = ("../data/test.json", "../data/valid.json")
    org_user_path = "../data/org_user.json"

    dst_path = "./data/dataset_valid_test.json"

    with open(sample_path[0], "r", encoding="utf-8") as inf:
        samples = json.load(inf)
    with open(sample_path[1], "r", encoding="utf-8") as inf:
        samples.extend(json.load(inf))
    
    with open(org_user_path, "r", encoding="utf-8") as inf:
        org_users = json.load(inf)
    user_orgs = {}
    for org in org_users:
        for user in org_users[org]:
            if user not in user_orgs:
                user_orgs[user] = []
            user_orgs[user].append(org)

    g = Graph(
        commit_rels_f="../../../GNN/DataPreprocess/full_graph/content/contributor_commit_repo.txt",
        follow_rels_f="../../../GNN/DataPreprocess/full_graph/content/contributor_follow_contributor.txt",
        star_rels_f="../../../GNN/DataPreprocess/full_graph/content/contributor_star_repo.txt",
        watch_rels_f="../../../GNN/DataPreprocess/full_graph/content/contributor_watch_repo.txt",
    )

    contributor_labels = {}
    for sample in samples:
        src_idx, pos_contributor_idx, neg_contributor_idx = sample
        if src_idx not in contributor_labels:
            contributor_labels[src_idx] = []
        for o in user_orgs[src_idx]:
            contributor_labels[src_idx].extend(org_users[o])

    samples = []
    for src_idx in contributor_labels:
        search_scope = g.get_contributor_by_contributor(src_idx)
        samples.append([src_idx, search_scope, contributor_labels[src_idx]])
    
    with open(dst_path, "w", encoding="utf-8") as ouf:
        json.dump(samples, ouf, indent=4, ensure_ascii=False)