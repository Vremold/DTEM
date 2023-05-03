import os
import sys
import json
import random

class UserRepoDataCollector():
    def __init__(self, repo_idx_file, contributor_idx_file, watch_file, contribution_file, threshold=30) -> None:
        with open(repo_idx_file, "r", encoding="utf-8") as inf:
            self.repo_idx = json.load(inf)
            self.repositories = set(self.repo_idx.keys())
        with open(contributor_idx_file, "r", encoding="utf-8") as inf:
            self.contributor_idx = json.load(inf)
            self.contributors = set(self.contributor_idx.keys())
        
        self.contributor_watch_repos = {}
        with open(watch_file, "r", encoding="utf-8") as inf:
            for line in inf:
                repo_name, watchers = line.strip().split("\t")
                watchers = json.loads(watchers)
                for w in watchers:
                    self.contributor_watch_repos.setdefault(w, set()).add(repo_name)
        
        self.contributor_commit_repos = {}
        with open(contribution_file, "r", encoding="utf-8") as inf:
            for line in inf:
                repo_name, contributors = line.strip().split("\t")
                contributors = json.loads(contributors)
                for c in contributors:
                    if c[1] > threshold:
                        self.contributor_commit_repos.setdefault(c[0], set()).add(repo_name)

    def collect_data(self, negative_samples=1):
        samples = []
        for c in self.contributors:
            c_idx = self.contributor_idx[c]
            watch_repos = self.contributor_watch_repos.get(c, set())
            commit_repos = self.contributor_commit_repos.get(c, set())
            watch_and_commit_repos = watch_repos.intersection(commit_repos)
            scope = list(self.repositories - watch_repos - commit_repos)
            for r in watch_and_commit_repos:
                pos_idx = self.repo_idx[r]
                neg_repos = random.sample(scope, negative_samples)
                neg_idxs = [self.repo_idx[x] for x in neg_repos]
                samples.append([c_idx, pos_idx] + neg_idxs)
        return samples


if __name__ == "__main__":
    # Please replace the following paths with your own paths
    repo_idx_file = "../../GNN/DataPreprocess/full_graph/content/repositories.json"
    contributor_idx_file = "../../GNN/DataPreprocess/full_graph/content/contributors.json"
    watch_path = "../../GHCrawler/repo_watchers.txt"
    contribution_file = "../../GHCrawler/cleaned/repo_contributions.txt"
    dst_file="./data/user_watch_repos.json"

    collector = UserRepoDataCollector(
        repo_idx_file=repo_idx_file, 
        contributor_idx_file=contributor_idx_file,
        watch_file=watch_path,
        contribution_file=contribution_file,
    )
    samples = collector.collect_data(negative_samples=1)
    with open(dst_file, "w", encoding="utf-8") as outf:
        json.dump(samples, outf, indent=4, ensure_ascii=False)
    

