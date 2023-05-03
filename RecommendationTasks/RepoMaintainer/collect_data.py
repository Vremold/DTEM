import os
import sys
import json
import random

class RepoOrganizerDataCollector():
    def __init__(self, repo_idx_file, contributor_idx_file) -> None:
        with open(repo_idx_file, "r", encoding="utf-8") as inf:
            self.repo_idx = json.load(inf)
        with open(contributor_idx_file, "r", encoding="utf-8") as inf:
            self.contributor_idx = json.load(inf)
            self.contributors = set(self.contributor_idx.keys())
        pass

    def collect_data(self, repo_data_file, negative_samples=1):
        samples = []
        with open(repo_data_file, "r", encoding="utf-8") as inf:
            for line in inf:
                repo_info = json.loads(line)
                repo_name = repo_info["full_name"].lower()
                repo_owner = repo_info["owner"]["login"]
                if repo_owner not in self.contributor_idx:
                    continue
                repo_idx = self.repo_idx[repo_name]
                pos_idx = self.contributor_idx[repo_owner]
                scope = list(self.contributors - set([repo_owner]))
                neg_contributors = random.sample(scope, negative_samples)
                neg_idxs = [self.contributor_idx[x] for x in neg_contributors]
                samples.append([repo_idx, pos_idx] + neg_idxs)
        return samples


if __name__ == "__main__":
    repo_idx_file = "../../GNN/DataPreprocess/full_graph/content/repositories.json"
    contributor_idx_file = "../../GNN/DataPreprocess/full_graph/content/contributors.json"
    repo_data_file = "../../GhCrawler/cleaned/repo_statistics.txt"
    dst_file="./data/repo_organizer_samples.json"

    pr_data_collector = RepoOrganizerDataCollector(
        repo_idx_file=repo_idx_file, 
        contributor_idx_file=contributor_idx_file
    )
    samples = pr_data_collector.collect_data(
        repo_data_file=repo_data_file,
        negative_samples=1,
    )

    with open(dst_file, "w", encoding="utf-8") as outf:
        json.dump(samples, outf, indent=4, ensure_ascii=False)

