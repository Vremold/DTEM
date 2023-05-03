import os
import sys
import json
import random

class PrDataCollecter():
    def __init__(self, repo_idx_file, pr_idx_file, contributor_idx_file) -> None:
        with open(repo_idx_file, "r", encoding="utf-8") as inf:
            self.repo_idx = json.load(inf)
        with open(pr_idx_file, "r", encoding="utf-8") as inf:
            self.pr_idx = json.load(inf)
        with open(contributor_idx_file, "r", encoding="utf-8") as inf:
            self.contributor_idx = json.load(inf)
            self.contributors = set(self.contributor_idx.keys())
        pass

    def collect_data(self, pr_data_file, negative_samples=1):
        pr_reviewers = {}
        samples = []
        with open(pr_data_file, "r", encoding="utf-8") as inf:
            for line in inf:
                repo_name, prs = line.strip().split("\t")
                prs = json.loads(prs)
                for pr in prs:
                    if not pr["if_merged"] or not pr["reviewers"]:
                        continue

                    pos_revewers = pr["reviewers"]
                    repo_idx = self.repo_idx[repo_name]
                    pr_idx = self.pr_idx["{}##{}".format(repo_name, pr["number"])]
                    pos_reviewer_idx = None
                    for r in pos_revewers:
                        if r in self.contributor_idx:
                            pos_reviewer_idx = self.contributor_idx[r]
                            break
                    if pos_reviewer_idx is None:
                        continue
                    pos_revewer_idxs = [self.contributor_idx[r] for r in pos_revewers if r in self.contributor_idx]
                    pr_reviewers[f"{repo_idx}#{pr_idx}"] = pos_revewer_idxs
                    neg_reviewer = random.sample(self.contributors - set(pos_revewers), negative_samples)
                    neg_reviewer_idx = [self.contributor_idx[r] for r in neg_reviewer]
                    samples.append([repo_idx, pr_idx, pos_reviewer_idx] + neg_reviewer_idx)
        return samples, pr_reviewers


if __name__ == "__main__":
    repo_idx_file = "../../GNN/DataPreprocess/full_graph/content/repositories.json"
    pr_idx_file = "../../GNN/DataPreprocess/full_graph/content/prs.json"
    contributor_idx_file = "../../GNN/DataPreprocess/full_graph/content/contributors.json"
    pr_data_file = "../../GHCrawler/cleaned/repo_prs.txt"
    dst_file = "./data/pr_reviewer.json"
    pr_reviewer_file = "./data/pr_reviewers.json"

    pr_data_collector = PrDataCollecter(
        repo_idx_file=repo_idx_file, 
        pr_idx_file=pr_idx_file, 
        contributor_idx_file=contributor_idx_file
    )
    samples, pr_reviewers = pr_data_collector.collect_data(
        pr_data_file=pr_data_file, negative_samples=1,
    )

    with open(pr_reviewer_file, "w", encoding="utf-8") as outf:
        json.dump(pr_reviewers, outf, indent=4)
    with open(dst_file, "w", encoding="utf-8") as outf:
        json.dump(samples, outf, indent=4)

