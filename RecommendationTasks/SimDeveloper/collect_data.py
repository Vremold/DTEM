import os
import sys
import json
import random

class SimUserDataCollector():
    def __init__(self, contributor_idx_file) -> None:
        with open(contributor_idx_file, "r", encoding="utf-8") as inf:
            self.contributor_idx = json.load(inf)
            self.contributors = set(self.contributor_idx.keys())
        pass

    def collect_data(self, user_organization_data_file, neg_sampling=1):
        org_users = {}
        with open(user_organization_data_file, "r", encoding="utf-8") as inf:
            for line in inf:
                user_name, orgs = line.strip().split("\t")
                orgs = json.loads(orgs)
                for org in orgs:
                    if org not in org_users:
                        org_users[org] = []
                    org_users[org].append(user_name)
        samples = []
        for org in org_users:
            if len(org_users[org]) <= 1:
                continue
            negative_range = self.contributors - set(org_users[org])
            negative_contributors = random.sample(negative_range, neg_sampling * (len(org_users[org]) - 1))
            for i in range(len(org_users[org]) - 1):
                src_idx = self.contributor_idx[org_users[org][i]]
                pos_idx = self.contributor_idx[org_users[org][i+1]]
                neg_contributor = negative_contributors[i]
                neg_idx = self.contributor_idx[neg_contributor]
                samples.append([src_idx, pos_idx, neg_idx])
        return samples, {k: [self.contributor_idx[x] for x in v] for k, v in org_users.items()}

if __name__ == "__main__":
    contributor_idx_file = "../../GNN/DataPreprocess/full_graph/content/contributors.json"
    user_organization_data_file = "../../GHCrawler/cleaned/user_organizations.txt"
    dst_file = "./data/sim_user.json"
    dst_org_user_file = "./data/org_user.json"

    pr_data_collector = SimUserDataCollector(
        contributor_idx_file=contributor_idx_file
    )
    smaples, org_users = pr_data_collector.collect_data(
        user_organization_data_file=user_organization_data_file,
        neg_sampling=1,
    )

    with open(dst_org_user_file, "w", encoding="utf-8") as outf:
        json.dump(org_users, outf, indent=4)

    with open(dst_file, "w", encoding="utf-8") as outf:
        json.dump(smaples, outf, indent=4)
