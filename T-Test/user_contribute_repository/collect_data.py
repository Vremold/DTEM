import os
import sys
import json
import pickle
import random

import numpy as np

import torch
from scipy import stats

class DatasetGenerator():
    def __init__(self, contributor_idx_file, repo_idx_file) -> None:
        with open(contributor_idx_file, "r", encoding="utf-8") as inf:
            self.contributor_idx = json.load(inf)

        with open(repo_idx_file, "r", encoding="utf-8") as inf:
            self.repo_idx = json.load(inf)
            self.repositories = set(self.repo_idx.keys())

    def generate_usr_contribute_repo(self, contributions_file):
        repo_contributors = {}
        positve_samples = []
        with open(contributions_file, "r", encoding="utf-8") as inf:
            for line in inf:
                repo_name, contris = line.strip().split("\t")
                contris = json.loads(contris)
                for c in contris:
                    positve_samples.append((repo_name, c[0]))
                    repo_contributors.setdefault(repo_name, set()).add(c[0])
        
        negative_samples = []
        for sample in positve_samples:
            curr_repo = sample[0]
            curr_user = sample[1]
            while True:
                repo = random.choice(list(self.repositories - set([curr_repo])))
                if repo not in repo_contributors or curr_user not in repo_contributors[repo]:
                    negative_samples.append((repo, curr_user))
                    break

        positve_samples = [(self.repo_idx[sample[0]], self.contributor_idx[sample[1]]) for sample in positve_samples]
        negative_samples = [(self.repo_idx[sample[0]], self.contributor_idx[sample[1]]) for sample in negative_samples]

        return positve_samples, negative_samples

class RepoUserPairSim():
    def __init__(self, node_embedding) -> None:
        self.repo_embedding = node_embedding["repository"].numpy()
        self.contributor_embedding = node_embedding["contributor"].numpy()
        self.repo_mean = np.mean(self.repo_embedding, axis=0)
        self.user_mean = np.mean(self.contributor_embedding, axis=0)
        pass

    def cosine_sim(self, repo_idx, user_idx):
        repo_embedding = self.repo_embedding[repo_idx]
        user_embedding = self.contributor_embedding[user_idx]
        dot = np.dot(repo_embedding, user_embedding)
        norm = np.linalg.norm(repo_embedding) * np.linalg.norm(user_embedding)
        return (dot / norm + 1) / 2

    def adjusted_cosine_sim(self, repo_idx, user_idx):
        repo_embedding = self.repo_embedding[repo_idx] - self.repo_mean
        user_embedding = self.contributor_embedding[user_idx] - self.user_mean
        dot = np.dot(repo_embedding, user_embedding)
        norm = np.linalg.norm(repo_embedding) * np.linalg.norm(user_embedding)
        return (dot / norm + 1) / 2
    
    def euclidean_sim(self, repo_idx, user_idx):
        repo_embedding = self.repo_embedding[repo_idx]
        user_embedding = self.contributor_embedding[user_idx]
        return 1 / (1 + np.linalg.norm(repo_embedding - user_embedding))

    def manhattan_sim(self, repo_idx, user_idx):
        repo_embedding = self.repo_embedding[repo_idx]
        user_embedding = self.contributor_embedding[user_idx]
        return 1 / (1 + np.sum(np.abs(repo_embedding - user_embedding)))
    
    def pearson_sim(self, repo_idx, user_idx):
        repo_embedding = self.repo_embedding[repo_idx]
        user_embedding = self.contributor_embedding[user_idx]
        return (np.corrcoef(repo_embedding, user_embedding)[0, 1] + 1) / 2
    

    def get_sim(self, positive_samples, negative_samples, sim_func_str="cosine"):
        if sim_func_str == "cosine":
            sim_func = self.cosine_sim
        elif sim_func_str == "euclidean":
            sim_func = self.euclidean_sim
        elif sim_func_str == "manhattan":
            sim_func = self.manhattan_sim
        elif sim_func_str == "pearson":
            sim_func = self.pearson_sim
        elif sim_func_str == "adjusted_cosine":
            sim_func = self.adjusted_cosine_sim
        else:
            raise ValueError("sim_func_str should be one of ['cosine', 'euclidean', 'manhattan', 'pearson']")
        positive_sim = [sim_func(sample[0], sample[1]) for sample in positive_samples]
        negative_sim = [sim_func(sample[0], sample[1]) for sample in negative_samples]
        return positive_sim, negative_sim

if __name__ == "__main__":
    # get positive and negative samples
    positive_cache_path = "./data/postive_cache"
    negative_cache_path = "./data/negative_cache"
    positive_samples = []
    negative_samples = []
    print("Loading positive and negative samples...")
    if os.path.exists(positive_cache_path) and os.path.exists(negative_cache_path):
        print("Loading from cache...")
        with open(positive_cache_path, "rb") as inf:
            positive_samples = pickle.load(inf)
        with open(negative_cache_path, "rb") as inf:
            negative_samples = pickle.load(inf)
    else:
        print("Generating from scratch...")
        repository_idx_file = "../../GNN/DataPreprocess/full_graph/content/repositories.json"
        contributor_idx_file = "../../GNN/DataPreprocess/full_graph/content/contributors.json"
        dataset_generator = DatasetGenerator(
            contributor_idx_file=contributor_idx_file,
            repo_idx_file=repository_idx_file,
        )
        positive_samples, negative_samples = dataset_generator.generate_usr_contribute_repo("../../GHCrawler/cleaned/repo_contributions.txt")
        with open(positive_cache_path, "wb") as ouf:
            pickle.dump(positive_samples, ouf)
        with open(negative_cache_path, "wb") as ouf:
            pickle.dump(negative_samples, ouf)
        
    # get node embedding
    node_embedding_path = "../../GNN/HetSAGE/node_embedding/HetSAGE_node_embedding.bin"
    node_embedding = torch.load(node_embedding_path, map_location=torch.device("cpu"))

    # get similarity
    sim_func_str = "cosine"
    sim_func_str = "euclidean"
    sim_func_str = "manhattan"
    sim_func_str = "pearson"
    sim_func_str = "adjusted_cosine"
    n_samples = 5000

    cache_sim_pos_path = f"./data/{sim_func_str}_positive_sim.pkl"
    cache_sim_neg_path = f"./data/{sim_func_str}_negative_sim.pkl"
    if os.path.exists(cache_sim_pos_path) and os.path.exists(cache_sim_neg_path):
        print("Loading similarity from cache...")
        with open(cache_sim_pos_path, "rb") as inf:
            positive_sim = pickle.load(inf)
        with open(cache_sim_neg_path, "rb") as inf:
            negative_sim = pickle.load(inf)
    else:
        print("Calculating similarity from scratch...")
        repo_user_pair_sim = RepoUserPairSim(node_embedding)
        positive_samples = random.sample(positive_samples, n_samples)
        negative_samples = random.sample(negative_samples, n_samples)
        positive_sim, negative_sim = repo_user_pair_sim.get_sim(positive_samples, negative_samples, sim_func_str=sim_func_str)        

        with open(cache_sim_pos_path, "wb") as ouf:
            pickle.dump(positive_sim, ouf)
        with open(cache_sim_neg_path, "wb") as ouf:
            pickle.dump(negative_sim, ouf)
    
    # T-test
    print("T-test for positive samples and negative samples")
    # print(stats.levene(positive_sim, negative_sim))
    print(np.mean(positive_sim), np.mean(negative_sim), sep="\n")
    print(stats.ttest_ind(positive_sim, negative_sim, equal_var=False))