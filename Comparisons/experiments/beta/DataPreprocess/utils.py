import json
import os
import pickle

import numpy as np
import torch

# Please replace this path with your own `absolute` path
# CACHE_DIR = "/root/wujw/DTEM/GNN/DataPreprocess/cache"
CACHE_DIR = "/media/dell/disk/vkx/DTEM/Comparisons/embedding/beta"
if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)

CAT = False

# Output:
# ./graph/content/issues.json
# There are 284 prs without text description features
# There are 0 prs without code features
# Total feature size is 1536
# There are 4691 repos without code features
# There are 102 repos without language features
# There are 2379 repos without text description features
# Total feature size is 1797
# There are 5196 issues without text description features

# /graph/content/issues.¡son
# There are 1081 prs without text description features
# There are 207697 ors without code features
# Total feature size is 1536
# There are 4691 repos without code features
# There are 102 repos without language features
# There are 2379 repos without text description features
# Total feature size is 1797
# There are 5196 issues without text description features

# ./full_graph/content/issues.json
# There are 1081 prs without text description features
# There are 207697 prs without code features
# Total feature size is 1536
# There are 4691 repos without code features
# There are 102 repos without language features
# There are 0 repos without topic features
# There are 349 repos without text description features
# Total feature size is 2048
# There are 5196 issues without text description features

class RepositoryFeatureLoader():
    def __init__(self, repo_idx_file) -> None:
        with open(repo_idx_file, "r", encoding="utf-8") as inf:
            self.repo_idx = json.load(inf)
        pass

    def load_code_feature_for_repo_node(self, repo_embedding_file, embed_size, device=torch.device("cpu")):
        code_features = np.zeros((len(self.repo_idx), embed_size), dtype=np.float32)
        missed_repos = 0
        with open(repo_embedding_file, "rb") as inf:
            repo_embedding = pickle.load(inf)
            for repo in self.repo_idx:
                if repo not in repo_embedding:
                    missed_repos += 1
                    continue
                code_features[self.repo_idx[repo]] = repo_embedding[repo]
        # Output: There are 4691 repos without code features
        print("There are {} repos without code features".format(missed_repos))
        return torch.FloatTensor(code_features.tolist()).to(device)

    def load_language_feature_for_repo_node(self, repo_embedding_file, embed_size, device=torch.device("cpu")):
        language_features = np.zeros((len(self.repo_idx), embed_size), dtype=np.float32)
        missed_repos = 0
        with open(repo_embedding_file, "rb") as inf:
            repo_embedding = pickle.load(inf)
            for repo in self.repo_idx:
                if repo not in repo_embedding:
                    missed_repos += 1
                    continue

                language_features[self.repo_idx[repo]] = repo_embedding[repo]
        
        # Output: There are 102 repos without language features
        print("There are {} repos without language features".format(missed_repos))
        return torch.FloatTensor(language_features.tolist()).to(device)

    def load_topic_feature_for_repo_node(self, repo_embedding_file, embed_size, device=torch.device("cpu")):
        topic_features = np.zeros((len(self.repo_idx), embed_size), dtype=np.float32)
        missed_repos = 0
        with open(repo_embedding_file, "rb") as inf:
            repo_embedding = pickle.load(inf)
            for repo in self.repo_idx:
                if repo not in repo_embedding:
                    missed_repos += 1
                    continue

                topic_features[self.repo_idx[repo]] = repo_embedding[repo]
        
        print("There are {} repos without topic features".format(missed_repos))
        return torch.FloatTensor(topic_features.tolist()).to(device)
    
    def load_text_feature_for_repo_node(self, repo_embedding_file, embed_size, device=torch.device("cpu")):
        text_features = np.zeros((len(self.repo_idx), embed_size), dtype=np.float32)
        missed_repos = 0
        with open(repo_embedding_file, "rb") as inf:
            repo_embedding = pickle.load(inf)
            for repo in self.repo_idx:
                if repo not in repo_embedding:
                    missed_repos += 1
                    continue

                text_features[self.repo_idx[repo]] = repo_embedding[repo]
        # Output: There are 2379 repos without text description features    
        print("There are {} repos without text description features".format(missed_repos))
        return torch.FloatTensor(text_features.tolist()).to(device)

    def load_embedding_for_repo_node(
            self, repo_code_embedding_file, code_embed_size, 
            repo_language_embedding_file, language_embed_size, 
            repo_topic_embedding_file, topic_embed_size, 
            repo_text_embedding_file, text_embed_size,
            device=torch.device("cpu"), include_topic=False, cache_file="repo_node_initial_embedding.pt", load_cache=True):
        cpu = torch.device("cpu")
        cache_path = None
        if include_topic:
            cache_path = os.path.join(CACHE_DIR, cache_file[:-3]+"_include_topic.pt")
            if load_cache and os.path.exists(cache_path):
                return torch.load(cache_path).to(device)
        else:
            cache_path = os.path.join(CACHE_DIR, cache_file[:-3]+"_exclude_topic.pt")
            if load_cache and os.path.exists(cache_path):
                return torch.load(cache_path).to(device)
            
        code_embedding = self.load_code_feature_for_repo_node(repo_code_embedding_file, code_embed_size, device)
        language_embedding = self.load_language_feature_for_repo_node(repo_language_embedding_file, language_embed_size, device)
        if include_topic:
            topic_embedding = self.load_topic_feature_for_repo_node(repo_topic_embedding_file, topic_embed_size, device)
        text_embedding = self.load_text_feature_for_repo_node(repo_text_embedding_file, text_embed_size, device)
        if include_topic:
            repo_initial_embeddings = torch.cat([text_embedding, code_embedding, language_embedding, topic_embedding], dim=1)
            print("Total feature size is {}".format(text_embed_size+code_embed_size+language_embed_size+topic_embed_size))
            torch.save(repo_initial_embeddings.to(cpu), cache_path)
            return repo_initial_embeddings

        # code_text_embedding = text_embedding + code_embedding
        # print("Total feature size is {}".format(code_embed_size+language_embed_size))
        # repo_initial_embeddings = torch.cat([code_text_embedding, language_embedding], dim=1)
        print("Total feature size is {}".format(code_embed_size+text_embed_size+language_embed_size))
        repo_initial_embeddings = torch.cat([text_embedding, code_embedding, language_embedding], dim=1)
        torch.save(repo_initial_embeddings.to(cpu), cache_path)
        return repo_initial_embeddings

class PRFeatureLoader():
    def __init__(self, pr_idx_file) -> None:
        with open(pr_idx_file, "r", encoding="utf-8") as inf:
            self.pr_idx = json.load(inf)
        pass

    def load_text_feature_for_pr_node(self, pr_embedding_file, embed_size, device=torch.device("cpu")):
        text_features = np.zeros((len(self.pr_idx), embed_size), dtype=np.float32)
        missed_prs = 0
        with open(pr_embedding_file, "rb") as inf:
            pr_embedding = pickle.load(inf)
            for pr in self.pr_idx:
                if pr not in pr_embedding:
                    missed_prs += 1
                    continue

                text_features[self.pr_idx[pr]] = pr_embedding[pr]
        # Output: 1081
        print("There are {} prs without text description features".format(missed_prs))
        return torch.FloatTensor(text_features.tolist()).to(device)
        pass

    def load_code_feature_for_pr_node(self, pr_embedding_file, embed_size, device=torch.device("cpu")):
        code_features = np.zeros((len(self.pr_idx), embed_size), dtype=np.float32)
        missed_prs = 0
        with open(pr_embedding_file, "rb") as inf:
            pr_embedding = pickle.load(inf)
            for pr in self.pr_idx:
                if pr not in pr_embedding:
                    missed_prs += 1
                    continue

                code_features[self.pr_idx[pr]] = pr_embedding[pr]
        # Output: 207697
        print("There are {} prs without code features".format(missed_prs))
        return torch.FloatTensor(code_features.tolist()).to(device)
        pass

    def load_embedding_for_pr_node(
            self, 
            pr_text_embdding_file, text_embed_size, 
            pr_code_embedding_file, code_embed_size,
            device=torch.device("cpu"), cache_file="pr_node_initial_embedding.pt", load_cache=True):
        cpu = torch.device("cpu")
        cache_path = os.path.join(CACHE_DIR, cache_file)
        if load_cache and os.path.exists(cache_path):
            return torch.load(cache_path).to(device)

        text_embedding = self.load_text_feature_for_pr_node(pr_text_embdding_file, text_embed_size, device)
        code_embedding = self.load_code_feature_for_pr_node(pr_code_embedding_file, code_embed_size, device)

        print("Total feature size is {}".format(code_embed_size+text_embed_size))
        # code_text_embedding = text_embedding + code_embedding
        pr_initial_embeddings = torch.cat([text_embedding, code_embedding], dim=1)
        torch.save(pr_initial_embeddings.to(cpu), cache_path)
        return pr_initial_embeddings

class IssueFeatureLoader():
    def __init__(self, issue_idx_file) -> None:
        print(issue_idx_file)
        with open(issue_idx_file, "r", encoding="utf-8") as inf:
            self.issue_idx = json.load(inf)
        pass

    def load_embedding_for_issue_node(
            self, text_embedding_file, embed_size,
            device=torch.device("cpu"), cache_file="issue_node_initial_embedding.pt", load_cache=True):
        cpu = torch.device("cpu")
        cache_path = os.path.join(CACHE_DIR, cache_file)
        if load_cache and os.path.exists(cache_path):
            return torch.load(cache_path).to(device)
        
        text_features = np.zeros((len(self.issue_idx), embed_size), dtype=np.float32)
        missed_issues = 0
        with open(text_embedding_file, "rb") as inf:
            issue_embedding = pickle.load(inf)
            for issue in self.issue_idx:
                if issue not in issue_embedding:
                    missed_issues += 1
                    continue

                text_features[self.issue_idx[issue]] = issue_embedding[issue]
        
        # Output: There are 5196 issues without text description features
        print("There are {} issues without text description features".format(missed_issues))
        issue_initial_embeddings = torch.FloatTensor(text_features.tolist()).to(device)
        torch.save(issue_initial_embeddings.to(cpu), cache_path)
        return issue_initial_embeddings