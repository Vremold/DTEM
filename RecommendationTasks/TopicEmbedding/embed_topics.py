import os
import sys
import json
import pickle

import numpy as np

class TopicEmbedder():
    def __init__(self, repo_topic_file) -> None:
        super().__init__()
        with open(repo_topic_file, "r", encoding="utf-8") as inf:
            repo_topic = json.load(inf)
            self.topics = {}
            for repo in repo_topic:
                for t in repo_topic[repo]:
                    self.topics[t] = self.topics.get(t, 0) + 1
        
        # filter out topics with less than 10 occurrences
        selected_topics = [t for t in self.topics if self.topics[t] > 10]
        self.featured_topics = {t: i for i, t in enumerate(selected_topics)}
        self.n_feature_topics = len(self.featured_topics)
        print(len(self.featured_topics))
        pass

    def embed_contributor(self, contributor_topic_file, contributor_idx_file):
        with open(contributor_idx_file, "r", encoding="utf-8") as inf:
            contributor_idx = json.load(inf)
        
        contributor_topic_embedding = {}
        with open(contributor_topic_file, "r", encoding="utf-8") as inf:
            contributor_topic = json.load(inf)
            for contributor in contributor_topic:
                c_idx = contributor_idx[contributor]
                topic = contributor_topic[contributor]
                topic_embedding = np.zeros(self.n_feature_topics).tolist()
                for t in topic:
                    if t not in self.featured_topics:
                        continue
                    topic_embedding[self.featured_topics[t]] = 1
                contributor_topic_embedding[c_idx] = topic_embedding
        return contributor_topic_embedding

    def embed_repo(self, repo_topic_file, repo_idx_file):
        with open(repo_idx_file, "r", encoding="utf-8") as inf:
            repo_idx = json.load(inf)
        
        repo_topic_embedding = {}
        with open(repo_topic_file, "r", encoding="utf-8") as inf:
            repo_topic = json.load(inf)
            for repo in repo_topic:
                r_idx = repo_idx[repo]
                topic = repo_topic[repo]
                topic_embedding = np.zeros(self.n_feature_topics).tolist()
                for t in topic:
                    if t not in self.featured_topics:
                        continue
                    topic_embedding[self.featured_topics[t]] = 1

                repo_topic_embedding[r_idx] = topic_embedding
        return repo_topic_embedding

if __name__ == "__main__":
    contributor_topic_file = "./data/contributor_topics.json"
    contributor_idx_file = "../../GNN/DataPreprocess/full_graph/content/contributors.json"
    repo_topic_file = "./data/repo_topics.json"
    repo_idx_file = "../../GNN/DataPreprocess/full_graph/content/repositories.json"

    topic_embedder = TopicEmbedder(repo_topic_file)

    contributor_topic_embedding = topic_embedder.embed_contributor(contributor_topic_file, contributor_idx_file)
    with open("./embed/contributor_topic_embedding.pkl", "wb",) as outf:
        pickle.dump(contributor_topic_embedding, outf)
    repo_topic_embedding = topic_embedder.embed_repo(repo_topic_file, repo_idx_file)
    with open("./embed/repo_topic_embedding.pkl", "wb",) as outf:
        pickle.dump(repo_topic_embedding, outf)