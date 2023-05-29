import os
import sys
import json
import pickle
import time

import numpy as np

class PRCodeEmbeddingAggregator(object):
    def __init__(self, project_path_embedding_file, embed_size) -> None:
        with open(project_path_embedding_file, "rb") as inf:
            self.project_path_embedding = pickle.load(inf)
        
        for project in self.project_path_embedding:
            for path in self.project_path_embedding[project]:
                assert self.project_path_embedding[project][path].shape[0] == embed_size
                break
            break
        
        self.embed_size = embed_size
        pass

    def load_pr_code_embedding(self, pr_modified_paths_file):
        pr_code_embeddings = {}
        with open(pr_modified_paths_file, "r", encoding="utf-8") as inf:
            pr_modified_paths = json.load(inf)
        for pr_name in pr_modified_paths:
            repo_name = pr_name.split("##")[0]
            n_path = 0
            pr_code_embeddings[pr_name] = np.zeros(self.embed_size, dtype=np.float32)
            for path in pr_modified_paths[pr_name]:
                if repo_name not in self.project_path_embedding:
                    continue
                if path not in self.project_path_embedding[repo_name]:
                    continue
                pr_code_embeddings[pr_name] += self.project_path_embedding[repo_name][path]
                n_path += 1
            if n_path == 0:
                continue
            print("Valid percent: {}".format(n_path / len(pr_modified_paths[pr_name])))
            pr_code_embeddings[pr_name] /= n_path
        
        return pr_code_embeddings

if __name__ == "__main__":
    pcea = PRCodeEmbeddingAggregator(
        project_path_embedding_file="./export/repo_code_path_emebeddings.pkl", 
        embed_size=768)
    pr_code_embeddings = pcea.load_pr_code_embedding("./PREmbedding/pr_modified_paths.json")
    
    # delete this object for freeing memory
    del pcea

    with open("./export/pr_code_embedding.pkl", "wb") as outf:
        pickle.dump(pr_code_embeddings, outf)