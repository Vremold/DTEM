import os
import sys
import json
import pickle
import time

import numpy as np

class ProjectEmbeddingAggregator():
    def __init__(self, path_embedding_dir, path_n_funcs_dir, embed_size) -> None:
        self.path_embedding_fnames = []
        self.path_n_funcs_fnames = []
        for fname in os.listdir(path_embedding_dir):
            if fname.endswith("path_embedding.pkl"):
                self.path_embedding_fnames.append(fname)
            if fname.endswith("path_n_funcs.json"):
                self.path_n_funcs_fnames.append(fname)
        self.path_n_funcs_fnames.sort()
        self.path_embedding_fnames.sort()

        self.path_embedding_dir = path_embedding_dir
        self.path_n_funcs_dir = path_n_funcs_dir
        self.embed_size = embed_size
        pass

    def load_path_embedding(self):
        project_path_embeddings = {}
        project_path_n_funcs = {}
        for embed_fname, n_funcs_fname in zip(self.path_embedding_fnames, self.path_n_funcs_fnames):
            print(embed_fname, n_funcs_fname)
            with open(os.path.join(self.path_embedding_dir, embed_fname), "rb") as inf1, open(os.path.join(self.path_n_funcs_dir, n_funcs_fname), "r", encoding="utf-8") as inf2:
                tmp_embeddings = pickle.load(inf1)
                tmp_n_funcs = json.load(inf2)
                for project in tmp_embeddings:
                    if project not in project_path_embeddings:
                        project_path_embeddings[project] = {}
                        project_path_n_funcs[project] = {}
                    for path in tmp_embeddings[project]:
                        if path not in project_path_embeddings[project]:
                            project_path_embeddings[project][path] = tmp_embeddings[project][path]
                            project_path_n_funcs[project][path] = tmp_n_funcs[project][path]
                        else:
                            project_path_embeddings[project][path] += tmp_embeddings[project][path]
                            project_path_n_funcs[project][path] += tmp_n_funcs[project][path]
            
        for project in project_path_embeddings:
            for path in project_path_embeddings[project]:
                project_path_embeddings[project][path] /= project_path_n_funcs[project][path]
        return project_path_embeddings

    def aggregate_project_embedding_for_project(self, project_path_embeddings):
        project_embeddings = dict()
        for project in project_path_embeddings:
            n_paths = len(project_path_embeddings[project])
            embed_sum = np.zeros(self.embed_size, dtype=np.float32)
            for path in project_path_embeddings[project]:
                embed_sum += np.array(project_path_embeddings[project][path])
            project_embeddings[project] = embed_sum / n_paths

        return project_embeddings

if __name__ == "__main__":
    pea = ProjectEmbeddingAggregator(
        path_embedding_dir="./RepositoryCodeEmbedding/result",
        path_n_funcs_dir="./RepositoryCodeEmbedding/result",
        embed_size=768)
    project_path_embeddings = pea.load_path_embedding()
    with open("./export/repository_code_path_emebeddings.pkl", "wb") as outf:
        pickle.dump(project_path_embeddings, outf)
    print("Aggregate path embedding finished")
    project_embeddings = pea.aggregate_project_embedding_for_project(project_path_embeddings)
    with open("./export/repository_code_embeddings.pkl", "wb") as outf:
        pickle.dump(project_embeddings, outf)
    print("Aggregate project embedding finished")


