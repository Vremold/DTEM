import os
import sys
import pickle
import json

import numpy as np

class TextEmbeddingAggregator(object):
    @staticmethod
    def load_issue_embedding(text_path, embedding_path, embed_size=None):
        issue_embedding = dict()
        with open(text_path, "r", encoding="utf-8") as textinf, open(embedding_path, "r", encoding="utf-8") as embinf:
            for emb_line in embinf:
                embs = json.loads(emb_line)
                bsz = len(embs)
                for i in range(bsz):
                    assert len(embs[i]) == embed_size
                    text_line = textinf.readline()
                    text_line = json.loads(text_line)
                    issue_name = "{}#{}".format(text_line["project"], text_line["number"])
                    issue_embedding[issue_name] = np.array(embs[i])
        return issue_embedding

    @staticmethod
    def load_pr_embedding(text_path, embedding_path, embed_size=None):
        pr_embedding = dict()
        with open(text_path, "r", encoding="utf-8") as textinf, open(embedding_path, "r", encoding="utf-8") as embinf:
            for emb_line in embinf:
                embs = json.loads(emb_line)
                bsz = len(embs)
                for i in range(bsz):
                    assert len(embs[i]) == embed_size
                    text_line = textinf.readline()
                    text_line = json.loads(text_line)
                    pr_name = "{}##{}".format(text_line["project"], text_line["number"])
                    pr_embedding[pr_name] = np.array(embs[i])
        return pr_embedding

    @staticmethod
    def load_repo_embedding(text_path, embedding_path, embed_size=None):
        repo_embedding = dict()
        with open(text_path, "r", encoding="utf-8") as textinf, open(embedding_path, "r", encoding="utf-8") as embinf:
            for emb_line in embinf:
                embs = json.loads(emb_line)
                bsz = len(embs)
                for i in range(bsz):
                    assert len(embs[i]) == embed_size
                    text_line = textinf.readline()
                    text_line = json.loads(text_line)
                    repo_name = text_line["project"]
                    repo_embedding[repo_name] = np.array(embs[i])
        return repo_embedding
    
if __name__ == "__main__":
    issue_embedding = TextEmbeddingAggregator.load_issue_embedding(
        text_path="./IssueEmbedding/issue_descriptions.txt",
        embedding_path="./IssueEmbedding/issue_description_embedding.txt",
        embed_size=768
    )
    with open("./export/issue_text_embedding.pkl", "wb") as outf:
        pickle.dump(issue_embedding, outf)
    print("Loading issue text embedding finished")
    pr_embedding = TextEmbeddingAggregator.load_pr_embedding(
        text_path="./PREmbedding/pr_descriptions.txt",
        embedding_path="./PREmbedding/pr_description_embedding.txt",
        embed_size=768
    )
    with open("./export/pr_text_embedding.pkl", "wb") as outf:
        pickle.dump(pr_embedding, outf)
    print("Loading pr text embedding finished")
    repo_embedding = TextEmbeddingAggregator.load_repo_embedding(
        text_path="./RepositoryEmbedding/repo_descriptions.txt",
        embedding_path="RepositoryEmbedding/repo_description_embedding.txt",
        embed_size=768
    )
    with open("./export/repo_text_embedding.pkl", "wb") as outf:
        pickle.dump(repo_embedding, outf)
    print("Loading repo text embedding finished")
