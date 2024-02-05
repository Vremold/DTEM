import os
import json

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset

class Graph():
    def __init__(self, commit_rels_f) -> None:
        self.commit_rels = {}
        self.reverse_commit_rels = {}
        self.following_rels = {}
        self.follower_rels = {}
        self.star_rels = {}
        self.watch_rels = {}
        with open(commit_rels_f, "r", encoding="utf-8") as inf:
            for line in inf:
                s, d, w = line.strip().split("\t")
                s, d = int(s), int(d)
                if s not in self.commit_rels:
                    self.commit_rels[s] = []
                if d not in self.reverse_commit_rels:
                    self.reverse_commit_rels[d] = []
                self.commit_rels[s].append(d)
                self.reverse_commit_rels[d].append(s)
    
    def get_contributor_by_repo(self, repo_idx):
        return self.reverse_commit_rels.get(repo_idx, [])

class ContributorIdx2Name():
    def __init__(self, contributors_f) -> None:
        with open(contributors_f, "r", encoding="utf-8") as inf:
            self.contributor_idx = json.load(inf)
        self.idx_contributor = {v: k for k, v in self.contributor_idx.items()}
        pass

    def get_name(self, idx):
        return self.idx_contributor.get(idx, "Unknown")

class Net(nn.Module):
    def __init__(self, embedding_dim):
        super(Net, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(self.embedding_dim * 2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.sigmoid(x).squeeze()

class MyDataset(Dataset):
    def __init__(self, repo_idx, contributor_idxs, repo_embedding, contributor_embedding, is_tensor=True) -> None:
        super().__init__()
        self.data = []
        print(repo_idx, contributor_idxs)
        src_embedding = repo_embedding[repo_idx]
        if is_tensor:
            src_embedding = src_embedding.cpu().numpy().tolist()
        for contributor_idx in contributor_idxs:
            dst_embedding = contributor_embedding[contributor_idx]
            if is_tensor:
                dst_embedding = dst_embedding.cpu().numpy().tolist()
            self.data.append([contributor_idx, src_embedding + dst_embedding])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    samples = []
    contributor_idxs = []
    for sample in batch:
        contributor_idxs.append(sample[0])
        samples.append(sample[1])
    return torch.FloatTensor(samples), torch.LongTensor(contributor_idxs)


if __name__ == "__main__":
    trained_model_path = "./bin/model.bin"
    node_embedding_path = "../../GNN/HetSAGE/node_embedding//node_embedding.bin"
    feat_size = 512
    repo_idxs = 16328

    all_embedding = torch.load(node_embedding_path)
    contributor_embedding = all_embedding["contributor"]
    repo_embedding = all_embedding["repository"]

    g = Graph(
        commit_rels_f="../../GNN/DataPreprocess/full_graph/content/contributor_commit_repo.txt",
    )
    ci2i = ContributorIdx2Name(
        contributors_f="../../GNN/DataPreprocess/full_graph/content/contributors.json",
    )

    model = Net(feat_size)
    model.load_state_dict(torch.load(trained_model_path, map_location="cpu"))

    candidate_contributor_idx = g.get_contributor_by_repo(repo_idxs)
    dataset = MyDataset(repo_idxs, candidate_contributor_idx, repo_embedding, contributor_embedding)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    output = {}
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            samples, contributor_idxs = batch
            contributor_idxs = contributor_idxs.numpy().tolist()
            results = model(samples).squeeze().numpy()
            if len(results.shape) == 0:
                results = [results]
            for idx, result in zip(contributor_idxs, results):
                output[idx] = result
    
    output = sorted(output.items(), key=lambda x: x[1], reverse=True)
    output = [x[0] for x in output[:20]]
    output_names = [ci2i.get_name(x) for x in output]
    print(output_names)