import json
import sys
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

import numpy as np

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
    def __init__(self, contributor_idx, c_idxs, contributor_embedding, is_tensor=True) -> None:
        super().__init__()
        self.data = []
        src_embedding = contributor_embedding[contributor_idx]
        if is_tensor:
            src_embedding = src_embedding.cpu().numpy().tolist()
        for c_idx in c_idxs:
            dst_embedding = contributor_embedding[c_idx]
            if is_tensor:
                dst_embedding = dst_embedding.cpu().numpy().tolist()
            self.data.append([c_idx, src_embedding + dst_embedding])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    samples = []
    c_idxs = []
    for sample in batch:
        c_idxs.append(sample[0])
        samples.append(sample[1])
    return torch.FloatTensor(samples), torch.LongTensor(c_idxs)

if __name__ == "__main__":
    trained_model_path = "../model.bin"
    feat_size = 64
    validation_dataset = "./data/dataset_valid_test.json"
    dst_result_path = "./result/model_result_valid_test.json"

    trained_embedding_path = "../../../final_node_embedding.bin"
    all_embedding = torch.load(trained_embedding_path)
    contributor_embedding = all_embedding["contributor"]

    model = Net(embedding_dim=feat_size)
    model.load_state_dict(torch.load(trained_model_path, map_location="cpu"))

    with open(validation_dataset, "r", encoding="utf-8") as inf:
        dataset = json.load(inf)
    
    topks = {}
    for contributor_idx, search_scope, labels in dataset:
        if len(labels) < 5:
            continue
        d = MyDataset(contributor_idx, search_scope, contributor_embedding)
        dataloader = DataLoader(d, batch_size=128, shuffle=False, collate_fn=collate_fn)
        model.eval()
        output = {}
        with torch.no_grad():
            for batch in dataloader:
                samples, repo_idxs = batch
                results = model(samples).squeeze().numpy()
                if len(results.shape) == 0:
                    results = [results]
                for idx, result in zip(repo_idxs, results):
                    output[idx] = result
        
        output = sorted(output.items(), key=lambda x: x[1], reverse=True)
        output = [x[0] for x in output[:20]]
        # print(output)
        rets = np.zeros(21).tolist()
        for i in range(1, 21):
            rets[i] = rets[i - 1]
            if i-1 < len(output) and output[i - 1] in labels:
                rets[i] = rets[i - 1] + 1

        topks[contributor_idx] = (rets, search_scope)
    
    with open(dst_result_path, "w", encoding="utf-8") as ouf:
        json.dump(topks, ouf, indent=4)