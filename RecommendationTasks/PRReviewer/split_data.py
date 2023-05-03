import os
import sys
import json

import torch
from torch.utils.data import random_split

if __name__ == "__main__":
    data_path = "./data/pr_reviewer.json"
    train_path = "./data/train.json"
    test_path = "./data/test.json"
    valid_path = "./data/valid.json"

    with open(data_path, "r", encoding="utf-8") as inf:
        samples = json.load(inf)
    
    train_length = int(len(samples) * 0.8)
    test_length = int(len(samples) * 0.1)
    valid_length = len(samples) - train_length - test_length
    train_samples, test_samples, valid_samples = random_split(samples, [train_length, test_length, valid_length], generator=torch.Generator().manual_seed(42))

    train_samples = list(train_samples)
    test_samples = list(test_samples)
    valid_samples = list(valid_samples)

    with open(train_path, "w", encoding="utf-8") as ouf:
        json.dump(train_samples, ouf)
    with open(test_path, "w", encoding="utf-8") as ouf:
        json.dump(test_samples, ouf)
    with open(valid_path, "w", encoding="utf-8") as ouf:
        json.dump(valid_samples, ouf)
        