import os
import sys
import json

import torch
from torch.utils.data import random_split

if __name__ == "__main__":
    data_path = "./data/repo_organizer_samples.json"
    train_path = "./data/train.json"
    test_path = "./data/test.json"
    valid_path = "./data/valid.json"

    with open(data_path, "r", encoding="utf-8") as inf:
        samples = json.load(inf)
    keys = set()
    for s in samples:
        key, pos, neg = s
        keys.add(key)
    
    train_keys_length = int(len(keys) * 0.8)
    test_keys_length = int(len(keys) * 0.1)
    valid_keys_length = len(keys) - train_keys_length - test_keys_length
    train_keys, test_keys, valid_keys = random_split(list(keys), [train_keys_length, test_keys_length, valid_keys_length], generator=torch.Generator().manual_seed(42))

    train_keys = set(train_keys)
    test_keys = set(test_keys)
    valid_keys = set(valid_keys)
    
    train_samples = []
    test_samples = []
    valid_samples = []
    for s in samples:
        key, pos, neg = s
        if key in train_keys:
            train_samples.append(s)
        elif key in test_keys:
            test_samples.append(s)
        elif key in valid_keys:
            valid_samples.append(s)
        else:
            raise Exception("Error")
    
    with open(train_path, "w", encoding="utf-8") as ouf:
        json.dump(train_samples, ouf)
    with open(test_path, "w", encoding="utf-8") as ouf:
        json.dump(test_samples, ouf)
    with open(valid_path, "w", encoding="utf-8") as ouf:
        json.dump(valid_samples, ouf)
        