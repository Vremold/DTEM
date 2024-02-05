import json
import sys
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import numpy as np
from tqdm import tqdm

from RecommendationTasks.SimDeveloper.train_nn import \
        Net as SimDeveloperNet, \
        DataLoader

from RecommendationTasks.SimDeveloper.metric.validate_model import \
        MyDataset as DataSet, \
        collate_fn

from ..general import \
    load_yaml_cfg

cfg = load_yaml_cfg()['alpha']
task_cfg = cfg['tasks']['sim_developer']


'''
这个文件自然是参考了: 
        RecommendationTasks/SimDeveloper/metric/validate_model.py 
'''
def validate(device=torch.device('cpu')):

    feat_size = 580

    trained_model_path = task_cfg['model']['model_file']
    validation_dataset = task_cfg['data']['valid_test_file']

    dst_result_path = task_cfg['result']['valid_test_result']

    trained_embedding_path = cfg['embedding']['contributor_merged_embedding']
    all_embedding = torch.load(trained_embedding_path)
    contributor_embedding = all_embedding

    model = SimDeveloperNet(embedding_dim=feat_size)
    model.load_state_dict(torch.load(trained_model_path))

    with open(validation_dataset, "r", encoding="utf-8") as inf:
        dataset = json.load(inf)
    
    topks = {}
    for contributor_idx, search_scope, labels in tqdm(dataset):
        if len(labels) < 5:
            continue
        d = DataSet(contributor_idx, search_scope, contributor_embedding)
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


validate()