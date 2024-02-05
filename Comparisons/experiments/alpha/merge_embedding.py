#!/usr/bin/env python3 

from typing import Dict
from ..general import load_yaml_cfg, load_contributor_index
import pickle
import numpy as np
from tqdm import tqdm
import torch 

cfg = load_yaml_cfg()['alpha']

emb_cfg = cfg['embedding']  # embedding configurations 

# FLAG SNAPPED! 
# 假设我们已经获得了 emb_cfg 中所指示的前三个嵌入向量 (contributor_{repo,issue,api}_embedding), 
# 接下来, 我们将为所有的开发者, 合并剩下的嵌入向量. 
def main(): 
    '''
        上述三个文件的内容是这样的: 
        Dict[str, np.ndarray], str 为开发者的名字, np.ndarray 是这个开发者的嵌入. 
        后续的工作, 我们将复用 RecommendationTasks/SimDeveloper中的train,test,valid集合, 
        而这些文件用的是编号. 

        所以我们要做的是这样几件事: 
        1. 把嵌入合并起来; 
        2. 在输出文件中, 用编号来表示开发者.
        3. 转换为 torch.Tensor 的格式. 

        @see also: 
            RecommendationTasks/SimDeveloper/train_nn.py
            RecommendationTasks/SimDeveloper/data/*
    '''

    embs = []
    for emb_type in {'repo', 'issue', 'api'}:
        with open(emb_cfg[f'contributor_{emb_type}_embedding'], 'rb') as fp: 
            embs.append(pickle.load(fp))

    contr_idx = load_contributor_index()  # contributor indices

    ret = [None] * len(contr_idx)
    for name, idx in tqdm(contr_idx.items()): 
        ret[idx] = np.concatenate([it[name] for it in embs])

    ret = np.array(ret)

    ret = torch.from_numpy(ret)
    torch.save(ret, emb_cfg['contributor_merged_embedding'])  # 394474 x 580


main()