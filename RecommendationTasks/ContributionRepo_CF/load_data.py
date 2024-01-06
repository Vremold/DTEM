#!/usr/bin/env python

# 根据 reviewer 的说法, 我们只看 contribute 这个关系. 
# 放在这个工作中, 应该是构建repo的向量, 基于相似的repo, 推荐相关的contributor. 

# 噢, 真是意外! 原来数据已经在 GHCrawler/cleaned/repo_contributions.txt 下了! 

import numpy as np 
import json
from tqdm import tqdm

FILE_PATH = '../../GHCrawler/cleaned/repo_contributions.txt'


def convert(line: str): 
    name, extra = line.split('\t')
    extra = json.loads(extra)
    extra = {it[0]: it[1] for it in extra}
    return name, extra


if __name__ == '__main__':
    # STEP 1. load data
    with open(FILE_PATH) as fp: 
        data = [convert(it) for it in fp.readlines()]

    # STEP 2. GET contributors and repos; Get index_map

    # TODO MODIFY THIS: EXTRACT FILEPATH        

    # len(repos) = 50,000
    with open('../../GNN/DataPreprocess/full_graph/content/repositories.json') as fp: 
        repo_idxes = json.load(fp)

    # len(contributor_idxes) = 394,474
    with open('../../GNN/DataPreprocess/full_graph/content/contributors.json') as fp: 
        contributor_idxes = json.load(fp)
    print(len(contributor_idxes))

    size = len(contributor_idxes)  # size of embedding
    matrix = np.zeros((len(repo_idxes), size), dtype=np.int16)

    for repo_name, contributors_rec in tqdm(data): 
        for contributor_name, cnt in contributors_rec.items(): 
            matrix[repo_idxes[repo_name], contributor_idxes[contributor_name]] = min(cnt, 255)  # TODO 无奈的选择: 之后再改. 

    print(matrix[0])

    np.save('matrix.npy', matrix)

    
