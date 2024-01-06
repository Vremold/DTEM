#!/usr/bin/env python3

from torch import topk
from RecommendationTasks.ContributionRepo_CF.config import \
    CF_DICT_FILE_PATH, VALID_TEST_DATA_FILE_PATH, load_data

from RecommendationTasks.ContributionRepo_CF.model  \
    import CollaborativeFiltering

from typing import List, Dict, Tuple


klee = CollaborativeFiltering()
klee.load_pickle(CF_DICT_FILE_PATH)

# @see: RecommendationTasks/ContributionRepo/README.md
dataset = load_data(VALID_TEST_DATA_FILE_PATH)

result: Dict[int, Tuple[List[int], List[int]]] = {}

# dev_id: int
# search_scope: List[int], 此开发者开发过的所有仓库(从图构建的). 
# gt: List[int], ground truth. 在训练的过程中没有看到的仓库列表. 
# 这个验证是希望说明, 在协同过滤(CF)中没有看到的仓库, 在此阶段也可以通过它在search_scope中搜索得到. 
for dev_id, search_scope, gt in dataset: 
    if len(gt) < 5: continue

    # Exists dev_id, s.t. len(recs) <= 20. Be careful. 
    # recs: recommendations
    recs: List[int] = klee.recommend(dev_id, search_scope)[:20]

    acc: List[int] = [0] * 21
    for i in range(20):
        acc[i + 1] = acc[i] + (i < len(recs) and recs[i] in gt)

    print(recs)
    exit(0)

    result[dev_id] = (acc, search_scope)


    pass