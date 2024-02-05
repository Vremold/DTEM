#!/usr/bin/env python3

from re import search
from RecommendationTasks.PRReviewer_CF.config import \
        CF_DICT_FILE_PATH, VALID_TEST_DATA_FILE_PATH, VALID_RESULT_PATH, \
        load_data

from RecommendationTasks.PRReviewer_CF.model  \
    import CollaborativeFiltering

import json
from typing import List, Dict, Tuple


def evaluate(model_postfix='full', partial=False): 
    '''
        这个函数用于评估协同过滤(CF)模型的效果. 模型的选择由 model_postfix 指定.
        会生成一个中间文件. 该文件放在 ./result 中. 

        model_postfix 的可选项可以在 
        RecommendationTasks/ContributionRepo_CF/bin/
        下找到. 
        比如如果存在文件 cf_dict.pkl.top40, 则可用的 postfix 为 top40 

        see also: 
            ../model.py -> train_model()
            ./metric/metric.py -> metric()
    '''

    if partial: model_postfix += '.partial'
    klee = CollaborativeFiltering()
    klee.load_pickle(CF_DICT_FILE_PATH + '.' + model_postfix)

    # @see: RecommendationTasks/ContributionRepo/README.md
    dataset = load_data(VALID_TEST_DATA_FILE_PATH)

    result: Dict[int, Tuple[List[int], List[int]]] = {}

    # ../README.md 解释了这个dataset
    
    cnt = 0
    for repo_id, pr_id, search_scope, gt in dataset:
        if len(gt) < 5: continue        # 345 recommendations with gt>=5 in total
        cnt += 1
        
        recs: List[int] = klee.recommend(repo_id, search_scope)[:20]
        acc: List[int] = [0] * 21
        
        # count for top-k
        for i in range(20):
            acc[i + 1] = acc[i] + (i < len(recs) and recs[i] in gt)
        result[repo_id] = (acc, search_scope)
    
    with open(VALID_RESULT_PATH + '.' + model_postfix, 'w') as fp:
        json.dump(result, fp)
    print(str(cnt) + "recommendations finished")