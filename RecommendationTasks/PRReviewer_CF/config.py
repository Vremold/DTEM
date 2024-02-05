#!/usr/bin/env python

import os, json
from typing import Dict, Tuple, List

from tqdm import tqdm


FILE_PREFIX = 'RecommendationTasks/PRReviewer_CF'

# MAYBE DEPRECATED
CF_DICT_FILE_PATH = 'RecommendationTasks/PRReviewer_CF/bin/cf_dict.pkl'

VALID_TEST_DATA_FILE_PATH   = f'{FILE_PREFIX}/metric/data/dataset_valid_test.json'
VALID_RESULT_PATH           = f'{FILE_PREFIX}/metric/result/result_valid_test.json'

PR_FILE_PATH                = 'GHCrawler/cleaned/repo_prs.txt'
PR_PARTIAL_FILE_PATH        = 'RecommendationTasks/PRReviewer_CF/data/train.json'
REPOSITORY_FILE_PATH        = 'GNN/DataPreprocess/full_graph/content/repositories.json'
REVIEWER_FILE_PATH          = 'GNN/DataPreprocess/full_graph/content/contributors.json'


# check files exists
for it in {
    VALID_TEST_DATA_FILE_PATH,
    PR_FILE_PATH,
    REPOSITORY_FILE_PATH,
    REVIEWER_FILE_PATH,
}: 
    assert os.path.exists(it)


def load_data(filepath: str) -> List[Tuple[int, int, int]]: 
    with open(filepath) as fp: 
        return json.load(fp)

def load_repo_prs() -> List[Tuple[int, int]]: 
    with open(REPOSITORY_FILE_PATH) as fp: 
        repo_idxes = json.load(fp)

    # len(contributor_idxes) = 394,474
    with open(REVIEWER_FILE_PATH) as fp: 
        reviewer_idxes = json.load(fp)

    with open(PR_FILE_PATH) as fp: 
        def convert(line: str): 
            name, extra = line.split('\t')
            extra = json.loads(extra)
            extra = [it['committer'] for it in extra]
            return name, extra
        prs = [convert(it) for it in fp.readlines()]
    
    ret: List[Tuple[int, int]] = []
    print("Loading all PRs")
    for repo_name, extra in tqdm(prs): 
        for reviewer_name in extra: 
            ret.append([reviewer_idxes[reviewer_name], repo_idxes[repo_name]])
    return ret


# 使用train.json来训练CF，而不是全局信息
def load_partial_repo_prs() -> List[Tuple[int, int]]:
    with open(PR_PARTIAL_FILE_PATH) as fp:
        prs = json.load(fp)
    ret: List[Tuple[int, int]] = []
    print('Loading partial PRs')
    for repo_idx, pr_idx, pos_reviewer_idx, neg_reviewer_idx in prs:
        ret.append([repo_idx, pos_reviewer_idx])
    return ret

