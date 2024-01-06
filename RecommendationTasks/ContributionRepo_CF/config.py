#!/usr/bin/env python

import os, json
from typing import Dict, Tuple, List

from tqdm.rich import tqdm


CONTRIBUTE_FILE_PREFIX = 'RecommendationTasks/ContributionRepo'

# MAYBE DEPRECATED
TEST_DATA_FILE_PATH         = f'{CONTRIBUTE_FILE_PREFIX}/data/test.json'
TRAIN_DATA_FILE_PATH        = f'{CONTRIBUTE_FILE_PREFIX}/data/train.json'
VALID_DATA_FILE_PATH        = f'{CONTRIBUTE_FILE_PREFIX}/data/valid.json'

CF_DICT_FILE_PATH = 'RecommendationTasks/ContributionRepo_CF/bin/cf_dict.pkl'

VALID_TEST_DATA_FILE_PATH   = f'{CONTRIBUTE_FILE_PREFIX}/metric/data/dataset_valid_test.json'
VALID_RESULT_PATH           = f'RecommendationTasks/ContributionRepo_CF/metric/result/result_valid_test.json'

CONTRIBUTION_FILE_PATH      = 'GHCrawler/cleaned/repo_contributions.txt'
REPOSITORY_FILE_PATH        = 'GNN/DataPreprocess/full_graph/content/repositories.json'
CONTRIBUTOR_FILE_PATH       = 'GNN/DataPreprocess/full_graph/content/contributors.json'


# check files exists
for it in {
    TEST_DATA_FILE_PATH,
    TRAIN_DATA_FILE_PATH,
    VALID_DATA_FILE_PATH,

    VALID_TEST_DATA_FILE_PATH,

    CONTRIBUTION_FILE_PATH,
    REPOSITORY_FILE_PATH,
    CONTRIBUTOR_FILE_PATH,
}: 
    assert os.path.exists(it)


def load_data(filepath: str) -> List[Tuple[int, int, int]]: 
    """
        see also: 
            RecommendationTasks/ContributionRepo_CF/metric/metric.py
        
        这个函数加载的数据对应 实验4. 
        
        This function is used to load such these 4 files: 
            - TEST_DATA_FILE_PATH
            - TRAIN_DATA_FILE_PATH
            - VALID_DATA_FILE_PATH

        
        for each element `it` in ret: 
        it[0]: contributor id
        it[1]: a positive repo id
        it[2]: a negative repo id

        hint: In this directory (RecommendationTasks/ContributionRepo_CF/), 
        we ignore negative repo (i.e. it[2]).
    """
    with open(filepath) as fp: 
        return json.load(fp)

def load_repo_contributions() -> List[Tuple[int, int, int]]: 

    """
        see also: 
            RecommendationTasks/ContributionRepo_CF/metric/metric.py
        对应其中的 实验3. 

        [DELETING] 这个实验有问题, 这个函数调起的数据和训练 model.bin 的不符. 
            应该去掉此实验, 此函数也没用了. [/]
    """

    with open(REPOSITORY_FILE_PATH) as fp: 
        repo_idxes = json.load(fp)

    # len(contributor_idxes) = 394,474
    with open(CONTRIBUTOR_FILE_PATH) as fp: 
        dev_idxes = json.load(fp)

    with open(CONTRIBUTION_FILE_PATH) as fp: 
        def convert(line: str): 
            name, extra = line.split('\t')
            extra = json.loads(extra)
            extra = {it[0]: it[1] for it in extra}
            return name, extra
        contributions = [convert(it) for it in fp.readlines()]
    
    ret: List[Tuple[int, int]] = []
    for repo_name, extra in tqdm(contributions): 
        for dev_name in extra: 
            ret.append([dev_idxes[dev_name], repo_idxes[repo_name]])
    return ret

