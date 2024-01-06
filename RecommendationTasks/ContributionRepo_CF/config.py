#!/usr/bin/env python

import os, json
from typing import Dict, Tuple, List


CONTRIBUTE_FILE_PREFIX = 'RecommendationTasks/ContributionRepo'

TEST_DATA_FILE_PATH         = f'{CONTRIBUTE_FILE_PREFIX}/data/test.json'
TRAIN_DATA_FILE_PATH        = f'{CONTRIBUTE_FILE_PREFIX}/data/train.json'
VALID_DATA_FILE_PATH        = f'{CONTRIBUTE_FILE_PREFIX}/data/valid.json'

CF_DICT_FILE_PATH = 'RecommendationTasks/ContributionRepo_CF/bin/cf_dict.pkl'

VALID_TEST_DATA_FILE_PATH   = f'{CONTRIBUTE_FILE_PREFIX}/metric/data/dataset_valid_test.json'

# check files exists
assert os.path.exists(TEST_DATA_FILE_PATH)
assert os.path.exists(TRAIN_DATA_FILE_PATH)
assert os.path.exists(VALID_DATA_FILE_PATH)
assert os.path.exists(VALID_DATA_FILE_PATH)

def load_data(filepath: str) -> List[Tuple[int, int, int]]: 
    """
        for each element `it` in ret: 
        it[0]: contributor id
        it[1]: a positive repo id
        it[2]: a negative repo id

        hint: In this directory (RecommendationTasks/ContributionRepo_CF/), 
        we ignore negative repo (i.e. it[2]).
    """
    with open(filepath) as fp: 
        return json.load(fp)