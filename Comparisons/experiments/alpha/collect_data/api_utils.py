#!/usr/bin/env python3

from Comparisons.experiments.general import \
    load_yaml_cfg, dict_invert

from typing import Any, Dict, Generator, List, Literal, Optional, Set, Tuple
import json, pickle
from collections import defaultdict

api_cfg = load_yaml_cfg()['alpha']['collect_data']['api']

'''
    return pr names
'''
def get_contr_prs() -> Dict[int, List[str]]: 

    def get_contributor_propose_pr_file() -> Generator[Tuple[int, int], None, None]: 
        with open(api_cfg['src']['contributor_propose_pr_file']) as fp:
            for line in fp: 
                user_id, pr_id, _ = line.split('\t')
                yield int(user_id), int(pr_id)

    with open(api_cfg['src']['pr_idx_file']) as fp: 
        pr_idx2name = dict_invert(json.load(fp))

    ret = defaultdict(list)
    for user_id, pr_id in get_contributor_propose_pr_file():
        ret[user_id].append(pr_idx2name[pr_id])

    return dict(ret)

if True:
# if __name__ == '__main__': 
    '''
    数量少得可怜啊! 只有95532个开发者是有pr过的. 这算什么事儿嘛! (总共有394474个开发者, 1/4 左右)
    '''
    # len(get_contr_prs()) == 95532 
    with open(api_cfg['dst']['contributor_api_names_file'], 'wb') as fp: 
        pickle.dump(get_contr_prs(), fp)
    
