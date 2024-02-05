#!/usr/bin/env python3

'''
    真不妙, 用户的 bio 信息也是缺失的. 
'''
from tqdm import tqdm
from Comparisons.experiments.general import \
    github_token, load_contributor_index, \
    load_yaml_cfg, \
    load_json, save_json, \
    data_divide

from typing import Dict, List, Any, Optional, Generator, Tuple
from github import Github


def crawl(idx: int, gh_token_idx=-1): 
    cfg = load_yaml_cfg()['alpha']

    contributors: Dict[str, int] = list(load_contributor_index())
    contributors = list(data_divide(contributors, idx, 3))
    
    klee = Github(github_token(gh_token_idx))

    out_file = cfg['raw']['user_bio_file_path'] + f'.{idx}'

    proc_data: Dict[str, str] = load_json(out_file)  # key: user_name; value: bio

    try: 
        for user in tqdm(contributors): 
            if user in proc_data: continue
            try: 
                proc_data[user] = klee.get_user(user).bio
            except KeyboardInterrupt: 
                break
            except: continue
    finally: 
        save_json(out_file, proc_data)
