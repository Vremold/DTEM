#!/usr/bin/env python3 

'''
这个脚本用来获取每一个开发者的commits. 
会筛选掉不在我们考虑范围内的文件. 
'''

from typing import Any, Dict, Generator, List, Literal, Optional, Set, Tuple

from Comparisons.experiments.general import \
    load_yaml_cfg, \
    dict_invert, dict_inspect, \
    pickle_load, pickle_dump, \
    file_readlines

from Comparisons.experiments.alpha.collect_data.api_utils import get_contr_prs
from Comparisons.experiments.alpha.collect_data.api import ApiTextExtractor

import pickle, json

from collections import Counter 


api_cfg = load_yaml_cfg()['alpha']['collect_data']['api']


if False: 
    # STEP 1. Get contributors' prs (store in file)
    with open(api_cfg['dst']['contributor_api_names_file'], 'rb') as fp: 
        contr_prs: Dict[int, List[str]] = pickle.load(fp)

    pr_contrs: Dict[str, int] = {
        pr_name: contr_id 
        for contr_id, pr_names in contr_prs.items() 
        for pr_name in pr_names
    }

    # STEP 2. read the huge file, get required files (for each pr)
    def get_pr_files(total=1829584 + 10):
        @file_readlines(
                filename=api_cfg['src']['repo_pr_commits_file'], 
                total=total)
        def gen(line: str) -> Generator[Tuple[str, List[Any]], None, None]: 
            repo_name, pr_idx, _, content = line.split('\t')
            pr_name = f'{repo_name}##{pr_idx}'
            content = json.loads(content)
            return pr_name, content
        
        for it in gen(): 
            yield it 


    contr_pr_files: Dict[int, List[Any]] = {
        contr: []
        for contr in contr_prs
    }

    for issue_name, content in get_pr_files(): 
        contr_id = pr_contrs.get(issue_name) 
        if contr_id is None: continue

        for item in content: 
            if 'file' in item: continue
            contr_pr_files[contr_id].append({
                'filename': item['filename'], 
                'raw_url': item['raw_url'], 
            })

    print(len([it for it in contr_pr_files if contr_pr_files[it] != []]))  # 50178
    pickle_dump(contr_pr_files, api_cfg['dst']['hugefile_cleaned'])


contr_pr_files = pickle_load(api_cfg['dst']['hugefile_cleaned'])
print(sum([len(it) for it in contr_pr_files.values()]))
c = Counter([len(it) for it in contr_pr_files.values()])







# print(dict_inspect(pr_contrs, 10))



# STEP 3. 