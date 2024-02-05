#!/usr/bin/evn python3

import yaml, json, pickle
from tqdm import tqdm
from typing import Callable, Dict, List, Any, Optional, Generator, TypedDict, Tuple
import random
import os, sys

class RepoDict(TypedDict): 
    name:   str
    tags:   List[str]
    topic:  str
    readme: str         # 本来应该也有这个的, 但数据实在是太大了. 重新复制一份也没有意义. 

class IssueDict(TypedDict): 
    name:    str  # e.g. datalux/osintgram#670
    title:   str
    content: str 


def ignore_exception(f): 
    try: 
        f()
    except KeyboardInterrupt as e: 
        raise e
    except:
        print('exception occurs...', file=sys.stderr)

CONFIG_FILE_PATH = 'Comparisons/experiments/config.yaml'
cfg: Dict[str, Any]

with open(CONFIG_FILE_PATH) as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)

def load_yaml_cfg() -> Dict[str, Any]: 
    return cfg

def load_contributor_index() -> Dict[str, int]: 
    with open(cfg['general']['filepath']['contributor_idx_file']) as fp: 
        return json.load(fp)

def load_repository_index() -> Dict[str, int]: 
    with open(cfg['general']['filepath']['repository_idx_file']) as fp: 
        return json.load(fp)
    
def load_issue_index() -> Dict[str, int]: 
    with open(cfg['general']['filepath']['issue_idx_file']) as fp: 
        return json.load(fp)
    
def load_contributor_commit_repo() -> Dict[int, Dict[int, int]]: 
    '''
        ret[user_id][repo_id] = cnt, 
        indicates that 
            the user (with `user_id`) contributed repo (with `repo_id`) `cnt` times. 
    '''

    ret: Dict[int, Dict[int, int]] = {}

    with open(cfg['general']['filepath']['contributor_commit_repo_file']) as fp: 
        for line in fp: 
            lst = line.split('\t')
            user_id, repo_id, commit_cnt = int(lst[0]), int(lst[1]), int(lst[2])
            ret.setdefault(user_id, {})
            ret[user_id].setdefault(repo_id, 0)
            ret[user_id][repo_id] += commit_cnt

    return ret

def file_readlines(filename: str, total: int=-1): 
    def decorator(f): 
        def ret(): 
            with open(filename) as fp: 
                gen = (it for it, _ in zip(fp, range(total))) if total != -1 \
                    else (it for it in fp)
                gen = tqdm(gen, total=total) if total != -1 else gen
                for line in gen: 
                    yield f(line)
        return ret

    return decorator

'''
    构建倒排表: Dict[A, B] => Dict[B, A]
'''
def dict_invert(d: Dict[Any, Any]) -> Dict[Any, Any]: 
    return { v: k for k, v in d.items() }

def dict_invert_mul(d: Dict[Any, Any]) -> Dict[Any, List[Any]]: 
    ret = {}
    for k, v in d.items(): 
        if v not in ret: ret[v] = []
        ret[v].append(k)
    return ret

'''
    取字典的前k个元素看看. 
    主要是中间过程为了确定是否执行正确做的. 
'''
def dict_inspect(d: Dict[Any, Any], size=3) -> Dict[Any, Any]: 
    return {k: v for (_, (k, v)) in zip(range(size), d.items())}

def github_token(idx=-1) -> str: 
    tokens = cfg['general']['tokens']
    idx = idx % len(tokens) if idx != -1 else \
          random.choice(range(len(tokens)))
    print(f'using github token {idx}.')
    return tokens[idx]


def load_jsonl(filepath: str) -> Generator[Dict, None, None]: 
    if not os.path.exists(filepath): return
    with open(filepath) as fp: 
        for line in fp: 
            yield json.loads(line)

def save_jsonl(filepath: str, data: Generator[Dict, None, None], mode='w'): 
    with open(filepath, mode) as fp:
        for it in data: 
            json.dump(it, fp)
            fp.write('\n')

def load_json(filepath: str): 
    if not os.path.exists(filepath): return {} 
    with open(filepath) as fp:
        return json.load(fp)
    
def save_json(filepath: str, data: Dict[Any, Any]): 
    # TODO remove unnecessary codes
    def custom_encoder(obj, level=0):
        if level == 1:
            return json.dumps(obj, indent=2)
        else:
            return json.dumps(obj)
    with open(filepath, 'w') as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False, default=lambda obj: custom_encoder(obj, 1))

def pickle_load(filename: str): 
    with open(filename, 'rb') as fp:  
        return pickle.load(fp)
    
def pickle_dump(content, filename: str): 
    with open(filename, 'wb') as fp: 
        pickle.dump(content, fp)



'''
    将数据分为 total 份, 
    取其中的第 idx 份 (从0开始). 
'''
def data_divide(data: List[Any], idx: int, total: int) -> Generator[Any, None, None]:
    size = len(data)
    rng = range(idx * size // total, (idx + 1) * size // total)  # range
    for i, it in enumerate(data): 
        if i not in rng: continue 
        yield it
