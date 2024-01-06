#!/usr/bin/env python3 

from .config import \
    TRAIN_DATA_FILE_PATH, CF_DICT_FILE_PATH, \
    load_data

import pickle
import math
import random
import sys, os
from typing import Dict, List, Optional

printe = lambda text: print(text, file=sys.stedrr)

class CollaborativeFiltering: 

    def __init__(self):
        self.user_sim_matrix: Dict[int, Dict[int, int]] = {}
        user_repos: Dict[int, List[int]] = {}

    def generate_from(self, filepath: str): 

        data = load_data(filepath)  # len(data) = 26834

        repo_users: Dict[int, List[int]] = {}
        for dev_id, repo_id, _ in data:  # dev: developer
            repo_users.setdefault(repo_id, [])
            repo_users[repo_id].append(dev_id)

        user_sim_matrix: Dict[int, Dict[int, int]] = {}
        user_repos: Dict[int, List[int]] = {}

        for repo, users in repo_users.items():

            for it in users: 
                user_repos.setdefault(it, [])
                user_repos[it].append(repo)

            size = len(users)
            for i in range(size):
                for j in range(i + 1, size): 
                    u, v = users[i], users[j]

                    user_sim_matrix.setdefault(u, {})
                    user_sim_matrix.setdefault(v, {})
                    user_sim_matrix[u].setdefault(v, 0)
                    user_sim_matrix[v].setdefault(u, 0)

                    user_sim_matrix[u][v] += 1
                    user_sim_matrix[v][u] += 1

        # <user count> = len(set(it[0] for it in data)) = 22340
        # <repo count> = len(set(it[1] for it in data)) = 19204

        # 这个数据集很小. 所以不做什么 top 20 之类的操作了. 再小就没有了...
        # 注意: 这个dict并没有包含数据中的所有仓库和用户. 
        # 如果一个用户做过贡献的所有仓库只有他自己, 那么他就不会出现在这个dict中. 
        self.user_sim_matrix = user_sim_matrix
        self.user_repos = user_repos

        """
        for k, v in `dict below gen from user_sim_matrix)`: 
            k := 和某个用户A共同贡献过同一个仓库的用户的数量
            v := 这样的用户A的个数. 
        {
            1: 4097,    2: 2172,    3: 1234,    4: 847,
            5: 505,     6: 429,     7: 285,     8: 161,
            9: 132,     10: 90,     11: 75,     12: 57,
            13: 97,     14: 71,     15: 64,     16: 7,
            17: 19,     18: 5,      19: 2,      20: 5,
            21: 23,     23: 19,     24: 1,      27: 23,
            29: 3,      35: 4,      38: 1,      41: 2
        }
        比如说, 
            只和其他1个用户共同贡献过同样仓库的用户的数量是4097个, 
            和其他2个用户则是2172个, 以此类推. 
        """

    def load_pickle(self, filepath: str): 
        with open(filepath, 'rb') as fp: 
            self.user_sim_matrix, self.user_repos = pickle.load(fp)

    def save_pickle(self, filepath: str): 

        if os.path.exists(filepath): 
            choice = input(f'file "{filepath}" exists. Overwrite it? [Y/n] ')
            if choice not in ['Y', 'y']: 
                return

        with open(filepath, 'wb') as fp:
            pickle.dump((self.user_sim_matrix, self.user_repos), fp)

    def recommend(self, dev_id: int, search_scope: Optional[List[int]]=None) -> List[int]: 

        recs: List[int]  # recs: recommendations, 实际被推荐的仓库. 

        if dev_id not in self.user_sim_matrix: 
            printe("warning: Nothing to recommend.")
            recs = []
        else: 
            # related.keys:   another user's id (idB); 
            # related.values: dev_id 和 idB 共同出现过的仓库的数量
            related: Dict[int, int] = self.user_sim_matrix[dev_id]  

            # ret.keys:     repo_id
            # ret.values:   rate of the repo
            ret: Dict[int, float] = {}

            for dev_id2 in related: 
                # use cosine-similarity
                weight = related[dev_id2] / \
                    math.sqrt(len(self.user_repos[dev_id]) * len(self.user_repos[dev_id2]))
                for repo_id in self.user_repos[dev_id2]: 
                    ret.setdefault(repo_id, 0)
                    ret[repo_id] += weight
            
            ret = sorted(ret.items(), key=lambda it: it[1], reverse=True)
            print(ret)
            recs = [it[0] for it in ret]  # 按照打分顺序给出推荐的仓库id. 

        if search_scope is None: 
            return ret  # ret 数量可能少于 20 个. 

        others = [it for it in search_scope if it not in recs]
        # random.shuffle(others)  # TODO rethink about this. Is this necessary? 
        return recs + others

def train_model():
    klee = CollaborativeFiltering()
    klee.generate_from(TRAIN_DATA_FILE_PATH)
    klee.save_pickle(CF_DICT_FILE_PATH)
