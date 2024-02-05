#!/usr/bin/env python3 

from config import \
    CF_DICT_FILE_PATH, \
    load_data, load_repo_maintainers, load_partial_repo_maintainers

import pickle
import math
import random
import sys, os
from typing import Dict, List, Optional

from tqdm import tqdm


'''
    如果你想训练协同过滤模型, 下面是你可以参考的代码(写入项目根目录下的某文件中并运行): 

    ```python
    from RecommendationTasks.ContributionRepo_CF.model import train_model
    from RecommendationTasks.ContributionRepo_CF.metric.validate import evaluate 
    from RecommendationTasks.ContributionRepo_CF.metric.metric import metric 

    train_model(20)   # 协同过滤(CF)中, 选择最接近的20个用户, 训练得到模型文件
    evaluate('top20') # 运行模型文件, 在测试集上生成一个结果的中间文件; 
    metric('top20')   # 根据中间文件的结果, 统计出整个集合上的效果 (stdout输出)
    ```

    see also: 
        train_model()

'''

class CollaborativeFiltering: 

    def __init__(self):
        self.repo_sim_matrix: Dict[int, Dict[int, float]] = {}
        self.repo_maintainers: Dict[int, List[int]] = {}

    def generate(self, top_count=0, partial=False): 

        if partial: 
            data = load_partial_repo_maintainers()  
        else: 
            data = load_repo_maintainers()

        maintainer_repos: Dict[int, List[int]] = {}

        for repo_id, maintainer_id in data:
            maintainer_repos.setdefault(maintainer_id, [])
            maintainer_repos[maintainer_id].append(repo_id)

        repo_sim_matrix: Dict[int, Dict[int, int]] = {}
        repo_maintainers: Dict[int, List[int]] = {}

        for maintainer, repos in maintainer_repos.items():

            for it in repos: 
                repo_maintainers.setdefault(it, [])
                repo_maintainers[it].append(maintainer)

            size = len(repos)
            for i in range(size):
                for j in range(i + 1, size): 
                    u, v = repos[i], repos[j]

                    repo_sim_matrix.setdefault(u, {})
                    repo_sim_matrix.setdefault(v, {})
                    repo_sim_matrix[u].setdefault(v, 0)
                    repo_sim_matrix[v].setdefault(u, 0)

                    repo_sim_matrix[u][v] += 1
                    repo_sim_matrix[v][u] += 1

        print("The number of repos is " + str(len(set(it[1] for it in data))))
        print("The number of maintainers is " + str(len(set(it[0] for it in data))))

        # <repo count>      = len(set(it[1] for it in data)) = 22655
        # <reviewer count>  = len(set(it[0] for it in data)) = 95532

        """
            ```python
            stat = {}
            for _, v in user_sim_matrix.items(): 
                size = len(v)
                stat.setdefault(size, 0)
                stat[size] += 1
            stat = {k: stat[k] for k in sorted(stat)}
            ```

            stat 满足这样的性质: 
            for k, v in stat: 
                k := 和某个用户A共同贡献过同一个仓库的用户的数量
                v := 这样的用户A的个数. 

            结果存放在: stat_codevelopers_count.txt 中. 
        """            

        # 现在开始计算每个用户和其他用户之间的相似度
        print("Calculating sim between all repos.")
        for repo1 in tqdm(repo_sim_matrix): 
            sims: Dict[int, float] = { 
                repo2: repo_sim_matrix[repo1][repo2] / \
                        math.sqrt(len(repo_maintainers[repo1]) * len(repo_maintainers[repo2]))
                for repo2 in repo_sim_matrix[repo1]
            }
            sims = {k: sims[k] for k in (
                sorted(sims) if top_count == 0 else sorted(sims)[:top_count]
            )}  
            repo_sim_matrix[repo1] = sims


        # 注意: 这个dict并没有包含数据中的所有仓库和用户. 
        # 如果一个用户做过贡献的所有仓库只有他自己, 那么他就不会出现在这个dict中. 
        self.repo_sim_matrix = repo_sim_matrix
        self.repo_maintainers = repo_maintainers

        print('Finish generation')


    def load_pickle(self, filepath: str): 
        with open(filepath, 'rb') as fp: 
            self.repo_sim_matrix, self.repo_maintainers = pickle.load(fp)

    def save_pickle(self, filepath: str): 

        # if os.path.exists(filepath): 
            # choice = input(f'file "{filepath}" exists. Overwrite it? [Y/n] ')
            # if choice not in ['Y', 'y']: 
            #     return

        with open(filepath, 'wb') as fp:
            pickle.dump((self.repo_sim_matrix, self.repo_maintainers), fp)

    def recommend(self, repo_id: int, search_scope: Optional[List[int]]=None) -> List[int]: 

        recs: List[int]  # recs: recommendations, 实际被推荐的仓库. 

        if repo_id not in self.repo_sim_matrix: 
            # print("warning: Nothing to recommend.")
            recs = []
        else: 
            # related.keys:   another user's id (idB); 
            # related.values: dev_id 和 idB 共同出现过的仓库的数量
            related: Dict[int, int] = self.repo_sim_matrix[repo_id]  

            # ret.keys:     repo_id
            # ret.values:   rate of the repo
            ret: Dict[int, float] = {}

            for repo_id2 in related: 
                # use cosine-similarity
                weight = related[repo_id2]
                for maintainer_id in self.repo_maintainers[repo_id2]: 
                    ret.setdefault(maintainer_id, 0)
                    ret[maintainer_id] += weight
            
            ret = sorted(ret.items(), key=lambda it: it[1], reverse=True)
            recs = [it[0] for it in ret]  # 按照打分顺序给出推荐的仓库id. 

        if search_scope is None: 
            return ret  # ret 数量可能少于 20 个. 
        print(len(recs))
        others = [it for it in search_scope if it not in recs]
        # random.shuffle(others)  # TODO rethink about this. Is this necessary? 
        return recs + others


def train_model(top_count=0, partial=False):
    '''
        这个函数用于训练模型. 给定了 top_count 的数量, 
        此函数将会在协同过滤中, 选择对应数量最佳相关开发者. 

        top_count = 0 时将使用所有的相关开发者, 此时模型输出文件后缀为 'full'
        其他情况为 f'top{top_count}'. 

        see also: 
            ./metric/validate.py -> evaluate
            ../model.py -> train_model
    '''

    klee = CollaborativeFiltering()
    klee.generate(top_count, partial)
    postfix = 'full' if top_count == 0 else f'top{top_count}'
    if partial: postfix += '.partial'
    klee.save_pickle(CF_DICT_FILE_PATH + '.' + postfix)
