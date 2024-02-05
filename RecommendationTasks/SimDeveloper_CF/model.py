#!/usr/bin/env python3 

from config import \
    CF_DICT_FILE_PATH, \
    load_data, load_partial_user1_user2

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
        self.user1_sim_matrix: Dict[int, Dict[int, float]] = {}
        self.user1_user2s: Dict[int, List[int]] = {}

    def generate(self, top_count=0, partial=False): 

        if partial: 
            data = load_partial_user1_user2()  
        else: 
            # data = load_repo_maintainers()
            print("Have not implemented overall relation yet")
            exit(0)

        user2_user1s: Dict[int, List[int]] = {}

        for user1, user2 in data:
            # 使用“属于同一组织”的关系是交换的
            user2_user1s.setdefault(user2, [])
            user2_user1s[user2].append(user1)
            user2_user1s.setdefault(user1, [])
            user2_user1s[user1].append(user2)

        user1_sim_matrix: Dict[int, Dict[int, int]] = {}
        user1_user2s: Dict[int, List[int]] = {}

        for user2, user1s in user2_user1s.items():

            for it in user1s: 
                user1_user2s.setdefault(it, [])
                user1_user2s[it].append(user2)

            size = len(user1s)
            for i in range(size):
                for j in range(i + 1, size): 
                    u, v = user1s[i], user1s[j]

                    user1_sim_matrix.setdefault(u, {})
                    user1_sim_matrix.setdefault(v, {})
                    user1_sim_matrix[u].setdefault(v, 0)
                    user1_sim_matrix[v].setdefault(u, 0)

                    user1_sim_matrix[u][v] += 1
                    user1_sim_matrix[v][u] += 1

        print("The number of user1s is " + str(len(set(it[1] for it in data))))
        print("The number of user2s is " + str(len(set(it[0] for it in data))))

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
        for user1 in tqdm(user1_sim_matrix): 
            sims: Dict[int, float] = { 
                user2: user1_sim_matrix[user1][user2] / \
                        math.sqrt(len(user1_user2s[user1]) * len(user1_user2s[user2]))
                for user2 in user1_sim_matrix[user1]
            }
            sims = {k: sims[k] for k in (
                sorted(sims) if top_count == 0 else sorted(sims)[:top_count]
            )}  
            user1_sim_matrix[user1] = sims


        # 注意: 这个dict并没有包含数据中的所有仓库和用户. 
        # 如果一个用户做过贡献的所有仓库只有他自己, 那么他就不会出现在这个dict中. 
        self.user1_sim_matrix = user1_sim_matrix
        self.user1_user2s = user1_user2s

        print('Finish generation')


    def load_pickle(self, filepath: str): 
        with open(filepath, 'rb') as fp: 
            self.user1_sim_matrix, self.user1_user2s = pickle.load(fp)

    def save_pickle(self, filepath: str): 

        # if os.path.exists(filepath): 
            # choice = input(f'file "{filepath}" exists. Overwrite it? [Y/n] ')
            # if choice not in ['Y', 'y']: 
            #     return

        with open(filepath, 'wb') as fp:
            pickle.dump((self.user1_sim_matrix, self.user1_user2s), fp)

    def recommend(self, user1_id: int, search_scope: Optional[List[int]]=None) -> List[int]: 

        recs: List[int]  # recs: recommendations, 实际被推荐的仓库. 

        if user1_id not in self.user1_sim_matrix: 
            # print("warning: Nothing to recommend.")
            recs = []
        else: 
            # related.keys:   another user's id (idB); 
            # related.values: dev_id 和 idB 共同出现过的仓库的数量
            related: Dict[int, int] = self.user1_sim_matrix[user1_id]  

            # ret.keys:     repo_id
            # ret.values:   rate of the repo
            ret: Dict[int, float] = {}

            for user1_id2 in related: 
                # use cosine-similarity
                weight = related[user1_id2]
                for user2_id in self.user1_user2s[user1_id2]: 
                    ret.setdefault(user2_id, 0)
                    ret[user2_id] += weight
            
            ret = sorted(ret.items(), key=lambda it: it[1], reverse=True)
            recs = [it[0] for it in ret]  # 按照打分顺序给出推荐的仓库id. 

        if search_scope is None: 
            return ret  # ret 数量可能少于 20 个. 
        return recs
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
