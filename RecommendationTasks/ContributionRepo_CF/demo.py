#!/usr/bin/env python

# 莽是不对的. 直接保存成矩阵实在是不应该. 重新来一次吧. 

import json
import random
from typing import List

CONTRIBUTION_FILE_PATH = 'GHCrawler/cleaned/repo_contributions.txt'
REPOSITORY_FILE_PATH = 'GNN/DataPreprocess/full_graph/content/repositories.json'
CONTRIBUTOR_FILE_PATH = 'GNN/DataPreprocess/full_graph/content/contributors.json'

DATASET_VALID_TEST_FILE_PATH = 'RecommendationTasks/ContributionRepo/metric/data/dataset_valid_test.json'

class Evaluation(): 


    """
        params: 
        - repo_idexes: Dict[str, int]. 仓库名和编号的对应关系
        - contributor_idxes: Dict[str, int]. 用户名和编号的对应关系
        - data: List[Tuple[int, Dict]]. 列表元素Tuple中, 第一项是仓库名, 后面的Dict是相关贡献者的名称和贡献数量.
        - ground_truth: TODO ON THE GO
    """

    def __init__(self): 

        with open(REPOSITORY_FILE_PATH) as fp: 
            self.repo_idxes = json.load(fp)

        # len(contributor_idxes) = 394,474
        with open(CONTRIBUTOR_FILE_PATH) as fp: 
            self.contributor_idxes = json.load(fp)

        with open(CONTRIBUTION_FILE_PATH) as fp: 
            def convert(line: str): 
                name, extra = line.split('\t')
                extra = json.loads(extra)
                extra = {it[0]: it[1] for it in extra}
                return name, extra
            self.data = [convert(it) for it in fp.readlines()]

        with open(DATASET_VALID_TEST_FILE_PATH) as fp: 
            ground_truth = json.load(fp)
            ground_truth = {it[0]: it[1] for it in ground_truth}
            self.ground_truth = ground_truth


    # 给定一个仓库的下标，根据协同过滤，
    # 给出这个仓库推荐的用户的下标. 
    # 注意: 这个函数未必给出20个推荐的结果, 因为可能没有可以推荐的内容. 
    # 此时, 根据extend的参数, 我们决定是否要从没有推荐的内容中, 再随机抽选几个, 以达到数量要求.
    def evaluate(self, idx: int, count:int=20, extend=False) -> List[int]: 
        data = self.data
        def distance(repo_a_idx, repo_b_idx): 
            cont_a: dict = data[repo_a_idx][1]
            cont_b: dict = data[repo_b_idx][1]

            return sum([cont_a[it] + cont_b[it] for it in cont_a.keys() & cont_b.keys()])

        lst = [distance(idx, i) for i in range(len(data))]
        tops = [i for it, i in sorted([(it, i) for i, it in enumerate(lst)]) if it != 0][-count-1:-1][::-1]

        if len(tops) != count and extend: 
            scope = [it for it in range(len(data)) if it not in tops]
            tops += random.sample(scope, count - len(tops))

        return tops
    
    def exact(self, idx: int) -> List[int]:
        return self.ground_truth[idx]


if __name__ == '__main__':

    target = 1

    klee = Evaluation()
    print(klee.evaluate(target, extend=True))

