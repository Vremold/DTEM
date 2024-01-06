import os
import json
import numpy as np

"""
    代码逻辑拷贝自下面的"see also", 有改动. 

    see also: 
        RecommendationTasks/ContributionRepo/metric/metric.py
"""

from RecommendationTasks.ContributionRepo_CF.config import \
    VALID_RESULT_PATH


def metric(model_prefix='full'): 
    '''
        读取 ../valiate.py 生成的文件, 并给出评价指标. 
        指标的部分结果在此文件的其他注释中有统计. 

        see also: 
            ../validate.py -> evaluate()
    '''

    model_filepath = VALID_RESULT_PATH + '.' + model_prefix
    print(f'Metrics of {model_filepath}: ')

    with open(model_filepath, "r", encoding="utf-8") as inf:
        src = json.load(inf)

    total = len(src)
    scope_length = 0

    top1 = 0
    top3 = 0
    top5 = 0
    top10 = 0
    MRR = 0

    for item in src:
        topks = src[item][0]
        tmp1, tmp3, tmp5, tmp10, mrr = analysis(topks)
        top1 += tmp1
        top3 += tmp3
        top5 += tmp5
        top10 += tmp10
        MRR += mrr
        scope_length += len(src[item][1])
        
    print("%.4f | %.4f | %.4f | %.4f | %.4f" % (top1 / total, top3 / total, top5 / total, top10 / total, MRR / total))
    # print(scope_length / total)

def analysis(topks):
    top1  = 0
    top3  = 0
    top5  = 0
    top10 = 0
    mrr   = 0  

    top1  += (sum(topks[:2])  != 0)
    top3  += (sum(topks[:4])  != 0)
    top5  += (sum(topks[:6])  != 0)
    top10 += (sum(topks[:11]) != 0)

    for i in range(1, len(topks)):
        if topks[i] > topks[i - 1]:
            mrr += 1 / i
            break
    return top1, top3, top5, top10, mrr

"""
    |--------------|--------|--------|--------|--------|--------|
    | Method       | top 1  | top 3  | top 5  | top 10 | MRR    |
    |--------------|--------|--------|--------|--------|--------|
    | Our Approach | 0.3095 | 0.5952 | 0.6905 | 0.8095 | 0.4769 |
    |--------------|--------|--------|--------|--------|--------|
    | baseline org | 0.3095 | 0.4286 | 0.5476 | 0.7143 | 0.4210 |
    | baseline new | 0.3333 | 0.5238 | 0.6428 | 0.7381 | 0.4624 |
    |--------------|--------|--------|--------|--------|--------|
    | CF top 5     | 0.5000 | 0.6190 | 0.7381 | 0.8333 | 0.5970 | (ON DELETE)
    | CF top 10    | 0.5714 | 0.7619 | 0.8095 | 0.9048 | 0.6732 | (ON DELETE)
    | CF top 20    | 0.5952 | 0.7381 | 0.8810 | 0.9286 | 0.7040 | (ON DELETE)
    | CF top 40    | 0.6905 | 0.8571 | 0.8810 | 0.9286 | 0.7870 | (ON DELETE)
    | CF FULL      | 0.5952 | 0.8810 | 0.9286 | 0.9762 | 0.7381 | (ON DELETE)
    |--------------|--------|--------|--------|--------|--------|
    | CF PARTIAL   | 0.1905 | 0.3333 | 0.3810 | 0.5000 | 0.2877 |
    |--------------|--------|--------|--------|--------|--------|

    说明: 
    1.  "Our Approach" 是我们使用的方法了, 其实验结果在论文中就是我们提出的方法 (WWW PDF, P8, Table 6, ContributionRepo, Our Method). 
        对应代码里的 RecommendationTasks/ContributionRepo/bin/model.bin 模型的运行效果. 

    2.  "baseline" 两行: 论文中同一个表格, 和 "Our Method" 对比的 "Topic Method". 
        在代码中是和 model.bin 放在一起的 basline.bin. 
        org 是原有指标, 见: RecommendationTasks/ContributionRepo/metric/metric.py
        new 是我重新跑模型得到的指标, 和原来不太一样. 

    3.  CF几行 (除了 CF PARTIAL): 协同过滤方法. 训练数据是 GHCrawler/cleaned/repo_contributions.txt 中的贡献. 
        这个数据集应该是包含了所有的贡献关系. 
        测试效果的时候(validate & metric), 用的是和 1,2 中一样的文件, 是从用户以有的部分数据的一部分拿出来预测. 

        因为数据很大, 所以用developer协同过滤的时候, 只挑选了和目标用户最接近的k个用户的信息来推荐. k = 5,10,...40, 以及不挑选. 

        可以看到, CF 的效果要明显好于 Our Approach, 即使只用 top 5. 

        [DELETING] 
            用到的数据集是不太一样的. 训练 model.bin (实验1 & 2) 用的是 R/ContributionRepo/data/user_watch_repos.json. 
            这个文件是学长构建的, 只有一个用户: 
                1. commit某仓库超过30次;
                2. watch了该仓库; 
            才认为这个用户和这个仓库有关, 收录进数据集. 

            但我们的工作是直接看 commit, 而且是没有按数量过滤的. 没有可比性. 

            实验4 的工作是在同样的数据集上进行的, 才有可比性. 这部分实验结果可以删除了. 
            (但实验4的数据量是真的少啊. 竟然能学到不错的效果, 不容易)
        [/]


    4. CF PARTIAL: 上面的几行训练CF的时候用的是所有的数据, 因此其实"训练"的时候也看到了测试时候的答案. 
       使用训练 model.bin (1) 的数据, 数据量为 26.8k. 

       数据量太低了, 这样得到的CF完全推荐不出来任何东西 (用协同过滤根本找不到可以推荐的仓库, 也就没法"根据推荐对scope中的仓库做排序"). 
       结果其实就是取了 scope 中的前20个. 
       CF 学不出东西, 算不算一个结果? 

    数据集是个大问题. 现在做的是下游任务, 其本身的模型(model.bin)的结构简单, 效果好是因为输入的就是特征向量了. 
    如果只用训练 model.bin 的数据来训练CF, 效果想好是不可能的 (实验4). 

"""
