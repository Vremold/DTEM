import os
import json
import numpy as np


from RecommendationTasks.SimDeveloper_CF.config import \
    VALID_RESULT_PATH


def metric(model_prefix='full', partial=False): 
    '''
        读取 ../valiate.py 生成的文件, 并给出评价指标. 
        指标的部分结果在此文件的其他注释中有统计. 

        see also: 
            ../validate.py -> evaluate()
    '''
    if partial:
        model_prefix += '.partial'
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

