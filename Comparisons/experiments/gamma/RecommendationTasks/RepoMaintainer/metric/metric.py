import os
import json
import numpy as np

def analysis(topks):
    top1 = 0
    top3 = 0
    top5 = 0
    top10 = 0
    mrr = 0

    top1 += (sum(topks[:2]) != 0)
    top3 += (sum(topks[:4]) != 0)
    top5 += (sum(topks[:6]) != 0)
    top10 += (sum(topks[:11]) !=0)

    for i in range(1, len(topks)):
        if topks[i] > topks[i - 1]:
            mrr += 1 / i
            break
    return top1, top3, top5, top10, mrr
"""
## Model RESULT
####################
0.3931034482758621 0.7103448275862069 0.8620689655172413 0.9517241379310345 0.58698669146945
8.951724137931034

0.02857142857142857 0.2 0.3142857142857143 0.6285714285714286 0.19734087341230197
17.771428571428572
####################

## Baseline RESULT
####################
0.4 0.6571428571428571 0.7714285714285715 0.8571428571428571 0.5606416202844774
17.771428571428572
####################
"""

if __name__ == "__main__":
    src_file = "./result/model_result_valid_test.json"
    with open(src_file, "r", encoding="utf-8") as inf:
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
        

    print(top1 / total, top3 / total, top5 / total, top10 / total, MRR / total)
    print(scope_length / total)


    