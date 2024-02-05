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
model_result_valid_test
#######################
0.4336004579278764 0.705208929593589 0.8005151688609045 0.8863766456783057 0.5905696380372543
49.706926159129935
#######################

0.3331425300515169 0.6087578706353749 0.73039496279336 0.8385804235832857 0.5021602635873925
49.706926159129935
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