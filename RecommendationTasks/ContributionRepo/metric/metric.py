import os
import json
import numpy as np

# top1 ~ top10 是, 前k个里面只要有 GT 中的数据, 就算它是 1 (它只能是0或者1)
# MRR 也是看首次出现GT的k是多少来计算的. 
# ...这个计算方法, 真的是正确的吗? 
# 
# 论文中, 这里想表达的是 HR@k.
# HR@k 定义的各处说法不一, 主要区别在于 GT 的数量. 
# 如果 GT 有不止一个, 参考这里: https://blog.csdn.net/qq_40006058/article/details/89432773
# 如果只有一个, 参考这里: https://zhuanlan.zhihu.com/p/493958358
# 
# 这里实际上GT应该不只有一个, 但用的却是第二个方法. 
# 这里应该是有些问题的. 不过其他实验也是这样做的, 将错就错了. 
def analysis(topks):
    top1  = 0
    top3  = 0
    top5  = 0
    top10 = 0
    mrr   = 0  # MRR: https://blog.csdn.net/jiangjiang_jian/article/details/108246103

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

## Model RESULT 
# model_result_valid_test
####################
0.30952380952380953 0.5952380952380952 0.6904761904761905 0.8095238095238095 0.4769295397584871
217.07142857142858
####################

# baseline_result_valid_test
####################
0.30952380952380953 0.42857142857142855 0.5476190476190477 0.7142857142857143 0.4210182222400268
217.07142857142858
####################

"""

# mdoel_result_valid_test 的结果是符合预期的. 
# 但 baseline_result_valid_test 的结果和我运行出来的不太一样: 
####################
# 0.3333333333333333 0.5238095238095238 0.6428571428571429 0.7380952380952381 0.4624360855453292
# 217.07142857142858
####################


if __name__ == "__main__":
    # src_file = "./result/baseline_result_valid_test.json"
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
