import os
import json
import numpy as np

'''
    BAD MANNER! 
    这个脚本参考了: 
        RecommendationTasks/SimDeveloper/metric/metric.py 
    有删改. 
'''

from RecommendationTasks.SimDeveloper.metric.metric import analysis
from ..general import load_yaml_cfg


cfg = load_yaml_cfg()['alpha']
task_cfg = cfg['tasks']['sim_developer']


'''
## 在 SimDeveloper 推荐任务上模型的表现

|----------------|-----------|--------|-------|
| Model          | Precision | Recall | F1    |
|----------------|-----------|--------|-------|
| Our Model      | 0.920     | 0.983  | 0.950 | 
| Compared Model | 0.779     | 0.902  | 0.836 |
| Topic Model    | 0.943     | 0.767  | 0.846 |
|----------------|-----------|--------|-------|

|----------------|-------|-------|-------|-------|-------|
| Model          | HR@1  | HR@3  | HR@5  | HR@10 | MRR   |
|----------------|-------|-------|-------|-------|-------|
| Our Model      | 0.434 | 0.705 | 0.801 | 0.886 | 0.591 |
| Compared Model | 0.385 | 0.655 | 0.769 | 0.868 | 0.547 |
| Topic Model    | 0.366 | 0.635 | 0.745 | 0.843 | 0.525 |
|----------------|-------|-------|-------|-------|-------|

Horay! We finished our work about comparing 
our method with another, about the modeling of develoepers' 
technic. 

The result shows that our method is better than the 
compared one at all aspect. 

Good! Let's take a break. 

'''

def main(): 

    src_file = task_cfg['result']['valid_test_result']
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

main()