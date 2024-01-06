#!/usr/bin/env python3


from RecommendationTasks.ContributionRepo_CF.model import train_model
from RecommendationTasks.ContributionRepo_CF.metric.validate import evaluate 
from RecommendationTasks.ContributionRepo_CF.metric.metric import metric 


evaluate('train')
metric('train')
exit(0)

for i in [0, 5, 10, 20, 40]: 
    train_model(i)
    if i == 0: 
        postfix = 'full'
    else: 
        postfix = f'top{i}'
    evaluate(postfix)
    metric(postfix)

