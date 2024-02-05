path = "/media/dell/disk/vkx/DTEM"
import sys
sys.path.append(path)

from RecommendationTasks.RepoMaintainer_CF.model import train_model
from RecommendationTasks.RepoMaintainer_CF.metric.validate import evaluate
from RecommendationTasks.RepoMaintainer_CF.metric.metric import metric

if __name__ == "__main__":
    topk = 0
    partial = True
    model_postfix = 'full'
    if topk > 0:
        model_postfix = 'top' + str(topk)
    
    train_model(top_count=topk, partial=partial)
    evaluate(model_postfix=model_postfix, partial=partial)
    metric(model_prefix=model_postfix, partial=partial)