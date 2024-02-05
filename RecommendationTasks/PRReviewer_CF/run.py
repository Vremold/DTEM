path = "/media/dell/disk/vkx/DTEM"
import sys
sys.path.append(path)

from RecommendationTasks.PRReviewer_CF.model import train_model
from RecommendationTasks.PRReviewer_CF.metric.validate import evaluate
from RecommendationTasks.PRReviewer_CF.metric.metric import metric

if __name__ == "__main__":
    train_model(partial=True)
    evaluate(partial=True)
    metric(partial=True)