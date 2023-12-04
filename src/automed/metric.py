from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

class Metric:
    def explain(self):
        return "Here is an explaination of how this metrics work"
    
    # TODO Remove multilabel parameters and split into two metrics. One Metric = Only behavior
    def compute(self, y, y_pred, multilabel=False):
        if multilabel:
            return accuracy_score(y, y_pred)
        else:
            return balanced_accuracy_score(y, y_pred)