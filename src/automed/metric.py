from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

class Metric:
    def explain(self):
        return "Here is an explaination of how this metrics work"
    
    def compute(self, input):
        if input.dataset.is_multilabel:
            return input.dataset.compute(input.model.sklearn_model, accuracy_score)
        else:
            return input.dataset.compute(input.model.sklearn_model, balanced_accuracy_score)