from sklearn.metrics import accuracy_score
# from sklearn.metrics import balanced_accuracy_score

class Metric:
    def explain(self):
        return "Here is an explaination of how this metrics work"
    
    def compute(self, input):
        return input.dataset.compute(input.model, accuracy_score)