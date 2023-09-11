from sklearn.metrics import accuracy_score

class Metric:
    def explain(self):
        return "Here is an explaination of how this metrics work"
    
    def compute(self, input):
        y_pred = input.model.predict(input.dataset.X_test)
        return accuracy_score(input.dataset.y_test, y_pred)