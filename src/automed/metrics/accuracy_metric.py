from ..step import *
from ..metric import Metric
from collections import Counter
import pandas as pd
from sklearn.metrics import *
 
class AccuracyMetric(Metric):
    
    def __str__(self):
        return 'accuracy'
    
    def explain(self):
        return 'Accuracy classification score.'
    
    # Get one label (numpy.array or pd.series) and return true is classes is balanced
    def __is_balanced(self, y):
        class_count = Counter(y)
        total_samples = y.shape[0]
        ideal_count = total_samples/len(class_count)
        threshold = 0.20 * ideal_count
        return not(any(abs(count - ideal_count) > threshold for count in class_count.values()))
    
    def suitable(self, dataset) -> bool:
        y = dataset.y_train
        return dataset.type_of_target in ['binary', 'multiclass'] and not(self.__is_balanced(y))
        
    def compute(self, y, y_pred):
        return accuracy_score(y, y_pred) 