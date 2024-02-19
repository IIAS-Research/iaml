from ...step import *
from ...metric import Metric
from sklearn.utils.multiclass import type_of_target
from collections import Counter
import pandas as pd
from sklearn.metrics import *
from .accuracy_metric import *
from .balanced_accuracy_metric import *

class ClassificationErrorMetric(Metric):
    
    def explain(self):
        return 'Computes the classification error, if accuracy is relevant then the classification error is calculated by 1 - accuracy otherwise if balanced accuracy is relevant then the classification error is calculated by 1 - balanced _accuracy.'
    
    def __str__(self):
        return 'classification_error'
    
    def _is_imbalanced(self, class_counts, total_samples):
        ideal_count = total_samples / len(class_counts)
        threshold = 0.20 * ideal_count
        return any(abs(count - ideal_count) > threshold for count in class_counts.values())
    
    def is_pertinent(self, y):
        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.iloc[:, 0]
        target_type = type_of_target(y)
        if target_type in ['binary', 'multiclass']:
            class_count = Counter(y)
            total_samples = sum(class_count.values())
            relevence = self._is_imbalanced(class_count, total_samples)
            if relevence :
                return True
            else:
                return False
            
    def suitable(self, y):
        return self.is_pertinent(y)
    
    def compute(self, y, y_pred):
        if AccuracyMetric().suitable(y):
            accuracy = AccuracyMetric().compute(y, y_pred)
            return 1 - accuracy
        elif BalancedAccuracyMetric().suitable(y):
            balanced_accuracy = BalancedAccuracyMetric().compute(y, y_pred)
            return 1 - balanced_accuracy