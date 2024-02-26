from ..step import *
from ..metric import Metric
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
    
    def suitable(self, dataset) -> bool:
        return dataset.type_of_target in ['binary', 'multiclass', 'multilabel-indicator']
    
    # Calculate the classification error using either accuracy or balanced accuracy, depending on relevence
    def compute(self, y, y_pred):
        balanced_accuracy = BalancedAccuracyMetric().compute(y, y_pred)
        return 1 - balanced_accuracy