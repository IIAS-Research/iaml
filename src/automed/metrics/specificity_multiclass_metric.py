from ..step import *
from ..metric import Metric
from collections import Counter
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

class SpecificityMulticlassMetric(Metric):
    
    def __str__(self):
        return 'specificity'    
    
    def explain(self):
        return 'Calculate specificity in a multiclass conext, where each instance belongs to just one of several classes'
    
    def suitable(self, dataset) -> bool:
        return dataset.type_of_target in ['multiclass']
    
    
    # Specificity is calculated by summing the true negartives and false positives for each class, then using these totals to obtain an overall specificity"
    def compute(self, y, y_pred):
        cm = confusion_matrix(y, y_pred)
        total_tn = 0
        total_fp = 0
        for i in range(len(cm)):
            total_tn += np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            total_fp += np.sum(cm[:, i]) - cm[i, i]
        return total_tn / (total_tn + total_fp)