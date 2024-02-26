from ..step import *
from ..metric import Metric
from collections import Counter
import pandas as pd
from sklearn.metrics import *
import numpy as np
 
class BalancedAccuracyMetric(Metric):
    
    def explain(self):
        return 'Compute the balanced accuracy.'
    
    def __str__(self):
        return 'balanced_accuracy'
    
    def suitable(self, dataset) -> bool:
        return dataset.type_of_target in ['binary', 'multiclass']
    
    def compute(self, y, y_pred):
        return balanced_accuracy_score(y, y_pred)