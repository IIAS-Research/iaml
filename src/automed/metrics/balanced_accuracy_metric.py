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
    
    def suitable(self, X:pd.DataFrame, y:pd.DataFrame, type_of_target:str) -> bool:
        return type_of_target in ['binary', 'multiclass']
    
    def compute(self, y, y_pred):
        return balanced_accuracy_score(y, y_pred)