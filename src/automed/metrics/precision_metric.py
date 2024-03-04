from ..step import *
from ..metric import Metric
from collections import Counter
import pandas as pd
from sklearn.metrics import *
from sklearn.utils.multiclass import type_of_target

class precisionMetric(Metric):
    
    def __str__(self):
        return 'precision'
    
    def explain(self):
        return 'Compute the precision: The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.'
    
    def suitable(self, X:pd.DataFrame, y:pd.DataFrame, type_of_target:str) -> bool:
        return type_of_target in ['binary', 'multiclass',  'multilabel-indicator']
        
    def compute(self, y, y_pred):
        if type_of_target(y) == 'binary':
            return precision_score(y, y_pred, pos_label=y[y.columns[0]][0]) # TODO Find something less arbitrary
        elif type_of_target(y) == 'multiclass':
            return precision_score(y, y_pred, average = 'weighted') 
        elif type_of_target(y) == 'multilabel-indicator':
            return precision_score(y, y_pred, average= 'samples')
        else:
            raise Exception('Metric not suitable') 