from ..step import *
from ..metric import Metric
from collections import Counter
import pandas as pd
from sklearn.metrics import r2_score

class R2ScoreMetric(Metric):
    
    def explain(self):
        return 'R2(coefficient of determination) regression score function. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).'
    
    def __str__(self):
        return 'r2_score'
        
    def suitable(self, X:pd.DataFrame, y:pd.DataFrame, type_of_target:str) -> bool:
        return type_of_target == 'continuous'

    def compute(self, y, y_pred):
        return r2_score(y, y_pred)