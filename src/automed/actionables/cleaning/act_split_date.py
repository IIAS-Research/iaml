from ...actionable import *
from ...data_type import DataType
from ...automed import Output
import pandas as pd
import numpy as np

@isStep('cleaning')
class ActSplitDate(Actionable):
    name = "Transform string column to date"
    def __init__(self):
        self.configurations = [{}]
    
    @runner
    def run(self, input, callback=None) -> Output:
        
        def transform(x, y, column):
            # Split !
            
            # Day
            x[column + '_weekday'] = x[column].dt.dayofweek.replace(np.NaN, -1)
            x[column + '_month'] = x[column].dt.month.replace(np.NaN, -1)
            x[column + '_year'] = x[column].dt.year.replace(np.NaN, -1)
            
            
            # Hour
            x[column + '_hour'] = x[column].dt.hour.replace(np.NaN, -1)
            x[column + '_minute'] = x[column].dt.minute.replace(np.NaN, -1)
            x[column + '_second'] = x[column].dt.second.replace(np.NaN, -1)
            
            return x, y
            
        for column in input.dataset.get_columns_names_by_type(DataType.DATE):
            input.dataset.apply(transform, column=column)
        
        return input.to_output(input.dataset, None, None)
    
        
    def priorize(self, input=None):
        return 0.5 # medium priority
    