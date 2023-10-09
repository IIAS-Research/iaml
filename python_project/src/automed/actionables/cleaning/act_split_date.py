from ...actionable import *
from ...automed import Output
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import pandas as pd

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
            x[column + '_weekday'] = x[column].dt.dayofweek
            x[column + '_month'] = x[column].dt.month
            x[column + '_year'] = x[column].dt.year
            
            
            # Hour
            x[column + '_hour'] = x[column].dt.hour
            x[column + '_minute'] = x[column].dt.minute
            x[column + '_second'] = x[column].dt.second
            
            return x, y
            
        for column, values in input.dataset.train_data.items():
            if is_datetime(values):
                input.dataset.apply(transform, column=column)
        
        return input.to_output(input.dataset, None, None)
    
        
    def priorize(self, input=None):
        return 0.5 # medium priority
    