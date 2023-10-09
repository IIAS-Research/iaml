from ...actionable import *
from ...automed import Output
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import pandas as pd
import numpy as np

@isStep('cleaning')
class ActStringToDate(Actionable):
    name = "Transform string column to date"
    def __init__(self):
        self.configurations = [{
            'threshold_ratio': {
                'description': 'Threshold on the ratio : successful cast / total row (without null values). Columns over the threshold will be cast to date',
                'default': 0.9
            }
        }]
    
    @runner
    def run(self, input, callback=None) -> Output:
        date_columns = []
        
        def transform(x, y, column, values):
            x[column] = values
            
            return x, y
        
        def remove_null_row(x, y, columns):
            # Delete row with null value. TODO Find a better method
            notnull = x[columns[0]].notnull()
            for column in columns[1:]:
                notnull = np.logical_and(notnull, x[column].notnull())
                
            print("<<>>", column)
            print(">", x.shape, y.shape)
            x = x[notnull]
            y = y[notnull]
            print("<", x.shape, y.shape)
            
            return x, y
            
        for column, values in input.dataset.train_data.items():
            if values.dtype == object:
                date_col = self.__values_to_date(values)
                print(date_col, type(date_col), None)
                if date_col is not None:
                    date_columns.append(column)
                    input.dataset.apply(transform, column=column, values=date_col)
        
        if any(date_columns):
            input.dataset.apply(remove_null_row, columns=date_columns)
        
        return input.to_output(input.dataset, None, None)
    
        
    def priorize(self, input=None):
        return 1 # High priority
    
    def __values_to_date(self, values):
        if values.dtype == 'object':
            try:
                # transformed = pd.to_datetime(values, errors='coerce')
                transformed = values.apply(pd.to_datetime, errors='coerce')
                if (transformed.notnull().sum() / len(values)) >= self.get_config('threshold_ratio'):
                    return transformed
                else:
                    return None
            except ValueError:
                return None