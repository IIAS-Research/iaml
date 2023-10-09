from ...actionable import *
from ...automed import Output
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import pandas as pd

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
        
        def transform(x, y, column, values):
            x[column] = values
            # x[column] = pd.to_datetime(values, errors='coerce')
            return x, y
            
        for column, values in input.dataset.train_data.items():
            if values.dtype == object:
                print("====================>")
                print(column)
                date_col = self.__values_to_date(values)
                if type(date_col) != None:
                    input.dataset.apply(transform, column=column, values=date_col)
        
        return input.to_output(input.dataset, None, None)
    
        
    def priorize(self, input=None):
        return 1 # High priority
    
    def __values_to_date(self, values):
        if values.dtype == 'object':
            try:
                # transformed = pd.to_datetime(values, errors='coerce')
                transformed = values.apply(pd.to_datetime, errors='coerce')
                print("trans", transformed.notnull().sum())
                print("base", values.notnull().sum())
                if (transformed.notnull().sum() / values.notnull().sum()) >= self.get_config('threshold_ratio'):
                    return transformed
                else:
                    return None
            except ValueError:
                return None