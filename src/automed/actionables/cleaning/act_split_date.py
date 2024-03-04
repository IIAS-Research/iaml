from ...actionable import *
from ...data_type import DataType
from ...automed import Output
import numpy as np

@is_step('cleaning')
class ActSplitDate(Actionable):
    name = "Transform string column to date"
    def __init__(self):
        self.configurations = [{}]
    
    @runner
    def run(self, input_data: Input, callback=None) -> Output:
        self.columns = input_data.dataset.get_columns_names_by_type(DataType.DATE)

        return input_data.add_transform(self)
            
    def transform(self, x) -> Output:
        for column in self.columns:
            # Day
            x[column + '_weekday'] = x[column].dt.dayofweek.replace(np.NaN, -1)
            x[column + '_month'] = x[column].dt.month.replace(np.NaN, -1)
            x[column + '_year'] = x[column].dt.year.replace(np.NaN, -1)

            # Hour
            x[column + '_hour'] = x[column].dt.hour.replace(np.NaN, -1)
            x[column + '_minute'] = x[column].dt.minute.replace(np.NaN, -1)
            x[column + '_second'] = x[column].dt.second.replace(np.NaN, -1) 
            
        return x

    
        
    def priorize(self, input_data=None):
        return 0.5 # medium priority
    