from ...actionable import *
from ...automed import Output
from ...data_type import DataType


@isStep('cleaning')
class ActMeanColumn(Actionable):
    name = "Fill missing values with mean"
    def __init__(self):
        self.configurations = [{
            'empty_threshold': {
                'description': 'Column with less or equal proportion of empty row will be fill with mean value. 1 will always fill void values',
                'default': 0.5
            }
        }]
    
    @runner
    def run(self, input: Input, callback=None) -> Output:  
        self.columns = []
        for column in input.dataset.get_columns_names_by_type(DataType.NUMERIC):
            values = input.dataset[column]
            if values.isnull().sum()/len(values) <= self.get_config('empty_threshold'):
                self.columns.append((column, values.mean()))
        
        return input.transform_dataset(self)
    
    def transform(self, x):
        for name, mean in self.columns:
            x[name].fillna(mean, inplace=True)

        return x
        
    
    def priorize(self, input=None):
        return 1-(input.dataset.features.isnull().sum().min()/len(input.dataset.features) ) # TODO -> Do something better. This function have no sense for now. Only an example.
