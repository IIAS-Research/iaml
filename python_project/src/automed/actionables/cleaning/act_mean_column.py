from ...actionable import *
from ...automed import Output
from pandas.api.types import is_numeric_dtype

@isStep('cleaning')
class ActMeanColumn(Actionable):
    name = "Fill missing values with mean"
    configuration = {
        'empty_threshold': {
            'description': 'Column with less or equal proportion of empty row will be fill with mean value. 1 will always fill void values',
            'default': 0.5
        }
    }
    
    @runner
    def run(self, input, callback=None) -> Output:
        for column, values in input.dataset.train_data.items():
            if is_numeric_dtype(values) and values.isnull().sum()/len(values) <= self.get_config('empty_threshold'):
                input.dataset.train_data[column].fillna(values.mean(), inplace=True)
                input.dataset.test_data[column].fillna(values.mean(), inplace=True)
        
        return input.to_output(input.dataset, None, None)
    
        
    
    def priorize(self, input=None):
        return 1-(input.dataset.train_data.isnull().sum().min()/len(input.dataset.train_data) ) # TODO -> Do something better. This function have no sense for now. Only an example.