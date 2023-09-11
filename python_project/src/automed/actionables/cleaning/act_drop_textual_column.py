from actionable import *
from automed import Output
from pandas.api.types import is_string_dtype


@isStep('cleaning')
class ActDropTextualColumn(Actionable):
    
    configuration = {
        # 'empty_threshold': {
        #     'description': 'Column with more or equal proportion of empty row will dropped. 1 will drop all columns',
        #     'default': 0.5
        # }
    }
    
    @runner
    def run(self, input) -> Output:
        for column, values in input.dataset.train_data.items():
            if is_string_dtype(values):
                input.dataset.train_data.drop(columns=[column], inplace=True)
                input.dataset.test_data.drop(columns=[column], inplace=True)
        
        return input.to_output(input.dataset, None, None)
        
    
    def priorize(self, input=None):
        return 0 # Last cleaning action