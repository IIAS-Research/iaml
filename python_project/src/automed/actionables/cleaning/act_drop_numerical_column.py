from ...actionable import *
from ...automed import Output

@isStep('cleaning')
class ActDropNumericalColumn(Actionable):
    name = "Drop Numerical Column"
    
    def __init__(self):
        self.configurations = [{
            'empty_threshold': {
                'description': 'Column with more or equal proportion of empty row will dropped. 1 will drop all columns',
                'default': 0.5
            }
        }]
    
    @runner
    def run(self, input, callback=None) -> Output:
        for column, values in input.dataset.train_data.items():
            if values.isnull().sum()/len(values) >= self.get_config('empty_threshold'):
                input.dataset.train_data.drop(columns=[column], inplace=True)
                input.dataset.test_data.drop(columns=[column], inplace=True)
        
        return input.to_output(input.dataset, None, None)
        
    
    def priorize(self, input=None):
        return 0 # Last cleaning action