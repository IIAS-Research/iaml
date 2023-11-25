from ...actionable import *
from ...data_type import DataType
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
        
        def transform(x, y, column):
            x = x.drop(columns=[column])
            return x, y
        
        for column in input.dataset.get_columns_names_by_type(DataType.NUMERIC):
            values = input.dataset.X_train[column]
            if values.isnull().sum()/len(values) >= self.get_config('empty_threshold'):
                input.dataset.apply(transform, column=column)
        
        return input.to_output(input.dataset, None, None)
        
    
    def priorize(self, input=None):
        return 0 # Last cleaning action