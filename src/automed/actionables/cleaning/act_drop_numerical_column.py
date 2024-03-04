from ...actionable import *
from ...data_type import DataType
from ...automed import Output

@is_step('cleaning')
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
    def run(self, input_data: Input, callback=None) -> Output:
        self.columns_to_drop = []     
        for column in input_data.dataset.get_columns_names_by_type(DataType.NUMERIC):
            values = input_data.dataset.X[column]
            if values.isnull().sum()/len(values) >= self.get_config('empty_threshold'):
                self.columns_to_drop.append(column)

        return input_data.add_transform(self)
    
    
    def transform(self, x) -> Output:
        return x.drop(self.columns_to_drop, axis=1)
        
    
    def priorize(self, input_data=None):
        return 0 # Last cleaning action
