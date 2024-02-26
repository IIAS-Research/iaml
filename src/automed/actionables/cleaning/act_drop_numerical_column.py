from ...actionable import *
from ...data_type import DataType
from ...automed import Output


def transform(x, y, columns: list) -> Output:
    return x.drop(columns, axis=1), y


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
    def run(self, input: Input, callback=None) -> Output:
        columns_to_drop = []     
        for column in input.dataset.get_columns_names_by_type(DataType.NUMERIC):
            values = input.dataset[column]
            if values.isnull().sum()/len(values) >= self.get_config('empty_threshold'):
                columns_to_drop.append(column)

        return input.transform_dataset(transform, columns_to_drop)
        
    
    def priorize(self, input=None):
        return 0 # Last cleaning action
