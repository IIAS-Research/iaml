from ...actionable import *
from ...automed import Output
from ...data_type import DataType


@isStep('cleaning')
class ActDropNumericalColumn(Actionable):
    name = 'Drop numerical columns'
    description = 'Drop numerical columns where the proportion of empty rows in the dataset is higher than {empty_threshold}.'
    
    def __init__(self):
        self.configurations = [{
            'empty_threshold': {
                'description': 'Column with more or equal proportion of empty row will dropped. 1 will drop all columns',
                'default': 0.5
            }
        }]
    
    @runner
    def run(self, input: Input, callback=None) -> Output:
        self.columns_to_drop = []
        explain = []

        for column in input.dataset.get_columns_names_by_type(DataType.NUMERIC):
            values = input.dataset.X_train[column]
            nan_values_count = values.isnull().sum()

            if nan_values_count / len(values) >= self.get_config('empty_threshold'):
                self.columns_to_drop.append(column)
                explain.append((nan_values_count, len(values), nan_values_count / len(values) * 100))

        input.pipeline.add_explanation(self, [
            f'Dropped column **`{c}`** because **{v[0]}** values out of **{v[1]}** (**{v[2]:.2f}%**) are empty.'
            for c, v in zip(self.columns_to_drop, explain)
        ])

        return input.transform_dataset(self)
    
    def transform(self, x) -> Output:
        return x.drop(self.columns_to_drop, axis=1)
        
    
    def priorize(self, input=None):
        return 0 # Last cleaning action
