from ...actionable import *
from ...automed import Output
from ...data_type import DataType


@isStep('cleaning')
class ActMeanColumn(Actionable):
    name = 'Fill missing values with mean'
    description = 'Fills missing values with the mean of non-missing values when the proportion of empty rows is lower than {empty_threshold}.'

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
        explain = []

        for column in input.dataset.get_columns_names_by_type(DataType.NUMERIC):
            values = input.dataset.train_data[column]
            nan_values_count = values.isnull().sum()

            if nan_values_count > 0 and nan_values_count / len(values) <= self.get_config('empty_threshold'):
                self.columns.append((column, values.mean()))
                explain.append((nan_values_count, len(values), nan_values_count / len(values) * 100))

        input.pipeline.add_explanation(self, [
            f'Filled missing values of column **`{c}`** with **{mean}** because **{v[0]}** out of **{v[1]}** values (**{v[2]}**%) were missing.'
            for (c, mean), v in zip(self.columns, explain)
        ])
        
        return input.transform_dataset(self)
    
    def transform(self, x):
        for name, mean in self.columns:
            x[name].fillna(mean, inplace=True)

        return x
    
    def priorize(self, input=None):
        return 1-(input.dataset.train_data.isnull().sum().min()/len(input.dataset.train_data) ) # TODO -> Do something better. This function have no sense for now. Only an example.
