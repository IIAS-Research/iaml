from actionable import *
from pandas.api.types import is_numeric_dtype

@isStep('cleaning')
class ActMeanColumn(Actionable):
    
    configuration = {
        'empty_threshold': {
            'description': 'Column with less or equal proportion of empty row will be fill with mean value. 1 will always fill void values',
            'default': 0.5
        }
    }
    
    @runner
    def run(self, dataset):
        for column, values in dataset.train_data.items():
            if is_numeric_dtype(values) and values.isnull().sum()/len(values) <= self.get_config()['empty_threshold']:
                dataset.train_data[column].fillna(values.mean(), inplace=True)
                dataset.test_data[column].fillna(values.mean(), inplace=True)
        
        return dataset
        
    
    def priorize(self, dataset=None):
        return 1-(dataset.train_data.isnull().sum().min()/len(dataset.train_data) ) # TODO -> Do something better. This function have no sense for now. Only an example.