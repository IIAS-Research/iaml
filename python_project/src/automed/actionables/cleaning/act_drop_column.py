from actionable import *

@isStep('cleaning')
class ActDropColumn(Actionable):
    
    configuration = {
        'empty_threshold': {
            'description': 'Column with more or equal proportion of empty row will dropped. 1 will drop all columns',
            'default': 0.5
        }
    }
    
    @runner
    def run(self, dataset):
        for column, values in dataset.items():
            if values.isnull().sum()/len(values) >= self.get_config()['empty_threshold']:
                dataset.drop(columns=[column], inplace=True)
        
        return dataset
        
    
    def evaluate(self, dataset=None):
        return 0 # Last cleaning action