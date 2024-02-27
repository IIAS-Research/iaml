from ...actionable import *
from ...automed import Output
import numpy as np

@isStep('features_selection')
class ActRemoveHighCorrelatedColumn(Actionable):
    name = "Remove High Correlated Column"
    def __init__(self):
        self.configurations = [{
            'threshold': {
                'description': 'If two columns is correlated over this value, only one will be kept',
                'default': 0.9
            }
        }]
    
    @runner
    def run(self, input, callback=None) -> Output:
        # Compute correlation matrix 
        corr_matrix = input.dataset.train_data.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
        
        # Find features with above-threshold correlation
        self.to_drop = [column for column in upper.columns if any(upper[column] >= self.get_config('threshold'))]
        
        return input.transform_dataset(self)
    
    
    def transform(self, x, y):
        return x.drop(self.to_drop, axis=1), y

        
    
    def priorize(self, input=None):
        return 0.5 # TODO -> Do something better. This function have no sense for now. Only an example.