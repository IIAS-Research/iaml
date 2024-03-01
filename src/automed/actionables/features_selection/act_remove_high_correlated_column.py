from ...actionable import *
from ...automed import Output
import numpy as np

@isStep('features_selection')
class ActRemoveHighCorrelatedColumn(Actionable):
    name = 'Remove High Correlated Column'
    description = 'Remove columns which correlation with other columns is higher than {threshold}.'

    def __init__(self):
        self.configurations = [{
            'threshold': {
                'description': 'If two columns is correlated over this value, only one will be kept',
                'default': 0.9
            }
        }]
    
    @runner
    def run(self, input: Input, callback=None) -> Output:
        # Compute correlation matrix 
        corr_matrix = input.dataset.train_data.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
        
        # Find features with above-threshold correlation
        # self.to_drop = list(to_drop.keys())
        corr = { c: (upper[upper[c] >= self.get_config('threshold')].index) for c in upper.columns }
        self.to_drop = [ c for c, v in corr.items() if len(v) > 0 ]

        input.pipeline.add_explanation(self, [
            f'Dropped column **`{c}`** because it was too correlated with {", ".join([ f"**`{i}`**" for i in corr[c] ])}.'
            for c in self.to_drop
        ])
        
        return input.transform_dataset(self)
    
    
    def transform(self, x):
        return x.drop(self.to_drop, axis=1)

        
    
    def priorize(self, input=None):
        return 0.5 # TODO -> Do something better. This function have no sense for now. Only an example.