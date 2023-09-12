from actionable import *
from automed import Output
import copy

from pandas.api.types import is_string_dtype
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

@isStep('cleaning')
class ActOnehot(Actionable):
    
    configuration = {
        'threshold': {
            'description': 'Threshold on the ratio : unique value / number of rows. Columns under the threshold will be computed as categorical data',
            'default': 0.05
        }
    }
    
    @runner
    def run(self, input) -> Output:
        dataset = copy.deepcopy(input.dataset)
        nb_rows = len(input.dataset.X_train)
        for column, values in input.dataset.X_train.items():
            if is_string_dtype(values.fillna('EMPTY')):
                nb_unique = len(input.dataset.X_train[column].unique())
                if (nb_unique / nb_rows) < self.get_config('threshold'):
                    jobs_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
                    
                    # TRAIN
                    transformed = jobs_encoder.fit_transform(dataset.X_train[column].to_numpy().reshape(-1, 1))
                    ohe_df = pd.DataFrame(transformed, columns=jobs_encoder.get_feature_names([column]))
                    dataset.X_train = pd.concat([dataset.X_train, ohe_df], axis=1).drop([column], axis=1)
                    
                    # TEST
                    transformed = jobs_encoder.transform(dataset.X_test[column].to_numpy().reshape(-1, 1))
                    ohe_df = pd.DataFrame(transformed, columns=jobs_encoder.get_feature_names([column]))
                    dataset.X_test = pd.concat([dataset.X_test, ohe_df], axis=1).drop([column], axis=1)
                
        
        return input.to_output(dataset, None, None)
        
    
    def priorize(self, input=None):
        return 0.5