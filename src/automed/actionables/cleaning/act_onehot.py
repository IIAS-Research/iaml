from ...actionable import *
from ...data_type import DataType
from ...automed import Output
import copy

from pandas.api.types import is_string_dtype
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

@isStep('cleaning')
class ActOnehot(Actionable):
    name="One hot encoding categorical features"
    def __init__(self):
        self.configurations = [{}]
    
    @runner
    def run(self, input, callback=None) -> Output:
        def transform(x, y, encoder, column):
            transformed = encoder.transform(x[column].to_numpy().reshape(-1, 1))
            ohe_df = pd.DataFrame(transformed, columns=encoder.get_feature_names_out([column]))
            x = pd.concat([x, ohe_df], axis=1).drop([column], axis=1)
            return x, y
            
        for column in input.dataset.get_columns_names_by_type(DataType.CATEGORICAL):
            values = input.dataset.X_train[column]
            jobs_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(values.to_numpy().reshape(-1, 1))
            
            input.dataset.apply(transform, encoder=jobs_encoder, column=column)
        
        return input.to_output(input.dataset, None, None)
        
    
    def priorize(self, input=None):
        return 0.5