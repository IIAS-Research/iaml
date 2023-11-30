from ...actionable import *
from ...data_type import DataType
from ...automed import Output

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def transform(input: Input, encoder: OneHotEncoder, columns: list[str]) -> Output:
    def t(x, y):
        transformed = encoder.transform(x[columns])
        ohe_df = pd.DataFrame(transformed, columns=encoder.get_feature_names_out(columns))

        x = x.drop(columns, axis=1)
        df = pd.concat([x, ohe_df], axis=1)

        return df, y


    input.dataset.apply(t)

    return input.to_output()


@isStep('cleaning')
class ActOnehot(Actionable):
    name="One hot encoding categorical features"
    def __init__(self):
        self.configurations = [{}]


    @runner
    def run(self, input: Input, callback=None) -> Output:
        columns = input.dataset.get_columns_names_by_type(DataType.CATEGORICAL)
        values = input.dataset.X_train[columns]
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(values)
        
        transform(input, encoder, columns)
        
        return input.to_output(input.dataset, None, transform, encoder, columns)
        
    
    def priorize(self, input=None):
        return 0.5
