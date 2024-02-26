from ...actionable import *
from ...data_type import DataType
from ...automed import Output

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def transform(x, y, encoder: OneHotEncoder, columns: list[str]) -> Output:
    transformed = encoder.transform(x[columns])
    ohe_df = pd.DataFrame(transformed, columns=encoder.get_feature_names_out(columns))

    x = x.drop(columns, axis=1)
    df = pd.concat([x, ohe_df], axis=1)

    return df, y


@isStep('cleaning')
class ActOnehot(Actionable):
    name="One hot encoding categorical features"
    def __init__(self):
        self.configurations = [{}]


    @runner
    def run(self, input: Input, callback=None) -> Output:
        columns = input.dataset.get_columns_names_by_type(DataType.CATEGORICAL)
        values = input.dataset[columns]
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(values)

        return input.transform_dataset(transform, encoder, columns)
        
    
    def priorize(self, input=None):
        return 0.5
