from ...actionable import *
from ...data_type import DataType
from ...automed import Output

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


@is_step('cleaning')
class ActOnehot(Actionable):
    name="One hot encoding categorical features"
    def __init__(self):
        self.configurations = [{}]


    @runner
    def run(self, input_data: Input, callback=None) -> Output:
        self.columns = input_data.dataset.get_columns_names_by_type(DataType.CATEGORICAL)
        values = input_data.dataset.X[self.columns]
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(values)

        return input_data.add_transform(self)
    
    
    def transform(self, x) -> Output:
        # Without Reset index, the join with OHE will create NAN (index mismatch)
        x = x.reset_index(drop=True)
        
        transformed = self.encoder.transform(x[self.columns])
        ohe_df = pd.DataFrame(transformed, columns=self.encoder.get_feature_names_out(self.columns))
        
        x = x.drop(self.columns, axis=1)
        df = x.join(ohe_df)

        return df
        
    
    def priorize(self, input_data=None):
        return 0.5
