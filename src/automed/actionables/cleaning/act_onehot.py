from ...actionable import *
from ...data_type import DataType
from ...automed import Output

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


@isStep('cleaning')
class ActOnehot(Actionable):
    name="One hot encoding categorical features"
    def __init__(self):
        self.configurations = [{}]


    @runner
    def run(self, input: Input, callback=None) -> Output:
        self.columns = input.dataset.get_columns_names_by_type(DataType.CATEGORICAL)
        values = input.dataset.X_train[self.columns]
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(values)

        feature_names = self.encoder.get_feature_names_out()
        
        features = { c: [] for c in self.columns }
        for c, f in map(lambda f: f.split('_'), feature_names):
            features[c].append(f)

        input.pipeline.add_explanation(self, [
            f'Encoded categorical column **`{c}`** into **{len(v)}** new columns.'
            for c, v in features.items() if len(v) > 0
        ])

        return input.transform_dataset(self)
    
    
    def transform(self, x) -> Output:
        # Without Reset index, the join with OHE will create NAN (index mismatch)
        x = x.reset_index(drop=True)
        
        transformed = self.encoder.transform(x[self.columns])
        ohe_df = pd.DataFrame(transformed, columns=self.encoder.get_feature_names_out(self.columns))
        
        x = x.drop(self.columns, axis=1)
        df = x.join(ohe_df)

        return df
        
    
    def priorize(self, input=None):
        return 0.5
