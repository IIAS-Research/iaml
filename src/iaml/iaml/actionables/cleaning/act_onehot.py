"""[STEP] One hot encoding categorical features"""

import textwrap
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from ...actionable import Actionable
from ...dataset import Dataset
from ...data_type import DataType
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('cleaning')
class ActOnehot(Actionable):
    """[STEP] One hot encoding categorical features"""

    name = 'One hot encoding categorical features'
    description = 'Encode categorical data to numerical values using One Hot Encoding Algorithm'
    description_long = textwrap.dedent('''\
        Retrieve all unique values from a column, then transform
        those values to multiple binary columns.
        Exemple: If a column "A" contain 3 uniques values like "coffee", "tea" and "water",
        this step will create binary columns "A_coffee", "A_tea" and "A_water".''')

    def __init__(self):
        self.columns: list[str] = None
        self.encoder: OneHotEncoder = None

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = dataset.get_columns_names_by_type(DataType.CATEGORICAL)
        values = dataset.X[self.columns]
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(values)

        # creates a dict with columns as keys and encoded categories as values
        features = dict(zip(self.columns, self.encoder.categories_))

        self.explanations = [
            f'Encoded categorical column **`{c}`** into **{len(v)}** new columns.'
            for c, v in features.items() if len(v) > 0
        ]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply One Hot Encoding to dataframe.

        :param pd.DataFrame x: DataFrame to transform.
        :return: Transformed dataset.
        """
        # Without Reset index, the join with OHE will create NAN (index mismatch)
        X = X.reset_index(drop=True)

        transformed = self.encoder.transform(X[self.columns])
        ohe_df = pd.DataFrame(transformed, columns=self.encoder.get_feature_names_out(self.columns))
        X = X.drop(self.columns, axis=1)
        df = X.join(ohe_df)

        return df

    def suitable(self, dataset: Dataset) -> bool:
        return bool(dataset.get_columns_names_by_type(DataType.CATEGORICAL))

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5
