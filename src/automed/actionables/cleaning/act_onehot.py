"""
[STEP] One hot encoding categorical features
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from ...actionable import Actionable
from ...dataset import Dataset
from ...data_type import DataType
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('cleaning')
class ActOnehot(Actionable):
    """
    [STEP] One hot encoding categorical features
    """
    name="One hot encoding categorical features"
    def __init__(self):
        self.configuration:dict = {}
        self.columns:list[str] = None
        self.encoder:OneHotEncoder = None


    def fit(self, dataset:Dataset) -> Actionable:
        """
        Find columns to encode and fit encoder

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
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
    
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Apply One Hot Encoding to dataframe

        Args:
            x (pd.DataFrame): DataFrame to transform

        Returns:
            pd.DataFrame: Transformed dataset
        """
        # Without Reset index, the join with OHE will create NAN (index mismatch)
        X = X.reset_index(drop=True)
        
        transformed = self.encoder.transform(X[self.columns])
        ohe_df = pd.DataFrame(transformed, columns=self.encoder.get_feature_names_out(self.columns))
        
        X = X.drop(self.columns, axis=1)
        df = X.join(ohe_df)

        return df
        
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5
