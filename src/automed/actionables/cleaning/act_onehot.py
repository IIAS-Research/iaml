"""
[STEP] One hot encoding categorical features
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from ...actionable import Actionable
from ...data_type import DataType
from ...automed import Output, Input
from ...step import is_step, runner


@is_step('cleaning')
class ActOnehot(Actionable):
    """
    [STEP] One hot encoding categorical features
    """
    name="One hot encoding categorical features"
    def __init__(self):
        self.configurations:list[dict] = [{}]
        self.columns:list[str] = None
        self.encoder:OneHotEncoder = None


    @runner
    def run(self, input_data: Input, callback:callable=None) -> Output: # pylint: disable=unused-argument
        """
        Find columns to encode and fit encoder

        Args:
            input_data (Input): Fit data
            callback (callable, optional): Call after each step. Defaults to None.

        Returns:
            Output: Transformed input
        """
        self.columns = input_data.dataset.get_columns_names_by_type(DataType.CATEGORICAL)
        values = input_data.dataset.X[self.columns]
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(values)

        return input_data.add_transform(self)
    
    
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
        
    
    def priorize(self, input_data:Input=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5
