"""
[STEP] Fill missing values with mean
"""
import pandas as pd
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step
from ...data_type import DataType


@is_step('cleaning')
class ActMeanColumn(Actionable):
    """
    [STEP] Fill missing values with mean
    """
    name = 'Fill missing values with mean'
    description = """Fills missing values with the mean of non-missing values
        when the proportion of empty rows is lower than {empty_threshold}."""

    def __init__(self):
        self.columns:list[str] = None
        self.configuration:dict = {
            'empty_threshold': {
                'description': "Column with less or equal proportion of empty row will \
                    be fill with mean value. 1 will always fill void values",
                'default': 1 # TODO Review when adding new kind of imputer
            }
        }
    
    def fit(self, dataset:Dataset) -> Actionable:
        # threshold = self.get_config('empty_threshold')

        self.columns = []
        explain = []

        for column in dataset.get_columns_names_by_type(DataType.NUMERIC):
            values = dataset.X[column]
            nan_values_count = values.isnull().sum()

            self.columns.append((column, values.mean()))
            explain.append((
                nan_values_count,
                len(values),
                nan_values_count / len(values) * 100,
            ))

        self.explanations = [
            f"""Filled missing values of column **`{c}`** with **{mean:.2f}**
                (**{v[0]}** out of **{v[1]}** values (**{v[2]:.2f}**%)
                were missing in train data)."""
            for (c, mean), v in zip(self.columns, explain)
        ]
        
        return self
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Fill NA values with mean

        Args:
            x (pd.DataFrame): DataFrame to transform

        Returns:
            pd.DataFrame: Transformed dataset
        """
        for name, mean in self.columns:
            X[name].fillna(mean, inplace=True)
        
        return X
        
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 1-(candidate.dataset.X.isnull().sum().min()/len(candidate.dataset.X))
