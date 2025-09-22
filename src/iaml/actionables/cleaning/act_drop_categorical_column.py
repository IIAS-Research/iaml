import textwrap
import pandas as pd
from ...actionable import Actionable
from ...dataset import Dataset
from ...data_type import DataType
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('cleaning', 'baseline_cleaning')
class ActDropCategoricalColumn(Actionable):
    """
    [STEP] Drop Categorical Column
    """
    name = 'Remove categorical columns'
    _description = 'Remove all columns containing categorical data from the dataset'
    _description_long = textwrap.dedent('''\
        Remove all columns containing categorical data from the dataset.
        This step is used to clean the dataset in order to perform other actions later on
        that can't be applied to categorical columns.''')
    can_be_disabled = False
    
    def __init__(self):
        self.columns_to_drop:list[str] = None
    
    def fit(self, dataset:Dataset) -> Actionable:
        """
        Find columns to drop

        Args:
            dataset (Dataset): Fit data

        Returns:
            Actionable: Transformed candidate
        """
        self.columns_to_drop = list(set(
            dataset.get_columns_names_by_type([DataType.CATEGORICAL]) + \
            list(dataset.X.select_dtypes(include=['category']).columns)
        ))

        self.explanations = [
            f'Dropped column **`{c}`**.' for c in self.columns_to_drop
        ]

        return self    

    def transform(self, X) -> pd.DataFrame:
        """
        Drop columns.

        Args:
            X (pd.DataFrame): DataFrame to transform

        Returns:
            pd.DataFrame: Transformed dataset
        """
        return X.drop(self.columns_to_drop, axis=1)
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize itself

        Return : continuous between 0 and 1
        """
        return 0  # Last cleaning action
    
    def suitable(self, dataset:Dataset) -> bool:
        return bool(dataset.get_columns_names_by_type([DataType.CATEGORICAL]))    
