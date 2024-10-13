"""
[STEP] Drop Rows with a ratio of Empty Columns
"""
import textwrap
import pandas as pd
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('features_precleaning')
class ActDropBadQualityRows(Actionable):
    """
    [STEP] Drop Rows with a ratio of Empty Columns
    """
    name = "Drop Rows with Empty Columns"
    description = textwrap.dedent('''\
        This step drops rows where the ratio of empty columns are over a threshold.
        It helps clean the dataset by removing rows with a significant
        amount of missing values.''')
    description_long = textwrap.dedent('''\
        In datasets, missing data is a common issue. 
        This step drops rows where the ratio of empty columns are over a threshold.
        This ensures that rows with too many missing values 
        are not included in the analysis, improving the quality of the dataset.''')

    refs = []

    def __init__(self):
        self.configuration:dict = {
            'empty_threshold': {
                'description': "Row with less or equal proportion of empty column will \
                    be drop. 0 will never never drop a row",
                'default': 0.3
            }
        }
    
    def fit(self, dataset: Dataset):  # pylint: disable=unused-argument
        """
        Fit method does nothing for dropping rows, but is needed for pipeline compatibility.
        
        Args:
            dataset (Dataset): The dataset to process.

        Returns:
            ActDropRowsWithEmptyColumns: The fitted transformation step.
        """
        return self

    def __transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the dropping of rows with at least 40% empty columns.
        
        Args:
            X (pd.DataFrame): The dataframe to clean.

        Returns:
            pd.DataFrame: The cleaned dataframe.
        """
        threshold = self.get_config('empty_threshold') * X.shape[1]  # 40% of the total columns
        return X.dropna(thresh=X.shape[1] - threshold)

    def resample(self, X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Resample method to apply transformation to both X and y.
        
        Args:
            X (pd.DataFrame): Features to transform.
            y (pd.DataFrame): Labels to keep aligned.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: The transformed X and aligned y.
        """
        X_clean = self.__transform(X.reset_index(drop=True))
        y_aligned = y[X_clean.index]  # Align y with the cleaned X
        X_clean.reset_index(drop=True, inplace=True)
        return X_clean, y_aligned

    def priorize(self, candidate: Candidate = None) -> float:
        """
        Assign priority to this action. Higher means higher priority.
        
        Returns:
            float: Priority score.
        """
        return 1

    def suitable(self, dataset: Dataset) -> bool:
        """
        Checks if this step is suitable for the dataset.
        
        Args:
            dataset (Dataset): The dataset to check suitability for.

        Returns:
            bool: True if the dataset has missing values.
        """
        return dataset.X.isnull().values.any()
