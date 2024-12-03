"""
[STEP] Drop Rows with a ratio of Empty Columns
"""
from typing import Any
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
    name: str = "Drop Rows with Empty Columns"
    description: str = textwrap.dedent('''\
        This step drops rows where the ratio of empty columns are over a threshold.
        It helps clean the dataset by removing rows with a significant
        amount of missing values.''')
    description_long: str = textwrap.dedent('''\
        In datasets, missing data is a common issue. 
        This step drops rows where the ratio of empty columns are over a threshold.
        This ensures that rows with too many missing values 
        are not included in the analysis, improving the quality of the dataset.''')
    refs: list[dict[str, Any]] = []

    def __init__(self):
        self.configuration = {
            'empty_threshold': {
                'description': "Row with less or equal proportion of empty column will \
                    be drop. 0 will never never drop a row",
                'default': 0.3
            },
            'min_size': {
                'description': "Minimal size of the resulting dataset. If new dataset is smaller \
                    than this value, old one will be restored",
                'default': 20
            }
        }

    def fit(self, dataset: Dataset):  # pylint: disable=unused-argument
        return self

    def __transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the dropping of rows with at least 40% empty columns.
        
        :param pd.DataFrame X: The dataframe to clean.
        :return: The cleaned dataframe.
        """
        threshold = self.get_config('empty_threshold') * X.shape[1]  # 40% of the total columns
        new_x = X.dropna(thresh=X.shape[1] - threshold)
        return new_x if new_x.shape[0] >= self.get_config('min_size') else X

    def resample(self, X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Resample method to apply transformation to both X and y.
        
        :param pd.DataFrame X: Features to transform.
        :param pd.DataFrame y: Labels to keep aligned.
        :return: The transformed X and aligned y.
        """
        x_clean = self.__transform(X.reset_index(drop=True))
        y_aligned = y[x_clean.index]  # Align y with the cleaned X
        x_clean.reset_index(drop=True, inplace=True)

        return x_clean, y_aligned

    def priorize(self, candidate: Candidate = None) -> float:
        return 1

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.X.isnull().values.any()
