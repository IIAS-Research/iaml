"""[STEP] Find and drop date column"""
import textwrap
import pandas as pd
from ...actionable import Actionable
from ...data_type import DataType
from ...candidate import Candidate
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('cleaning', 'baseline_cleaning')
class ActDropDateColumn(Actionable):
    """Finds and drops date columns."""

    name: str = 'Remove date columns'
    _description: str = 'Remove all columns containing Date from the dataset'
    _usage: str = 'Use when date columns are irrelevant and you want to remove them; for other types use ActDropNumericalColumn or ActDropCategoricalColumn. Applicable to columns typed as DataType.DATE. Avoid when dates carry predictive signal or need feature extraction.'
    _description_long: str = textwrap.dedent('''\
        Remove all columns containing Data from the dataset
        This step is used to clean the dataset in order to perform other actions later on
        that can't be applied to date columns.''')
    can_be_disabled: bool = False

    def __init__(self):
        self.columns_to_drop: list[str] = None

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns_to_drop = dataset.get_columns_names_by_type(DataType.DATE)

        self.explanations = [
            f'Dropped column **`{c}`**.' for c in self.columns_to_drop
        ]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop columns.

        :param pd.DataFrame X: DataFrame to transform.
        :return: Transformed DataFrame.
        """
        return X.drop(self.columns_to_drop, axis=1)

    def priorize(self, candidate: Candidate = None) -> float:
        return 0

    def suitable(self, dataset: Dataset) -> bool:
        return bool(dataset.get_columns_names_by_type(DataType.DATE))
