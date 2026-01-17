"""[STEP] Drop duplicate rows"""
import textwrap
from typing import Any
import pandas as pd
from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('features_precleaning')
class ActDropDuplicateRows(Actionable):
    """[STEP] Drop duplicate rows"""

    name: str = 'Drop duplicate rows'
    _usage: str = "Use when duplicated feature rows indicate repeated records and you want one kept copy. Applicable to tabular data with aligned y; unlike ActDropIdLikeColumns, it drops rows not columns. Avoid when repeats are meaningful (time series, sampling) or counts must be preserved."
    _description: str = textwrap.dedent('''\
        Remove duplicated rows using keep={keep}.''')
    _description_long: str = textwrap.dedent('''\
        Duplicated rows can bias model training by repeating the same signal.
        This step removes duplicates based on feature values and keeps y/groups aligned.''')
    refs: list[dict[str, Any]] = []

    def __init__(self):
        self.configuration = {
            'keep': {
                'description': "Which duplicate to keep when removing duplicates.",
                'default': 'first',
                'categorical': ['first', 'last', False]
            }
        }
        self.columns_to_check: list[str] = []
        self.duplicate_count: int = 0
        self.duplicate_ratio: float = 0.0

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns_to_check = self.__candidate_columns(dataset)
        if dataset.X.empty or not self.columns_to_check:
            self.duplicate_count = 0
            self.duplicate_ratio = 0.0
            self.explanations = []
            return self

        duplicate_mask = dataset.X.duplicated(
            subset=self.columns_to_check,
            keep=self.get_config('keep')
        )
        self.duplicate_count = int(duplicate_mask.sum())
        total_rows = len(dataset.X)
        self.duplicate_ratio = self.duplicate_count / max(1, total_rows)

        if self.duplicate_count:
            self.explanations = [
                f'Dropped {self.duplicate_count} duplicated rows '
                f'({self.duplicate_ratio:.2%} of {total_rows}) using keep={self.get_config("keep")}.'
            ]
        else:
            self.explanations = []

        return self

    def resample(self, X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Drop duplicate rows and align y.

        :param pd.DataFrame X: Features to clean.
        :param pd.DataFrame y: Labels to align.
        :return: Cleaned X and aligned y.
        """
        if X.empty:
            return X, y

        x_reset = X.reset_index(drop=True)
        x_clean = self.__transform(x_reset)
        if len(x_clean) == len(x_reset):
            return X, y

        if not self.__has_aligned_y(y, len(x_reset)):
            x_clean.reset_index(drop=True, inplace=True)
            return x_clean, y

        y_aligned = y[x_clean.index]
        x_clean.reset_index(drop=True, inplace=True)
        return x_clean, y_aligned

    def priorize(self, candidate: Candidate = None) -> float:
        if candidate is None:
            return 1.0
        columns = self.__candidate_columns(candidate.dataset)
        if not columns or candidate.dataset.X.empty:
            return 0.0
        duplicate_ratio = candidate.dataset.X.duplicated(
            subset=columns,
            keep=self.get_config('keep')
        ).mean()
        return min(1.5, 0.5 + duplicate_ratio)

    def suitable(self, dataset: Dataset) -> bool:
        columns = self.__candidate_columns(dataset)
        if not columns or dataset.X.empty:
            return False
        return dataset.X.duplicated(
            subset=columns,
            keep=self.get_config('keep')
        ).any()

    def __transform(self, X: pd.DataFrame) -> pd.DataFrame:
        subset = self.__subset_columns(X)
        if not subset:
            return X
        return X.drop_duplicates(subset=subset, keep=self.get_config('keep'))

    def __subset_columns(self, X: pd.DataFrame) -> list[str]:
        if self.columns_to_check:
            return [column for column in self.columns_to_check if column in X.columns]
        return list(X.columns)

    def __candidate_columns(self, dataset: Dataset) -> list[str]:
        columns = dataset.get_columns_names_by_type(list(DataType))
        if len(columns) != dataset.X.shape[1]:
            missing = [column for column in dataset.X.columns if column not in columns]
            columns.extend(missing)
        return columns

    def __has_aligned_y(self, y: Any, rows_count: int) -> bool:
        if y is None:
            return False
        try:
            return len(y) == rows_count
        except TypeError:
            return False
