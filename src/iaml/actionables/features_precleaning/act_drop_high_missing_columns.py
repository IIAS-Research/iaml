"""[STEP] Drop Columns with High Missing Values"""
import textwrap
from typing import Any
import pandas as pd
from ...actionable import Actionable
from ...dataset import Dataset
from ...data_type import DataType
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('features_precleaning')
class ActDropHighMissingColumns(Actionable):
    """[STEP] Drop Columns with High Missing Values"""

    name: str = 'Drop columns with high missing values'
    _description: str = textwrap.dedent('''\
        Drop columns where the missing ratio is above {missing_threshold:.0%}.''')
    _description_long: str = textwrap.dedent('''\
        In datasets, some columns can be mostly empty.
        This step removes columns whose missing-value ratio exceeds a configured threshold,
        keeping the dataset focused on informative features.''')
    _usage: str = 'Use when many features are mostly missing and should be removed rather than imputed. Applicable to tabular data with real NaN or after ActSentinelToNaN reveals missingness. Avoid when sparsity is meaningful or columns are ID-like, consider ActDropIdLikeColumns.'
    refs: list[dict[str, Any]] = []

    def __init__(self):
        self.configuration = {
            'missing_threshold': {
                'description': textwrap.dedent('''\
                    Column with a missing ratio greater than or equal to this value will be
                    dropped. 1 will drop only columns that are fully missing.'''),
                'default': 0.5,
                'range': [0.0, 1.0]
            }
        }
        self.columns_to_drop: list[str] = []

    def fit(self, dataset: Dataset) -> Actionable:
        columns = self.__candidate_columns(dataset)
        if not columns:
            self.columns_to_drop = []
            self.explanations = []
            return self

        missing_ratio = dataset.X[columns].isna().mean()
        threshold = self.get_config('missing_threshold')
        self.columns_to_drop = missing_ratio[missing_ratio >= threshold].index.tolist()

        self.explanations = [
            f'Dropped column **`{column}`** because missing ratio is '
            f'**{missing_ratio[column]:.2%}** (threshold {threshold:.2%}).'
            for column in self.columns_to_drop
        ]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop columns.

        :param pd.DataFrame X: DataFrame to transform.
        :return: Transformed DataFrame.
        """
        if not self.columns_to_drop:
            return X

        drop_columns = [column for column in self.columns_to_drop if column in X.columns]
        if not drop_columns:
            return X
        return X.drop(columns=drop_columns)

    def priorize(self, candidate: Candidate = None) -> float:
        if candidate is None:
            return 1.0
        columns = self.__candidate_columns(candidate.dataset)
        if not columns:
            return 0.0
        missing_ratio = candidate.dataset.X[columns].isna().mean()
        threshold = self.get_config('missing_threshold')
        ratio = (missing_ratio >= threshold).sum() / max(1, len(columns))
        return min(1.5, 0.5 + ratio)

    def suitable(self, dataset: Dataset) -> bool:
        columns = self.__candidate_columns(dataset)
        if not columns:
            return False
        missing_ratio = dataset.X[columns].isna().mean()
        return (missing_ratio >= self.get_config('missing_threshold')).any()

    def __candidate_columns(self, dataset: Dataset) -> list[str]:
        columns = dataset.get_columns_names_by_type(list(DataType))
        if len(columns) != dataset.X.shape[1]:
            missing = [column for column in dataset.X.columns if column not in columns]
            columns.extend(missing)
        return columns
