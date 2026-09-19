"""[STEP] Add missing value count feature."""
import textwrap

import pandas as pd

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('cleaning')
class ActMissingCountFeature(Actionable):
    """[STEP] Add missing value count feature."""

    name: str = 'Add missing count feature'
    _usage: str = 'Use when per-row missingness may carry signal; Applicable to datasets with any column types that contain missing values; Avoid when you should impute with ActCategoricalImputer or drop fields with ActDropNumericalColumn.'
    _description: str = textwrap.dedent('''\
        Add a "{feature_name}" column with the number of missing values per row.''')
    _description_long: str = textwrap.dedent('''\
        Count missing values across selected columns and append the count as a single
        numeric feature for each row. This captures the global missingness signal
        that can be useful for downstream models.''')

    def __init__(self) -> None:
        self.configuration = {
            'feature_name': {
                'description': 'Name of the missing count feature.',
                'default': 'missing_count'
            }
        }
        self.columns: list[str] = []
        self.feature_name: str | None = None

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = self._select_columns(dataset)
        self.feature_name = None
        self.explanations = []

        if not self.columns or dataset.X.empty:
            return self

        feature_name = self.get_config('feature_name')
        if not feature_name:
            feature_name = 'missing_count'
        feature_name = str(feature_name)

        reserved = set(dataset.X.columns)
        self.feature_name = self._unique_name(feature_name, reserved)

        missing_counts = dataset.X[self.columns].isna().sum()
        total_missing = int(missing_counts.sum())
        if total_missing > 0:
            total_values = int(dataset.X[self.columns].size)
            missing_columns = int((missing_counts > 0).sum())
            ratio = (total_missing / total_values) if total_values else 0.0
            self.explanations.append(
                f"Added `{self.feature_name}` counting missing values per row "
                f"({total_missing} missing across {missing_columns} columns, "
                f"{ratio:.2%} of values)."
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.columns or not self.feature_name:
            return X

        columns = [column for column in self.columns if column in X.columns]
        if not columns:
            return X

        X[self.feature_name] = X[columns].isna().sum(axis=1)

        return X

    def suitable(self, dataset: Dataset) -> bool:
        columns = self._select_columns(dataset)
        if not columns or dataset.X.empty:
            return False
        return bool(dataset.X[columns].isna().any().any())

    def priorize(self, candidate: Candidate = None) -> float:
        if candidate is None or candidate.dataset.X.empty:
            return 0.0
        columns = self._select_columns(candidate.dataset)
        if not columns:
            return 0.0
        missing = candidate.dataset.X[columns].isna().sum().sum()
        total = candidate.dataset.X[columns].size or 1
        return min(1.0, missing / total)

    def _select_columns(self, dataset: Dataset) -> list[str]:
        columns = dataset.get_columns_names_by_type(list(DataType))
        if len(columns) != dataset.X.shape[1]:
            missing = [column for column in dataset.X.columns if column not in columns]
            columns.extend(missing)
        return columns

    def _unique_name(self, name: str, reserved: set[str]) -> str:
        if name not in reserved:
            return name
        idx = 1
        candidate = f"{name}_{idx}"
        while candidate in reserved:
            idx += 1
            candidate = f"{name}_{idx}"
        return candidate
