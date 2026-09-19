"""[STEP] Add missing value indicator columns."""
import textwrap

import pandas as pd

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('cleaning')
class ActMissingIndicator(Actionable):
    """[STEP] Add missing value indicator columns."""

    name: str = 'Add missing indicators'
    _description: str = textwrap.dedent('''\
        Add binary "{suffix}" indicator columns to flag missing values.''')
    _description_long: str = textwrap.dedent('''\
        For each selected column, create a companion column named
        "<column>{suffix}" that contains 1 when the value is missing and 0 otherwise.
        The "features" option controls whether indicators are added for all columns
        or only for columns that contain missing values (current: {features}).''')
    _usage: str = "Use when missingness may be predictive and you want to keep columns rather than ActDropNumericalColumn or ActDropCategoricalColumn. Applicable to numerical or categorical columns with NaNs. Avoid when missingness is negligible or you plan to drop columns instead."

    def __init__(self) -> None:
        self.configuration = {
            'suffix': {
                'description': 'Suffix appended to indicator columns.',
                'default': '_is_missing'
            },
            'features': {
                'description': textwrap.dedent('''\
                    Create indicators for all columns or only those with missing values.'''),
                'default': 'all',
                'categorical': ['all', 'missing-only']
            }
        }
        self.columns: list[str] = []
        self.indicator_columns: dict[str, str] = {}

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = self._select_columns(dataset)
        self.indicator_columns = {}
        self.explanations = []

        if not self.columns or dataset.X.empty:
            return self

        missing_counts = dataset.X[self.columns].isna().sum()
        feature_mode = self.get_config('features')
        suffix = self.get_config('suffix')
        if suffix is None:
            suffix = '_is_missing'
        suffix = str(suffix)

        if feature_mode == 'missing-only':
            selected = missing_counts[missing_counts > 0].index.tolist()
        else:
            selected = self.columns

        if not selected:
            return self

        reserved = set(dataset.X.columns)
        for column in selected:
            indicator_name = self._unique_name(column, suffix, reserved)
            reserved.add(indicator_name)
            self.indicator_columns[column] = indicator_name

            missing = int(missing_counts.get(column, 0))
            if missing > 0:
                self.explanations.append(
                    f"Added missing indicator `{indicator_name}` for `{column}` "
                    f"({missing} missing)."
                )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.indicator_columns:
            return X

        present_columns = [column for column in self.indicator_columns if column in X.columns]
        if not present_columns:
            return X

        mapping = {column: self.indicator_columns[column] for column in present_columns}
        indicators = X[present_columns].isna().rename(columns=mapping).astype('int8')
        X[indicators.columns] = indicators

        return X

    def suitable(self, dataset: Dataset) -> bool:
        columns = self._select_columns(dataset)
        if not columns or dataset.X.empty:
            return False
        return bool(dataset.X[columns].isna().any().any())

    def priorize(self, candidate: Candidate = None) -> float:
        if candidate is None or candidate.dataset.X.empty:
            return 0.0
        missing = candidate.dataset.X.isna().sum().sum()
        total = candidate.dataset.X.size or 1
        return min(1.0, missing / total)

    def _select_columns(self, dataset: Dataset) -> list[str]:
        columns = dataset.get_columns_names_by_type(list(DataType))
        if len(columns) != dataset.X.shape[1]:
            missing = [column for column in dataset.X.columns if column not in columns]
            columns.extend(missing)
        return columns

    def _unique_name(self, column: str, suffix: str, reserved: set[str]) -> str:
        base = f"{column}{suffix}"
        if base not in reserved:
            return base
        idx = 1
        candidate = f"{base}_{idx}"
        while candidate in reserved:
            idx += 1
            candidate = f"{base}_{idx}"
        return candidate
