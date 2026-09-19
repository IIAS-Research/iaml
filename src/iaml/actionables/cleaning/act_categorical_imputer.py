"""[STEP] Impute missing categorical values."""
import textwrap
from typing import Any

import pandas as pd

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('cleaning')
class ActCategoricalImputer(Actionable):
    """[STEP] Impute missing categorical values."""

    name: str = 'Impute missing categorical values'
    _usage: str = 'Use when categorical columns have missing labels you want to keep (vs ActDropCategoricalColumn). Applicable to categorical/category dtype features with NA gaps. Avoid when missingness is extreme or cardinality is high; consider ActDropHighCardinalityCategorical.'
    _description: str = textwrap.dedent('''\
        Impute missing categorical values using the {strategy} strategy.''')
    _description_long: str = textwrap.dedent('''\
        Replace missing values in categorical columns with either the most frequent
        observed category or a constant "missing" label. The label can be customized
        via missing_label and is also used as a fallback when no mode can be computed.''')

    def __init__(self) -> None:
        self.columns: list[str] = []
        self.fill_values: dict[str, Any] = {}
        self._missing_stats: dict[str, tuple[int, int, float]] = {}

        self.configuration = {
            'strategy': {
                'description': 'Imputation strategy for categorical columns.',
                'default': 'most_frequent',
                'categorical': ['most_frequent', 'missing']
            },
            'missing_label': {
                'description': 'Label used when strategy="missing" or when no mode exists.',
                'default': 'missing'
            }
        }

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = self._select_columns(dataset)
        self.fill_values = {}
        self._missing_stats = {}
        self.explanations = []

        if not self.columns or dataset.X.empty:
            return self

        X_cat = dataset.X[self.columns]
        total_rows = len(X_cat)
        if total_rows == 0:
            return self

        missing_counts = X_cat.isna().sum()
        strategy = self.get_config('strategy')
        missing_label = self.get_config('missing_label')

        for column in self.columns:
            series = X_cat[column]
            missing = int(missing_counts[column])
            pct = (missing / total_rows * 100.0) if total_rows else 0.0
            self._missing_stats[column] = (missing, total_rows, pct)

            if strategy == 'missing':
                fill_value = missing_label
            else:
                mode_series = series.mode(dropna=True)
                if not mode_series.empty:
                    fill_value = mode_series.iloc[0]
                else:
                    fill_value = missing_label

            self.fill_values[column] = fill_value
            if missing > 0:
                self.explanations.append(
                    f"Filled {missing} missing values in `{column}` with {fill_value!r}."
                )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.columns or not self.fill_values:
            return X

        for column, fill_value in self.fill_values.items():
            if column not in X.columns:
                continue
            if pd.api.types.is_categorical_dtype(X[column]):
                if fill_value not in X[column].cat.categories:
                    X[column] = X[column].cat.add_categories([fill_value])
            X[column] = X[column].fillna(fill_value)

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
        columns = dataset.get_columns_names_by_type(DataType.CATEGORICAL)
        category_columns = list(dataset.X.select_dtypes(include=['category']).columns)
        seen = set()
        ordered = []
        for column in columns + category_columns:
            if column in dataset.X.columns and column not in seen:
                ordered.append(column)
                seen.add(column)
        return ordered
