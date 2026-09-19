"""[STEP] Ordinal encoding categorical features."""
import textwrap
from typing import Any

import pandas as pd

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('cleaning')
class ActOrdinalEncoder(Actionable):
    """[STEP] Ordinal encoding categorical features."""

    name: str = 'Ordinal encoding'
    _description: str = textwrap.dedent('''\
        Encode categorical columns into integer codes with unknown handling.''')
    _description_long: str = textwrap.dedent('''\
        Replace categorical values with integer codes. Categories observed during training
        are assigned consecutive integers starting at {start_value}. Missing or unseen
        categories are encoded as {unknown_value}.''')
    _usage: str = "Use when you need numeric codes for categorical features; ActCountVectorizer is for text-like categories. Applicable to low-cardinality categorical columns with stable labels. Avoid when categories are high-cardinality or should be dropped (ActDropHighCardinalityCategorical)."

    def __init__(self) -> None:
        self.columns: list[str] = []
        self.categories: dict[str, list[Any]] = {}
        self.unknown_value: int = -1
        self.start_value: int = 0
        self.sort_categories: bool = True

        self.configuration = {
            'unknown_value': {
                'description': 'Value used for unseen or missing categories.',
                'default': -1
            },
            'start_value': {
                'description': 'Starting integer assigned to known categories.',
                'default': 0
            },
            'sort_categories': {
                'description': 'Sort categories before encoding for deterministic mapping.',
                'default': True
            }
        }

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = self._select_columns(dataset)
        self.categories = {}
        self.explanations = []

        if not self.columns or dataset.X.empty:
            return self

        self.unknown_value = self._coerce_int(self.get_config('unknown_value'), -1)
        self.start_value = self._coerce_int(self.get_config('start_value'), 0)
        self.sort_categories = bool(self.get_config('sort_categories'))

        for column in self.columns:
            series = dataset.X[column]
            categories = self._extract_categories(series, self.sort_categories)
            self.categories[column] = categories

            if categories:
                self.explanations.append(
                    f"Ordinal-encoded `{column}` with {len(categories)} categories."
                )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.columns or not self.categories:
            return X

        for column in self.columns:
            if column not in X.columns:
                continue
            categories = self.categories.get(column, [])
            X[column] = self._encode_series(
                X[column],
                categories,
                self.start_value,
                self.unknown_value
            )

        return X

    def suitable(self, dataset: Dataset) -> bool:
        columns = self._select_columns(dataset)
        if not columns or dataset.X.empty:
            return False
        return True

    def priorize(self, candidate: Candidate = None) -> float:
        if candidate is None or candidate.dataset.X.empty:
            return 0.0

        columns = self._select_columns(candidate.dataset)
        if not columns:
            return 0.0

        total_rows = len(candidate.dataset.X)
        if total_rows <= 0:
            return 0.0

        unique_counts = candidate.dataset.X[columns].nunique(dropna=True)
        avg_cardinality = float((unique_counts / total_rows).mean())
        low_cardinality = 1.0 - min(1.0, max(0.0, avg_cardinality))

        total_columns = candidate.dataset.X.shape[1] or 1
        cat_ratio = len(columns) / total_columns

        return min(1.0, max(cat_ratio, low_cardinality))

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

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _extract_categories(cls, series: pd.Series, sort_categories: bool) -> list[Any]:
        if pd.api.types.is_categorical_dtype(series):
            categories = list(series.cat.categories)
        else:
            categories = series.dropna().unique().tolist()

        if sort_categories and categories:
            categories = cls._safe_sorted(categories)

        return categories

    @staticmethod
    def _safe_sorted(values: list[Any]) -> list[Any]:
        try:
            return sorted(values)
        except TypeError:
            return sorted(values, key=lambda item: str(item))

    @staticmethod
    def _encode_series(
        series: pd.Series,
        categories: list[Any],
        start_value: int,
        unknown_value: int
    ) -> pd.Series:
        if not categories:
            return pd.Series(unknown_value, index=series.index, dtype='int64')

        cat = pd.Categorical(series, categories=categories)
        codes = pd.Series(cat.codes, index=series.index)
        unknown_mask = codes.eq(-1)

        if start_value:
            codes = codes + start_value

        if unknown_mask.any():
            codes = codes.astype('int64', copy=False)
            codes.loc[unknown_mask] = int(unknown_value)
        else:
            codes = codes.astype('int64', copy=False)

        return codes
