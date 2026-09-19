"""[STEP] Encode categorical features by relative frequency."""
import textwrap
from typing import Any

import pandas as pd

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('cleaning')
class ActFrequencyEncoder(Actionable):
    """[STEP] Encode categorical features by relative frequency."""

    name: str = 'Frequency encoding'
    _usage: str = "Use when categorical features need numeric encoding and frequencies are stable; compare ActDropHighCardinalityCategorical. Applicable to categorical/string features with moderate cardinality. Avoid when categories are very sparse, drifting, or too few rows to estimate."
    _description: str = textwrap.dedent('''\
        Encode categorical columns using their relative frequency.''')
    _description_long: str = textwrap.dedent('''\
        Replace each category with its relative frequency observed in the training data.
        Frequencies are computed on non-missing values and unseen or missing values are
        mapped to {unknown_value}.''')

    def __init__(self) -> None:
        self.columns: list[str] = []
        self.encodings: dict[str, dict[Any, float]] = {}
        self.fallback_values: dict[str, float] = {}
        self.unknown_value: float = 0.0

        self.configuration = {
            'unknown_value': {
                'description': 'Value used for unseen or missing categories.',
                'default': 0.0
            }
        }

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = self._select_columns(dataset)
        self.encodings = {}
        self.fallback_values = {}
        self.explanations = []

        if not self.columns or dataset.X.empty:
            return self

        self.unknown_value = self._coerce_float(self.get_config('unknown_value'), 0.0)

        for column in self.columns:
            series = dataset.X[column]
            counts = series.value_counts(dropna=True)
            total = int(counts.sum())
            if total <= 0 or counts.empty:
                self.fallback_values[column] = self.unknown_value
                continue

            frequencies = (counts / total).to_dict()
            self.encodings[column] = frequencies
            self.fallback_values[column] = self.unknown_value
            self.explanations.append(
                f"Encoded `{column}` using {len(frequencies)} categories."
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.columns:
            return X

        for column in self.columns:
            if column not in X.columns:
                continue

            mapping = self.encodings.get(column, {})
            fallback = self.fallback_values.get(column, self.unknown_value)
            fallback = self._coerce_float(fallback, 0.0)

            encoded = X[column].map(mapping)
            X[column] = encoded.fillna(fallback).astype(float)

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

        total_columns = candidate.dataset.X.shape[1] or 1
        cat_ratio = len(columns) / total_columns

        return min(1.0, max(cat_ratio, avg_cardinality))

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
    def _coerce_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
