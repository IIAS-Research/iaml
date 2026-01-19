"""[STEP] Vectorize short text with CountVectorizer."""
import textwrap
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('cleaning')
class ActCountVectorizer(Actionable):
    """[STEP] Vectorize short text with CountVectorizer."""

    name: str = 'Count Vectorizer'
    _usage: str = 'Use when short text is predictive and you want bag-of-words counts instead of ActDropTextualColumn. Applicable to short text columns with a manageable vocabulary. Avoid when text is long, extremely sparse, or you would drop text entirely (ActDropTextualColumn).'
    _description: str = textwrap.dedent('''\
        Vectorize short text columns into bag-of-words counts with n-grams.''')
    _description_long: str = textwrap.dedent('''\
        Build a vocabulary on each short text column and replace it with count features
        for each token or n-gram observed in the training data. Adjust the n-gram range
        and vocabulary size to control sparsity and keep runtime manageable.''')

    def __init__(self) -> None:
        self.columns: list[tuple[str, CountVectorizer]] = []
        self.configuration = {
            'ngram_min': {
                'description': 'Minimum n-gram size to include.',
                'default': 1
            },
            'ngram_max': {
                'description': 'Maximum n-gram size to include.',
                'default': 2
            },
            'max_features': {
                'description': 'Maximum size of the vocabulary (None keeps all).',
                'default': 2000
            },
            'min_df': {
                'description': 'Minimum document frequency for a term to be kept.',
                'default': 1
            },
            'max_df': {
                'description': 'Maximum document frequency for a term to be kept.',
                'default': 1.0
            }
        }

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = []
        self.explanations = []

        columns = dataset.get_columns_names_by_type([DataType.SHORT_TEXT])
        if not columns or dataset.X.empty:
            return self

        params = self._build_vectorizer_params(len(dataset.X))

        for column in columns:
            values = dataset.X[column].fillna('').astype(str)
            vectorizer = CountVectorizer(**params)
            try:
                vectorizer.fit(values)
            except ValueError:
                continue

            feature_names = vectorizer.get_feature_names_out()
            if len(feature_names) == 0:
                continue

            self.columns.append((column, vectorizer))
            self.explanations.append(
                f'Encoded text column **`{column}`** into **{len(feature_names)}** count features.'
            )

        if not self.columns and columns:
            raise RuntimeError(
                "Count vectorizer failed: no usable vocabulary in short text columns."
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply CountVectorizer to short text columns.

        :param pd.DataFrame X: DataFrame to transform.
        :return: Transformed dataset.
        """
        if not self.columns:
            return X

        X = X.reset_index(drop=True)

        for name, vectorizer in self.columns:
            if name not in X.columns:
                continue

            values = X[name].fillna('').astype(str)
            transformed = vectorizer.transform(values)
            feature_names = vectorizer.get_feature_names_out()
            if len(feature_names) == 0:
                X = X.drop([name], axis=1)
                continue

            new_names = [f"{name}_{token}" for token in feature_names]
            vector_df = pd.DataFrame(transformed.toarray(), columns=new_names)

            X = pd.concat([X, vector_df], axis=1).drop([name], axis=1)

        return X

    def suitable(self, dataset: Dataset) -> bool:
        if dataset.X.empty:
            return False
        return bool(dataset.get_columns_names_by_type([DataType.SHORT_TEXT]))

    def priorize(self, candidate: Candidate = None) -> float:
        if candidate is None or candidate.dataset.X.empty:
            return 0.0

        columns = candidate.dataset.get_columns_names_by_type([DataType.SHORT_TEXT])
        if not columns:
            return 0.0

        total_columns = candidate.dataset.X.shape[1] or 1
        return min(1.0, len(columns) / total_columns)

    def _build_vectorizer_params(self, n_rows: int) -> dict[str, Any]:
        ngram_min = self._coerce_int(self.get_config('ngram_min'), 1)
        ngram_max = self._coerce_int(self.get_config('ngram_max'), max(ngram_min, 1))
        ngram_min = max(1, ngram_min)
        ngram_max = max(ngram_min, ngram_max)

        min_df = self._coerce_df(self.get_config('min_df'), 1)
        max_df = self._coerce_df(self.get_config('max_df'), 1.0)

        if n_rows > 0:
            min_df = self._clamp_df(min_df, n_rows)
            max_df = self._clamp_df(max_df, n_rows)
            if self._effective_df(min_df, n_rows) > self._effective_df(max_df, n_rows):
                max_df = min_df

        max_features = self._coerce_optional_int(self.get_config('max_features'))

        params = {
            'ngram_range': (ngram_min, ngram_max),
            'min_df': min_df,
            'max_df': max_df
        }
        if max_features is not None:
            params['max_features'] = max_features

        return params

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return None
        if numeric <= 0:
            return None
        return numeric

    @staticmethod
    def _coerce_df(value: Any, default: float | int) -> float | int:
        if value is None or isinstance(value, bool):
            return default
        if isinstance(value, int):
            return max(0, value)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        if numeric < 0:
            return default
        if numeric <= 1.0:
            return float(numeric)
        return int(round(numeric))

    @staticmethod
    def _clamp_df(value: float | int, n_rows: int) -> float | int:
        if isinstance(value, float):
            return min(max(value, 0.0), 1.0)
        return min(max(value, 0), n_rows)

    @staticmethod
    def _effective_df(value: float | int, n_rows: int) -> float:
        if isinstance(value, float):
            return value * n_rows
        return float(value)
