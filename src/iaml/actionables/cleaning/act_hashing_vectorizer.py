"""[STEP] Vectorize large text with HashingVectorizer."""
import textwrap
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('cleaning')
class ActHashingVectorizer(Actionable):
    """[STEP] Vectorize large text with HashingVectorizer."""

    name: str = 'Hashing Vectorizer'
    _description: str = textwrap.dedent('''\
        Vectorize text columns with the hashing trick for large vocabularies.''')
    _description_long: str = textwrap.dedent('''\
        Convert long text columns into fixed-size hashed feature vectors without
        building an explicit vocabulary. This keeps memory usage bounded even for
        very large vocabularies, at the cost of possible hash collisions.''')
    _usage: str = "Use when text columns have huge vocabularies and you need fixed-size features, especially vs ActCountVectorizer for memory bounds. Applicable to free-text columns that you want numeric n-grams from. Avoid when token interpretability or collision-free features are required."

    def __init__(self) -> None:
        self.columns: list[str] = []
        self.vectorizer: HashingVectorizer | None = None
        self.configuration = {
            'n_features': {
                'description': 'Number of hash bins (power of two recommended).',
                'default': 4096
            },
            'ngram_min': {
                'description': 'Minimum n-gram size to include.',
                'default': 1
            },
            'ngram_max': {
                'description': 'Maximum n-gram size to include.',
                'default': 2
            },
            'alternate_sign': {
                'description': 'Use alternating signs to reduce hash collisions.',
                'default': False
            },
            'binary': {
                'description': 'If True, store binary occurrences instead of counts.',
                'default': False
            },
            'norm': {
                'description': 'Vector normalization ("l1", "l2", or None).',
                'default': 'l2'
            },
            'lowercase': {
                'description': 'Lowercase text before hashing.',
                'default': True
            }
        }

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = []
        self.vectorizer = None
        self.explanations = []

        columns = dataset.get_columns_names_by_type([DataType.TEXT])
        if not columns or dataset.X.empty:
            return self

        params = self._build_vectorizer_params()
        self.vectorizer = HashingVectorizer(**params)
        self.columns = [column for column in columns if column in dataset.X.columns]

        n_features = params['n_features']
        for column in self.columns:
            self.explanations.append(
                f'Encoded text column **`{column}`** into **{n_features}** hashed features.'
            )

        if not self.columns:
            self.explanations.append(
                'Hashing vectorizer skipped: no usable text columns.'
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply HashingVectorizer to text columns.

        :param pd.DataFrame X: DataFrame to transform.
        :return: Transformed dataset.
        """
        if not self.columns or self.vectorizer is None:
            return X

        X = X.reset_index(drop=True)

        for name in self.columns:
            if name not in X.columns:
                continue

            values = X[name].fillna('').astype(str)
            transformed = self.vectorizer.transform(values)
            n_features = transformed.shape[1]
            new_names = [f"{name}_hash_{i}" for i in range(n_features)]
            vector_df = pd.DataFrame(transformed.toarray(), columns=new_names)

            X = pd.concat([X, vector_df], axis=1).drop([name], axis=1)

        return X

    def suitable(self, dataset: Dataset) -> bool:
        if dataset.X.empty:
            return False
        return bool(dataset.get_columns_names_by_type([DataType.TEXT]))

    def priorize(self, candidate: Candidate = None) -> float:
        if candidate is None or candidate.dataset.X.empty:
            return 0.0

        columns = candidate.dataset.get_columns_names_by_type([DataType.TEXT])
        if not columns:
            return 0.0

        total_columns = candidate.dataset.X.shape[1] or 1
        return min(1.0, len(columns) / total_columns)

    def _build_vectorizer_params(self) -> dict[str, Any]:
        n_features = self._coerce_positive_int(self.get_config('n_features'), 4096)
        ngram_min = self._coerce_int(self.get_config('ngram_min'), 1)
        ngram_max = self._coerce_int(self.get_config('ngram_max'), max(ngram_min, 1))
        ngram_min = max(1, ngram_min)
        ngram_max = max(ngram_min, ngram_max)

        return {
            'n_features': n_features,
            'ngram_range': (ngram_min, ngram_max),
            'alternate_sign': self._coerce_bool(self.get_config('alternate_sign'), False),
            'binary': self._coerce_bool(self.get_config('binary'), False),
            'norm': self._coerce_norm(self.get_config('norm')),
            'lowercase': self._coerce_bool(self.get_config('lowercase'), True)
        }

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_positive_int(value: Any, default: int) -> int:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return default
        if numeric <= 0:
            return default
        return numeric

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {'true', '1', 'yes', 'y'}:
                return True
            if normalized in {'false', '0', 'no', 'n'}:
                return False
        if value is None:
            return default
        return bool(value)

    @staticmethod
    def _coerce_norm(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {'none', ''}:
                return None
            if normalized in {'l1', 'l2'}:
                return normalized
        return 'l2'
