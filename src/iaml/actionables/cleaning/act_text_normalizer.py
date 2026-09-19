"""[STEP] Normalize text columns before vectorization."""
from __future__ import annotations

import re
import string
import textwrap

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


_PUNCTUATION_RE = re.compile(rf"[{re.escape(string.punctuation)}]+")
_MULTISPACE_RE = re.compile(r"\s+")


@is_step('cleaning')
class ActTextNormalizer(Actionable):
    """[STEP] Normalize text columns before vectorization."""

    name: str = 'Text normalizer'
    _usage: str = 'Use when normalizing TEXT/SHORT_TEXT before ActCountVectorizer. Applicable to free-form text columns needing consistent casing, punctuation, or stopword cleanup. Avoid when text must remain verbatim or ActCountVectorizer settings already cover normalization.'
    _description: str = textwrap.dedent('''\
        Normalize text columns with lowercasing, punctuation removal, and stopword filtering.''')
    _description_long: str = textwrap.dedent('''\
        Prepare text columns for downstream vectorizers by standardizing casing, removing
        punctuation, dropping common stopwords, and cleaning up extra whitespace. This step
        targets columns typed as TEXT or SHORT_TEXT and keeps the processing deterministic.''')

    def __init__(self) -> None:
        self.configuration = {
            'lowercase': {
                'description': 'Lowercase text before normalization.',
                'default': True
            },
            'remove_punctuation': {
                'description': 'Replace punctuation characters with spaces.',
                'default': True
            },
            'remove_stopwords': {
                'description': 'Remove stopwords from text.',
                'default': True
            },
            'stopwords': {
                'description': 'Stopwords to remove ("english", "none", or a list).',
                'default': 'english'
            },
            'collapse_whitespace': {
                'description': 'Collapse repeated whitespace and strip edges.',
                'default': True
            }
        }

        self.columns: list[str] = []
        self.stop_words: set[str] = set()
        self._stopword_pattern: re.Pattern | None = None
        self._lowercase: bool = True
        self._remove_punctuation: bool = True
        self._remove_stopwords: bool = True
        self._collapse_whitespace: bool = True

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = self._select_columns(dataset)
        self.explanations = []

        self._lowercase = self._coerce_bool(self.get_config('lowercase'), True)
        self._remove_punctuation = self._coerce_bool(self.get_config('remove_punctuation'), True)
        self._remove_stopwords = self._coerce_bool(self.get_config('remove_stopwords'), True)
        self._collapse_whitespace = self._coerce_bool(self.get_config('collapse_whitespace'), True)

        self.stop_words = self._normalize_stopwords(self._resolve_stopwords())
        self._stopword_pattern = self._build_stopword_pattern(self.stop_words)

        if not self.columns or dataset.X.empty:
            return self

        if not self._has_enabled_operations():
            self.columns = []
            self.explanations.append('Text normalizer skipped: no enabled operations.')
            return self

        operations = self._operations_summary()
        for column in self.columns:
            self.explanations.append(
                f"Normalized text column **`{column}`** with {', '.join(operations)}."
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.columns or not self._has_enabled_operations():
            return X

        for column in self.columns:
            if column not in X.columns:
                continue

            series = X[column].fillna('').astype(str)

            if self._lowercase:
                series = series.str.lower()
            if self._remove_punctuation:
                series = series.str.replace(_PUNCTUATION_RE, ' ', regex=True)
            if self._stopword_pattern is not None:
                series = series.str.replace(self._stopword_pattern, ' ', regex=True)
            if self._collapse_whitespace:
                series = series.str.replace(_MULTISPACE_RE, ' ', regex=True).str.strip()

            X[column] = series

        return X

    def suitable(self, dataset: Dataset) -> bool:
        if dataset.X.empty:
            return False
        if not self._select_columns(dataset):
            return False
        return self._has_enabled_operations(configured_only=True)

    def priorize(self, candidate: Candidate = None) -> float:
        if candidate is None or candidate.dataset.X.empty:
            return 0.0

        columns = self._select_columns(candidate.dataset)
        if not columns or not self._has_enabled_operations(configured_only=True):
            return 0.0

        total_columns = candidate.dataset.X.shape[1] or 1
        return min(1.0, len(columns) / total_columns)

    def _select_columns(self, dataset: Dataset) -> list[str]:
        columns = dataset.get_columns_names_by_type([DataType.TEXT, DataType.SHORT_TEXT])
        return [column for column in columns if column in dataset.X.columns]

    def _resolve_stopwords(self) -> set[str]:
        if not self._coerce_bool(self.get_config('remove_stopwords'), True):
            return set()

        stopwords = self.get_config('stopwords')
        if stopwords is None:
            return set()

        if isinstance(stopwords, str):
            normalized = stopwords.strip().lower()
            if normalized in {'', 'none', 'false', 'off', 'no'}:
                return set()
            if normalized in {'english', 'sklearn'}:
                return set(ENGLISH_STOP_WORDS)
            tokens = re.split(r'[,;\s]+', stopwords.strip())
            return {token for token in tokens if token}

        if isinstance(stopwords, (list, tuple, set, frozenset)):
            return {
                str(word).strip()
                for word in stopwords
                if word is not None and str(word).strip()
            }

        return set()

    def _normalize_stopwords(self, stopwords: set[str]) -> set[str]:
        if not stopwords:
            return set()

        normalized: set[str] = set()
        for word in stopwords:
            token = str(word).strip()
            if not token:
                continue
            if self._lowercase:
                token = token.lower()
            if self._remove_punctuation:
                token = _PUNCTUATION_RE.sub(' ', token)
            token = _MULTISPACE_RE.sub(' ', token).strip()
            if not token:
                continue
            normalized.update(token.split())

        return normalized

    def _build_stopword_pattern(self, stopwords: set[str]) -> re.Pattern | None:
        if not stopwords:
            return None

        escaped = [re.escape(word) for word in stopwords if word]
        if not escaped:
            return None

        escaped.sort(key=len, reverse=True)
        flags = re.IGNORECASE if not self._lowercase else 0
        pattern = r'\b(?:' + '|'.join(escaped) + r')\b'
        return re.compile(pattern, flags=flags)

    def _operations_summary(self) -> list[str]:
        operations: list[str] = []
        if self._lowercase:
            operations.append('lowercasing')
        if self._remove_punctuation:
            operations.append('punctuation removal')
        if self._stopword_pattern is not None:
            operations.append('stopword filtering')
        if self._collapse_whitespace:
            operations.append('whitespace cleanup')
        if not operations:
            operations.append('normalization')
        return operations

    def _has_enabled_operations(self, configured_only: bool = False) -> bool:
        if configured_only:
            lowercase = self._coerce_bool(self.get_config('lowercase'), True)
            remove_punctuation = self._coerce_bool(self.get_config('remove_punctuation'), True)
            collapse_whitespace = self._coerce_bool(self.get_config('collapse_whitespace'), True)
            remove_stopwords = self._coerce_bool(self.get_config('remove_stopwords'), True)
            if lowercase or remove_punctuation or collapse_whitespace:
                return True
            if remove_stopwords:
                return bool(self._resolve_stopwords())
            return False

        if self._lowercase or self._remove_punctuation or self._collapse_whitespace:
            return True
        return self._stopword_pattern is not None

    @staticmethod
    def _coerce_bool(value: object, default: bool) -> bool:
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
