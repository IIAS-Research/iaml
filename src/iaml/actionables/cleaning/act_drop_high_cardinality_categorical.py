"""[STEP] Drop or hash high-cardinality categorical columns."""
import textwrap
from typing import Any

import pandas as pd

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('cleaning')
class ActDropHighCardinalityCategorical(Actionable):
    """[STEP] Drop or hash high-cardinality categorical columns."""

    name: str = 'Handle high-cardinality categorical columns'
    _description: str = textwrap.dedent('''\
        Drop or hash categorical columns whose cardinality meets or exceeds {max_unique}
        or {unique_ratio_threshold:.0%} of non-missing rows.''')
    _description_long: str = textwrap.dedent('''\
        High-cardinality categorical columns can create large sparse encodings.
        Columns with too many distinct values are either removed or replaced with
        hashed integer codes (modulo {hash_bins}) depending on strategy {strategy}.
        Columns with fewer than {min_non_null} non-missing rows are ignored.''')
    _usage: str = "Use when categorical columns are extremely high-cardinality and you want drop/hash instead of ActCountVectorizer. Applicable to categorical features with high unique-to-row ratios. Avoid when categories are low-cardinality or predictive, or ActCategoricalImputer is enough."

    def __init__(self) -> None:
        self.configuration = {
            'strategy': {
                'description': 'How to handle high-cardinality columns.',
                'default': 'drop',
                'categorical': ['drop', 'hash']
            },
            'max_unique': {
                'description': 'Maximum number of distinct values before handling.',
                'default': 50
            },
            'unique_ratio_threshold': {
                'description': 'Minimum unique/non-missing ratio to mark as high-cardinality.',
                'default': 0.5,
                'range': [0.0, 1.0]
            },
            'min_non_null': {
                'description': 'Minimum number of non-missing rows to evaluate.',
                'default': 10
            },
            'hash_bins': {
                'description': 'Number of hash bins when strategy="hash".',
                'default': 64
            },
            'hash_key': {
                'description': 'Stable hash key used for hashing (16 chars recommended).',
                'default': 'iaml_hashing_key'
            },
            'missing_value': {
                'description': 'Numeric value used for missing categories when hashing.',
                'default': -1
            }
        }
        self.columns_to_drop: list[str] = []
        self.columns_to_hash: list[str] = []
        self.cardinality_stats: dict[str, dict[str, float]] = {}
        self.strategy: str = 'drop'
        self.hash_bins: int = 0
        self.hash_key: str | bytes | None = None
        self.missing_value: int = -1

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns_to_drop = []
        self.columns_to_hash = []
        self.cardinality_stats = {}
        self.explanations = []

        if dataset.X.empty:
            return self

        columns = self._select_columns(dataset)
        if not columns:
            return self

        self.strategy = self._coerce_strategy(self.get_config('strategy'))
        max_unique = self._coerce_int(self.get_config('max_unique'), 0)
        ratio_threshold = self._coerce_ratio(self.get_config('unique_ratio_threshold'), 0.0)
        min_non_null = self._coerce_int(self.get_config('min_non_null'), 0)

        if max_unique <= 0 and ratio_threshold <= 0:
            return self

        stats = self._cardinality_stats(dataset.X[columns])
        high_columns = self._high_cardinality_columns(
            columns,
            stats,
            max_unique,
            ratio_threshold,
            min_non_null
        )

        if not high_columns:
            return self

        if self.strategy == 'hash':
            self.hash_bins = self._coerce_bins(self.get_config('hash_bins'), 64)
            self.hash_key = self._coerce_hash_key(self.get_config('hash_key'))
            self.missing_value = self._coerce_int(self.get_config('missing_value'), -1)
            self.columns_to_hash = high_columns
        else:
            self.columns_to_drop = high_columns

        self.cardinality_stats = self._stats_to_dict(stats, high_columns)
        self._build_explanations(stats, high_columns)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns_to_drop:
            drop_columns = [col for col in self.columns_to_drop if col in X.columns]
            if drop_columns:
                X = X.drop(columns=drop_columns)

        if not self.columns_to_hash:
            return X

        hash_bins = self.hash_bins or self._coerce_bins(self.get_config('hash_bins'), 64)
        hash_key = self.hash_key if self.hash_key is not None else \
            self._coerce_hash_key(self.get_config('hash_key'))
        missing_value = self.missing_value

        for column in self.columns_to_hash:
            if column not in X.columns:
                continue
            X[column] = self._hash_series(X[column], hash_bins, hash_key, missing_value)

        return X

    def suitable(self, dataset: Dataset) -> bool:
        if dataset.X.empty:
            return False

        columns = self._select_columns(dataset)
        if not columns:
            return False

        max_unique = self._coerce_int(self.get_config('max_unique'), 0)
        ratio_threshold = self._coerce_ratio(self.get_config('unique_ratio_threshold'), 0.0)
        min_non_null = self._coerce_int(self.get_config('min_non_null'), 0)

        if max_unique <= 0 and ratio_threshold <= 0:
            return False

        stats = self._cardinality_stats(dataset.X[columns])
        high_columns = self._high_cardinality_columns(
            columns,
            stats,
            max_unique,
            ratio_threshold,
            min_non_null
        )
        return bool(high_columns)

    def priorize(self, candidate: Candidate = None) -> float:
        if candidate is None or candidate.dataset.X.empty:
            return 0.0

        columns = self._select_columns(candidate.dataset)
        if not columns:
            return 0.0

        max_unique = self._coerce_int(self.get_config('max_unique'), 0)
        ratio_threshold = self._coerce_ratio(self.get_config('unique_ratio_threshold'), 0.0)
        min_non_null = self._coerce_int(self.get_config('min_non_null'), 0)

        if max_unique <= 0 and ratio_threshold <= 0:
            return 0.0

        stats = self._cardinality_stats(candidate.dataset.X[columns])
        high_columns = self._high_cardinality_columns(
            columns,
            stats,
            max_unique,
            ratio_threshold,
            min_non_null
        )
        if not high_columns:
            return 0.0

        ratio = len(high_columns) / max(1, len(columns))
        return min(1.5, 0.5 + ratio)

    def _select_columns(self, dataset: Dataset) -> list[str]:
        columns = dataset.get_columns_names_by_type(DataType.CATEGORICAL)
        category_columns = list(dataset.X.select_dtypes(include=['category']).columns)
        seen: set[str] = set()
        ordered: list[str] = []
        for column in columns + category_columns:
            if column in dataset.X.columns and column not in seen:
                ordered.append(column)
                seen.add(column)
        return ordered

    def _cardinality_stats(self, X: pd.DataFrame) -> pd.DataFrame:
        non_null = X.notna().sum()
        unique = X.nunique(dropna=True)
        ratio = unique / non_null.replace(0, pd.NA)
        ratio = ratio.fillna(0.0)
        return pd.DataFrame({'unique': unique, 'non_null': non_null, 'ratio': ratio})

    def _high_cardinality_columns(
        self,
        columns: list[str],
        stats: pd.DataFrame,
        max_unique: int,
        ratio_threshold: float,
        min_non_null: int
    ) -> list[str]:
        eligible = stats['non_null'] >= min_non_null
        conditions = pd.Series(False, index=stats.index)
        if max_unique > 0:
            conditions |= stats['unique'] >= max_unique
        if ratio_threshold > 0:
            conditions |= stats['ratio'] >= ratio_threshold
        high = eligible & conditions
        return [column for column in columns if column in high.index and bool(high[column])]

    def _build_explanations(self, stats: pd.DataFrame, columns: list[str]) -> None:
        for column in columns:
            values = stats.loc[column]
            unique = int(values['unique'])
            non_null = int(values['non_null'])
            ratio = float(values['ratio']) if non_null else 0.0
            if self.strategy == 'hash':
                self.explanations.append(
                    f"Hashed `{column}` into {self.hash_bins} bins "
                    f"(unique {unique}/{non_null}, ratio {ratio:.2%})."
                )
            else:
                self.explanations.append(
                    f"Dropped `{column}` with {unique} unique values "
                    f"({ratio:.2%} of {non_null} non-missing)."
                )

    @staticmethod
    def _stats_to_dict(stats: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for column in columns:
            values = stats.loc[column]
            result[column] = {
                'unique': float(values['unique']),
                'non_null': float(values['non_null']),
                'ratio': float(values['ratio'])
            }
        return result

    @staticmethod
    def _hash_series(
        series: pd.Series,
        bins: int,
        hash_key: str | bytes | None,
        missing_value: int
    ) -> pd.Series:
        if bins <= 0:
            return pd.Series(missing_value, index=series.index, dtype='int64')

        mask = series.isna()
        values = series.astype('object')
        hashed = ActDropHighCardinalityCategorical._hash_values(values, hash_key)
        hashed = (hashed % bins).astype('int64')

        if missing_value is not None and mask.any():
            hashed = hashed.where(~mask, int(missing_value))

        return hashed

    @staticmethod
    def _hash_values(values: pd.Series, hash_key: str | bytes | None) -> pd.Series:
        if hash_key is None:
            return pd.util.hash_pandas_object(values, index=False)
        try:
            return pd.util.hash_pandas_object(values, index=False, hash_key=hash_key)
        except TypeError:
            return pd.util.hash_pandas_object(values, index=False)

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_ratio(value: Any, default: float) -> float:
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            return default
        return min(1.0, max(0.0, ratio))

    @staticmethod
    def _coerce_bins(value: Any, default: int) -> int:
        try:
            bins = int(value)
        except (TypeError, ValueError):
            return default
        if bins < 2:
            return default
        return bins

    @staticmethod
    def _coerce_strategy(value: Any) -> str:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {'drop', 'hash'}:
                return normalized
        return 'drop'

    @staticmethod
    def _coerce_hash_key(value: Any) -> str | bytes | None:
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray)):
            key = bytes(value)
            if not key:
                return None
            if len(key) < 16:
                key = key.ljust(16, b'0')
            elif len(key) > 16:
                key = key[:16]
            return key
        key = str(value)
        if not key:
            return None
        if len(key) < 16:
            key = key.ljust(16, '0')
        elif len(key) > 16:
            key = key[:16]
        return key
