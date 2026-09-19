"""[STEP] Drop id-like columns"""
import re
import textwrap
from typing import Any
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


_UUID_PATTERN = (
    r'^(?:[0-9a-f]{32}|'
    r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$'
)
_HEX_PATTERN = r'^[0-9a-f]{16,}$'
_TOKEN_PATTERN = r'^[A-Za-z0-9_-]+$'
_DIGITS_PATTERN = r'^\d+$'
_CAMEL_RE = re.compile(r'([a-z])([A-Z])')
_SPLIT_RE = re.compile(r'[^a-z0-9]+')


@is_step('features_precleaning')
class ActDropIdLikeColumns(Actionable):
    """[STEP] Drop id-like columns"""

    name: str = 'Drop id-like columns'
    _description: str = textwrap.dedent('''\
        Drop columns that are quasi-unique and look like identifiers.''')
    _description_long: str = textwrap.dedent('''\
        Identifier columns such as IDs, UUIDs, hashes, and keys are often quasi-unique
        and do not carry predictive signal. This step removes columns that are nearly
        unique and match identifier-like name or value patterns.''')
    _usage: str = 'Use when columns are quasi-unique identifiers with little signal, rather than ActDropDuplicateRows. Applicable to mixed tabular data with ID/UUID/hash-like fields. Avoid when IDs encode meaning or when missingness cleanup via ActDropHighMissingColumns is the issue.'
    refs: list[dict[str, Any]] = []

    def __init__(self) -> None:
        self.configuration = {
            'unique_ratio_threshold': {
                'description': textwrap.dedent('''\
                    Minimum ratio of unique (non-null) values for a column to be
                    considered quasi-unique.'''),
                'default': 0.98,
                'range': [0.0, 1.0]
            },
            'min_unique': {
                'description': 'Minimum number of unique values to consider a column.',
                'default': 20
            },
            'min_non_null': {
                'description': 'Minimum number of non-null values to evaluate a column.',
                'default': 10
            },
            'name_tokens': {
                'description': 'Tokens indicating identifier-like column names.',
                'default': [
                    'id', 'uuid', 'guid', 'identifier', 'key', 'code', 'ref',
                    'reference', 'hash', 'token', 'serial', 'sequence', 'seq', 'index'
                ]
            },
            'sample_size': {
                'description': textwrap.dedent('''\
                    Sample size used for pattern detection on text columns.
                    Set to -1 to scan the full column.'''),
                'default': 500
            },
            'uuid_ratio_threshold': {
                'description': 'Minimum ratio of UUID-like values.',
                'default': 0.8,
                'range': [0.0, 1.0]
            },
            'hex_ratio_threshold': {
                'description': 'Minimum ratio of hex-like values.',
                'default': 0.9,
                'range': [0.0, 1.0]
            },
            'digits_ratio_threshold': {
                'description': 'Minimum ratio of digit-only values.',
                'default': 0.9,
                'range': [0.0, 1.0]
            },
            'digits_min_length': {
                'description': 'Minimum length for digit-only values to count as IDs.',
                'default': 4
            },
            'token_ratio_threshold': {
                'description': 'Minimum ratio of mixed alphanumeric tokens.',
                'default': 0.9,
                'range': [0.0, 1.0]
            },
            'token_min_length': {
                'description': 'Minimum length for mixed alphanumeric tokens.',
                'default': 6
            },
            'integer_ratio_threshold': {
                'description': 'Minimum ratio of integer-like values for numeric checks.',
                'default': 0.95,
                'range': [0.0, 1.0]
            }
        }
        self.columns_to_drop: list[str] = []
        self.drop_reasons: dict[str, str] = {}

    def fit(self, dataset: Dataset) -> Actionable:
        matches = self.__find_id_like_columns(dataset)
        self.columns_to_drop = [match['column'] for match in matches]
        self.drop_reasons = {
            match['column']: match['reason']
            for match in matches
        }
        self.explanations = [
            f'Dropped column **`{match["column"]}`** because it is quasi-unique '
            f'({match["unique_ratio"]:.2%}) and {match["reason"]}.'
            for match in matches
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop id-like columns.

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
        if candidate is None or candidate.dataset.X.empty:
            return 0.0
        matches = self.__find_id_like_columns(candidate.dataset)
        if not matches:
            return 0.0
        ratio = len(matches) / max(1, candidate.dataset.X.shape[1])
        return min(1.5, 0.5 + ratio)

    def suitable(self, dataset: Dataset) -> bool:
        if dataset.X.empty:
            return False
        return bool(self.__find_id_like_columns(dataset))

    def __find_id_like_columns(self, dataset: Dataset) -> list[dict[str, Any]]:
        columns = self.__candidate_columns(dataset)
        if not columns or dataset.X.empty:
            return []

        unique_threshold = self.get_config('unique_ratio_threshold')
        min_unique = self.get_config('min_unique')
        min_non_null = self.get_config('min_non_null')

        matches: list[dict[str, Any]] = []
        for column in columns:
            series = dataset.X[column]
            non_null = int(series.notna().sum())
            if non_null < min_non_null:
                continue
            unique_count = int(series.nunique(dropna=True))
            if unique_count < min_unique:
                continue
            unique_ratio = unique_count / non_null if non_null else 0.0
            if unique_ratio < unique_threshold:
                continue

            reason = self.__id_like_reason(column, series)
            if reason:
                matches.append({
                    'column': column,
                    'unique_ratio': unique_ratio,
                    'unique_count': unique_count,
                    'non_null': non_null,
                    'reason': reason
                })

        return matches

    def __candidate_columns(self, dataset: Dataset) -> list[str]:
        columns = dataset.get_columns_names_by_type(list(DataType))
        if len(columns) != dataset.X.shape[1]:
            missing = [column for column in dataset.X.columns if column not in columns]
            columns.extend(missing)
        return columns

    def __id_like_reason(self, column: object, series: pd.Series) -> str | None:
        reasons: list[str] = []
        if self.__name_looks_like_id(column):
            reasons.append('column name looks like an identifier')

        if is_datetime64_any_dtype(series):
            return '; '.join(reasons) if reasons else None

        if is_numeric_dtype(series):
            numeric_reason = self.__numeric_reason(series)
            if numeric_reason:
                reasons.append(numeric_reason)
        elif is_categorical_dtype(series) or is_string_dtype(series) or is_object_dtype(series):
            text_reason = self.__text_reason(series)
            if text_reason:
                reasons.append(text_reason)

        if reasons:
            return '; '.join(reasons)
        return None

    def __numeric_reason(self, series: pd.Series) -> str | None:
        values = pd.to_numeric(series, errors='coerce').dropna()
        if values.empty:
            return None

        integer_ratio = self.__integer_ratio(values)
        if integer_ratio < self.get_config('integer_ratio_threshold'):
            return None

        unique_count = values.nunique(dropna=True)
        if unique_count <= 1:
            return None

        value_min = values.min()
        value_max = values.max()
        if value_max - value_min == unique_count - 1:
            return 'values form a consecutive integer range'

        return None

    def __text_reason(self, series: pd.Series) -> str | None:
        sample = self.__sample_text(series)
        if sample.empty:
            return None

        lower = sample.str.lower()
        uuid_ratio = lower.str.fullmatch(_UUID_PATTERN).mean()
        if uuid_ratio >= self.get_config('uuid_ratio_threshold'):
            return f'values look like UUIDs ({uuid_ratio:.0%})'

        hex_ratio = lower.str.fullmatch(_HEX_PATTERN).mean()
        if hex_ratio >= self.get_config('hex_ratio_threshold'):
            return f'values look like hex hashes ({hex_ratio:.0%})'

        lengths = sample.str.len()
        digits_mask = lower.str.fullmatch(_DIGITS_PATTERN)
        digits_ratio = (digits_mask & (lengths >= self.get_config('digits_min_length'))).mean()
        if digits_ratio >= self.get_config('digits_ratio_threshold'):
            return f'values are digit-only tokens ({digits_ratio:.0%})'

        token_mask = sample.str.fullmatch(_TOKEN_PATTERN)
        mixed_mask = (
            token_mask
            & (lengths >= self.get_config('token_min_length'))
            & sample.str.contains(r'[A-Za-z]', regex=True)
            & sample.str.contains(r'\d', regex=True)
        )
        mixed_ratio = mixed_mask.mean()
        if mixed_ratio >= self.get_config('token_ratio_threshold'):
            return f'values are mixed alphanumeric tokens ({mixed_ratio:.0%})'

        return None

    def __sample_text(self, series: pd.Series) -> pd.Series:
        values = series.dropna()
        if values.empty:
            return values.astype(str)

        sample_size = self.get_config('sample_size')
        if sample_size is not None and sample_size > 0 and len(values) > sample_size:
            values = values.sample(n=sample_size, random_state=0)

        return values.astype(str)

    def __name_looks_like_id(self, name: object) -> bool:
        if name is None:
            return False
        raw = str(name)
        normalized = _CAMEL_RE.sub(r'\1_\2', raw).lower()
        tokens = [token for token in _SPLIT_RE.split(normalized) if token]
        id_tokens = {token.lower() for token in (self.get_config('name_tokens') or [])}
        return any(token in id_tokens for token in tokens)

    def __integer_ratio(self, values: pd.Series) -> float:
        numeric = pd.to_numeric(values, errors='coerce').dropna().to_numpy()
        if numeric.size == 0:
            return 0.0
        frac = np.mod(numeric, 1)
        return float(np.isclose(frac, 0).mean())
