"""[STEP] Normalize column names"""
import re
import textwrap
import pandas as pd
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


_NON_ALNUM_RE = re.compile(r'[^0-9a-zA-Z_]+')
_MULTI_UNDERSCORE_RE = re.compile(r'_+')


@is_step('features_precleaning')
class ActNormalizeColumnNames(Actionable):
    """[STEP] Normalize column names"""

    name: str = 'Normalize column names'
    _description: str = 'Standardize column names with lowercase and underscores'
    _usage: str = 'Use when column names are messy or inconsistent; run before ActCoerceNumericStrings or ActDateConverter. Applicable to raw tables with human-entered headers, spaces, symbols, or duplicates. Avoid when names already standardized or must remain exact for downstream joins.'
    _description_long: str = textwrap.dedent('''\
        Standardize column names so they are lowercase and use underscores.
        This reduces naming collisions and keeps downstream feature selection consistent.
    ''')

    refs = []

    def __init__(self):
        self.configuration = {
            'lowercase': {
                'description': 'Convert column names to lowercase',
                'default': True
            },
            'strip': {
                'description': 'Trim leading and trailing spaces',
                'default': True
            },
            'replace_spaces': {
                'description': 'Replace spaces with underscores',
                'default': True
            },
            'replace_non_alnum': {
                'description': 'Replace non alphanumeric characters with underscores',
                'default': True
            },
            'collapse_underscores': {
                'description': 'Collapse consecutive underscores',
                'default': True
            },
            'strip_underscores': {
                'description': 'Trim leading and trailing underscores',
                'default': True
            },
            'deduplicate': {
                'description': 'Ensure column names are unique after normalization',
                'default': True
            },
            'dedupe_sep': {
                'description': 'Separator used when deduplicating column names',
                'default': '_'
            },
            'empty_fallback': {
                'description': 'Fallback base name for empty column names',
                'default': 'column'
            },
        }

        self.columns: list = None
        self.normalized_columns: list[str] = None

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = dataset.features
        normalized = self._normalize_columns(self.columns)
        if self.get_config('deduplicate'):
            normalized = self._deduplicate(normalized)

        self.normalized_columns = normalized
        self.explanations = [
            f'Normalize column name **`{old}`** -> **`{new}`**.'
            for old, new in zip(self.columns, self.normalized_columns)
            if old != new
        ]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names.

        :param pd.DataFrame X: DataFrame to transform.
        :return: Transformed dataset.
        """
        if not self.normalized_columns or list(X.columns) == self.normalized_columns:
            return X

        X.columns = self.normalized_columns
        return X

    def suitable(self, dataset: Dataset) -> bool:
        columns = dataset.features
        normalized = self._normalize_columns(columns)
        if self.get_config('deduplicate'):
            normalized = self._deduplicate(normalized)
        return columns != normalized

    def priorize(self, candidate: Candidate = None) -> float:
        return 1.0

    def _normalize_columns(self, columns: list) -> list[str]:
        return [self._normalize_name(column) for column in columns]

    def _normalize_name(self, name: object) -> str:
        value = '' if name is None else str(name)

        if self.get_config('strip'):
            value = value.strip()
        if self.get_config('lowercase'):
            value = value.lower()
        if self.get_config('replace_spaces'):
            value = re.sub(r'\s+', '_', value)
        if self.get_config('replace_non_alnum'):
            value = _NON_ALNUM_RE.sub('_', value)
        if self.get_config('collapse_underscores'):
            value = _MULTI_UNDERSCORE_RE.sub('_', value)
        if self.get_config('strip_underscores'):
            value = value.strip('_')

        if value == '':
            value = self.get_config('empty_fallback')

        return value

    def _deduplicate(self, names: list[str]) -> list[str]:
        used = set()
        counts = {}
        sep = self.get_config('dedupe_sep')
        deduped = []

        for name in names:
            base = name or self.get_config('empty_fallback')
            idx = counts.get(base, 0)

            if base in used:
                idx = max(idx, 1)
                candidate = f"{base}{sep}{idx}"
                while candidate in used:
                    idx += 1
                    candidate = f"{base}{sep}{idx}"
                name = candidate
            else:
                name = base

            used.add(name)
            counts[base] = idx + 1
            deduped.append(name)

        return deduped
