"""[STEP] Replace sentinel values with NaN"""
import textwrap
from typing import Any
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype
from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('features_precleaning')
class ActSentinelToNaN(Actionable):
    """[STEP] Replace sentinel values with NaN"""

    name: str = 'Replace sentinel values'
    _description: str = textwrap.dedent('''\
        Replace configured sentinel values (e.g., -999, "NA", "unknown") with NaN.''')
    _description_long: str = textwrap.dedent('''\
        Sentinel values are placeholders for missing data.
        This step replaces common numeric and text sentinels with NaN so downstream
        steps can treat missing values consistently.''')
    _usage: str = "Use when numeric or text columns contain sentinel placeholders (e.g., -999, 'NA') and you want them treated as missing; run before ActCoerceNumericStrings. Applicable to datasets with explicit sentinel codes or empty-string markers across numeric and categorical fields. Avoid when sentinel values are meaningful domain codes or you should remove sparse fields via ActDropHighMissingColumns."
    refs: list[dict[str, Any]] = []

    def __init__(self) -> None:
        self.configuration = {
            'numeric_sentinels': {
                'description': textwrap.dedent('''\
                    Numeric sentinel values to replace with NaN.'''),
                'default': [-999, -9999, -99999]
            },
            'text_sentinels': {
                'description': textwrap.dedent('''\
                    Text sentinel values to replace with NaN.'''),
                'default': ['NA', 'N/A', 'NULL', 'NONE', 'UNKNOWN', 'MISSING', 'NAN']
            },
            'case_insensitive': {
                'description': 'Match text sentinels ignoring case.',
                'default': True
            },
            'strip_whitespace': {
                'description': 'Trim whitespace before matching text sentinels.',
                'default': True
            },
            'include_empty_string': {
                'description': 'Treat empty strings as missing values.',
                'default': True
            },
            'numeric_in_text': {
                'description': 'Also match numeric sentinels stored as text.',
                'default': True
            }
        }
        self.numeric_columns: list[str] = []
        self.text_columns: list[str] = []
        self.numeric_sentinels: list[float] = []
        self.text_sentinels: list[str] = []

    def fit(self, dataset: Dataset) -> Actionable:
        self.numeric_columns, self.text_columns = self.__candidate_columns(dataset)
        self.numeric_sentinels, self.text_sentinels = self.__normalized_sentinels()

        self.explanations = []
        if dataset.X.empty:
            return self

        if self.numeric_sentinels:
            for column in self.numeric_columns:
                count = int(dataset.X[column].isin(self.numeric_sentinels).sum())
                if count:
                    self.explanations.append(
                        f'Replaced {count} numeric sentinel values in **`{column}`**.'
                    )

        if self.text_sentinels:
            for column in self.text_columns:
                mask = self.__text_sentinel_mask(dataset.X[column], self.text_sentinels)
                count = int(mask.sum())
                if count:
                    self.explanations.append(
                        f'Replaced {count} text sentinel values in **`{column}`**.'
                    )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace sentinel values with NaN.

        :param pd.DataFrame X: DataFrame to transform.
        :return: Transformed DataFrame.
        """
        if X.empty:
            return X

        if self.numeric_sentinels and self.numeric_columns:
            columns = [column for column in self.numeric_columns if column in X.columns]
            if columns:
                X[columns] = X[columns].replace(self.numeric_sentinels, np.nan)

        if self.text_sentinels and self.text_columns:
            for column in self.text_columns:
                if column not in X.columns:
                    continue
                mask = self.__text_sentinel_mask(X[column], self.text_sentinels)
                if mask.any():
                    X[column] = X[column].mask(mask, np.nan)

        return X

    def priorize(self, candidate: Candidate = None) -> float:
        if candidate is None or candidate.dataset.X.empty:
            return 0.0

        numeric_sentinels, text_sentinels = self.__normalized_sentinels()
        if not numeric_sentinels and not text_sentinels:
            return 0.0

        numeric_columns, text_columns = self.__candidate_columns(candidate.dataset)
        total = candidate.dataset.X.size or 1
        count = self.__count_sentinels(
            candidate.dataset,
            numeric_columns,
            text_columns,
            numeric_sentinels,
            text_sentinels
        )
        return min(1.5, 0.5 + count / total)

    def suitable(self, dataset: Dataset) -> bool:
        if dataset.X.empty:
            return False

        numeric_sentinels, text_sentinels = self.__normalized_sentinels()
        if not numeric_sentinels and not text_sentinels:
            return False

        numeric_columns, text_columns = self.__candidate_columns(dataset)
        if numeric_sentinels:
            for column in numeric_columns:
                if dataset.X[column].isin(numeric_sentinels).any():
                    return True

        if text_sentinels:
            for column in text_columns:
                if self.__text_sentinel_mask(dataset.X[column], text_sentinels).any():
                    return True

        return False

    def __candidate_columns(self, dataset: Dataset) -> tuple[list[str], list[str]]:
        numeric_candidates = set(dataset.get_columns_names_by_type(DataType.NUMERIC))
        text_candidates = set(dataset.get_columns_names_by_type(
            [DataType.CATEGORICAL, DataType.TEXT, DataType.SHORT_TEXT]
        ))

        numeric_columns = []
        text_columns = []

        for column in dataset.X.columns:
            if column in numeric_candidates:
                numeric_columns.append(column)
                continue
            if column in text_candidates:
                text_columns.append(column)
                continue

            series = dataset.X[column]
            if is_numeric_dtype(series):
                numeric_columns.append(column)
            elif is_string_dtype(series) or is_object_dtype(series):
                text_columns.append(column)

        return numeric_columns, text_columns

    def __normalized_sentinels(self) -> tuple[list[float], list[str]]:
        numeric_sentinels = self.__normalize_numeric_sentinels()
        text_sentinels = self.__normalize_text_sentinels(numeric_sentinels)
        return numeric_sentinels, text_sentinels

    def __normalize_numeric_sentinels(self) -> list[float]:
        values = self.get_config('numeric_sentinels') or []
        cleaned: list[float] = []
        seen: set[float] = set()

        for value in values:
            if value is None or isinstance(value, bool):
                continue
            try:
                num = float(value)
            except (TypeError, ValueError):
                continue
            if np.isnan(num) or num in seen:
                continue
            cleaned.append(num)
            seen.add(num)

        return cleaned

    def __normalize_text_sentinels(self, numeric_sentinels: list[float]) -> list[str]:
        values = list(self.get_config('text_sentinels') or [])
        if self.get_config('include_empty_string'):
            values.append('')
        if self.get_config('numeric_in_text'):
            values.extend(self.__numeric_sentinels_as_text(numeric_sentinels))

        cleaned: list[str] = []
        seen: set[str] = set()

        for value in values:
            if value is None:
                continue
            text = str(value)
            if self.get_config('strip_whitespace'):
                text = text.strip()
            if self.get_config('case_insensitive'):
                text = text.lower()
            if text == '' and not self.get_config('include_empty_string'):
                continue
            if text not in seen:
                cleaned.append(text)
                seen.add(text)

        return cleaned

    def __numeric_sentinels_as_text(self, numeric_sentinels: list[float]) -> list[str]:
        values: list[str] = []
        for value in numeric_sentinels:
            if np.isnan(value):
                continue
            if float(value).is_integer():
                int_value = int(value)
                values.append(str(int_value))
                values.append(str(float(int_value)))
            else:
                values.append(str(value))
        return values

    def __text_sentinel_mask(self, series: pd.Series, sentinels: list[str]) -> pd.Series:
        if not sentinels or series.empty:
            return pd.Series(False, index=series.index)

        values = series.astype('string')
        if self.get_config('strip_whitespace'):
            values = values.str.strip()
        if self.get_config('case_insensitive'):
            values = values.str.lower()
        mask = values.isin(sentinels)
        return mask.fillna(False)

    def __count_sentinels(self,
        dataset: Dataset,
        numeric_columns: list[str],
        text_columns: list[str],
        numeric_sentinels: list[float],
        text_sentinels: list[str]
    ) -> int:
        count = 0
        if numeric_sentinels:
            for column in numeric_columns:
                count += int(dataset.X[column].isin(numeric_sentinels).sum())
        if text_sentinels:
            for column in text_columns:
                count += int(self.__text_sentinel_mask(
                    dataset.X[column],
                    text_sentinels
                ).sum())
        return count
