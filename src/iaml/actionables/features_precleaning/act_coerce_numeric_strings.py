"""[STEP] Coerce numeric strings to numeric values"""
import textwrap
from typing import Any
import pandas as pd
from ...actionable import Actionable
from ...dataset import Dataset
from ...data_type import DataType
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('features_precleaning')
class ActCoerceNumericStrings(Actionable):
    """[STEP] Coerce numeric strings to numeric values"""

    name: str = 'Coerce numeric strings'
    _description: str = textwrap.dedent('''\
        Convert text columns that look numeric (commas, spaces, %) into numeric values.''')
    _usage: str = 'Use when text columns contain numeric-like strings with separators or %; prefer over ActDateConverter for dates. Applicable to text or categorical columns that are mostly numeric. Avoid when values are identifiers or mostly non-numeric; use ActSentinelToNaN for missing tokens.'
    _description_long: str = textwrap.dedent('''\
        This step attempts to coerce text or categorical columns into numeric values
        by removing spaces, thousands separators and percent signs. A column is converted
        only if the error ratio is below {authorized_error_ratio:.0%} of the non-missing
        values in a sample of size {sample_size}.''')
    refs: list[dict[str, Any]] = []

    def __init__(self):
        self.configuration = {
            'authorized_error_ratio': {
                'description': textwrap.dedent('''\
                    Maximum ratio of non-convertible values allowed to coerce a column.'''),
                'default': 0.1
            },
            'sample_size': {
                'description': textwrap.dedent('''\
                    Size of the sample used to detect numeric strings.
                    Set to -1 to use the whole dataset.'''),
                'default': 200
            },
            'percent_to_fraction': {
                'description': 'Convert values with percent signs into fractions (divide by 100).',
                'default': True
            }
        }
        self.columns: list[tuple[str, str]] = None

    def fit(self, dataset: Dataset) -> Actionable:
        sample = self.__sample(dataset.X)
        self.columns = []
        explanations = []

        for column in self.__candidate_columns(dataset):
            strategy, ratio = self.__choose_strategy(sample[column])
            if strategy is None:
                continue
            self.columns.append((column, strategy))
            explanations.append(
                f'Coerced column **`{column}`** to numeric using {strategy} decimal separator '
                f'(success {ratio:.0%} on sample).'
            )

        self.explanations = explanations
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Coerce numeric-like strings into numeric values.

        :param pd.DataFrame X: DataFrame to transform.
        :return: Transformed DataFrame.
        """
        if not self.columns:
            return X

        for column, strategy in self.columns:
            if column not in X.columns:
                continue
            X[column] = self.__convert_series(X[column], strategy)

        return X

    def priorize(self, candidate: Candidate = None) -> float:
        if candidate is None:
            return 1.0
        total = max(1, candidate.dataset.X.shape[1])
        candidates = len(self.__candidate_columns(candidate.dataset))
        return min(1.0, 0.5 + candidates / total)

    def suitable(self, dataset: Dataset) -> bool:
        sample = self.__sample(dataset.X)
        for column in self.__candidate_columns(dataset):
            strategy, _ = self.__choose_strategy(sample[column])
            if strategy is not None:
                return True
        return False

    def __candidate_columns(self, dataset: Dataset) -> list[str]:
        return dataset.get_columns_names_by_type(
            [DataType.SHORT_TEXT, DataType.TEXT, DataType.CATEGORICAL]
        )

    def __sample(self, X: pd.DataFrame) -> pd.DataFrame:
        if X.empty:
            return X
        sample_size = self.get_config('sample_size')
        if sample_size is None or sample_size == 0:
            return X
        if sample_size < 0 or sample_size >= len(X):
            return X
        return X.sample(n=sample_size, random_state=42)

    def __choose_strategy(self, series: pd.Series) -> tuple[str | None, float]:
        non_null = series.notna().sum()
        if non_null == 0:
            return None, 0.0

        ratios = {}
        for strategy in ('dot', 'comma'):
            converted = self.__convert_series(series, strategy)
            ratios[strategy] = converted.notna().sum() / non_null

        best = max(ratios, key=ratios.get)
        best_ratio = ratios[best]
        if best_ratio < 1 - self.get_config('authorized_error_ratio'):
            return None, best_ratio

        other = 'comma' if best == 'dot' else 'dot'
        if abs(ratios[best] - ratios[other]) <= 0.02:
            inferred = self.__infer_decimal_separator(series)
            if ratios.get(inferred, 0) >= 1 - self.get_config('authorized_error_ratio'):
                best = inferred

        return best, best_ratio

    def __normalize_strings(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        values = series.where(series.notna(), '')
        values = values.astype(str).str.strip()
        percent_mask = values.str.contains('%', regex=False, na=False)
        values = values.str.replace('%', '', regex=False)
        values = values.str.replace(r'\s+', '', regex=True)
        return values, percent_mask

    def __convert_series(self, series: pd.Series, strategy: str) -> pd.Series:
        values, percent_mask = self.__normalize_strings(series)

        if strategy == 'comma':
            values = values.str.replace('.', '', regex=False)
            values = values.str.replace(',', '.', regex=False)
        else:
            values = values.str.replace(',', '', regex=False)

        numeric = pd.to_numeric(values, errors='coerce')
        if self.get_config('percent_to_fraction'):
            numeric = numeric.mask(percent_mask, numeric / 100)

        return numeric

    def __infer_decimal_separator(self, series: pd.Series) -> str:
        values, _ = self.__normalize_strings(series)

        has_comma = values.str.contains(',', regex=False, na=False)
        has_dot = values.str.contains('.', regex=False, na=False)

        both = has_comma & has_dot
        if both.any():
            last_comma = values[both].str.rfind(',')
            last_dot = values[both].str.rfind('.')
            comma_last = (last_comma > last_dot).sum()
            dot_last = (last_dot > last_comma).sum()
            if comma_last != dot_last:
                return 'comma' if comma_last > dot_last else 'dot'

        comma_only = has_comma & ~has_dot
        dot_only = has_dot & ~has_comma

        if comma_only.any() and not dot_only.any():
            return self.__separator_preference(values[comma_only], ',')
        if dot_only.any() and not comma_only.any():
            return self.__separator_preference(values[dot_only], '.')
        if comma_only.any() or dot_only.any():
            return 'comma' if comma_only.sum() > dot_only.sum() else 'dot'

        return 'dot'

    def __separator_preference(self, values: pd.Series, separator: str) -> str:
        after = values.str.rsplit(separator, n=1).str[-1]
        decimal_like = after.str.len().isin([1, 2]).sum()
        thousand_like = after.str.len().eq(3).sum()

        if decimal_like > thousand_like:
            return 'comma' if separator == ',' else 'dot'
        if thousand_like > decimal_like:
            return 'dot' if separator == ',' else 'comma'

        return 'comma' if separator == ',' else 'dot'
