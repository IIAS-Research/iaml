"""[STEP] Group rare categories into a shared label."""
import math
import textwrap

import pandas as pd

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('cleaning')
class ActRareCategoryGrouper(Actionable):
    """[STEP] Group rare categories into a shared label."""

    name: str = 'Group rare categories'
    _description: str = textwrap.dedent('''\
        Replace rare categorical values with an "other" label to limit sparsity.''')
    _description_long: str = textwrap.dedent('''\
        Categories whose frequency is below {min_frequency:.0%} of the non-missing values
        or below {min_count} occurrences are grouped into {other_label!r}. This reduces
        the number of distinct categories before downstream encoders are applied.''')
    _usage: str = 'Use when rare labels in categorical features add sparsity and you want grouping instead of ActDropHighCardinalityCategorical. Applicable to categorical or pandas category columns before encoding. Avoid when rare labels are important or you prefer ActDropCategoricalColumn.'

    def __init__(self) -> None:
        self.configuration = {
            'min_frequency': {
                'description': textwrap.dedent('''\
                    Minimum ratio of non-missing rows required to keep a category.'''),
                'default': 0.01
            },
            'min_count': {
                'description': 'Minimum absolute count required to keep a category.',
                'default': 2
            },
            'other_label': {
                'description': 'Label used to replace rare categories.',
                'default': 'other'
            }
        }
        self.columns: list[str] = []
        self.rare_categories: dict[str, set] = {}

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = self._select_columns(dataset)
        self.rare_categories = {}
        self.explanations = []

        if not self.columns or dataset.X.empty:
            return self

        min_count = max(0, int(self.get_config('min_count')))
        min_frequency = max(0.0, float(self.get_config('min_frequency')))
        other_label = self.get_config('other_label')

        for column in self.columns:
            counts = dataset.X[column].value_counts(dropna=True)
            total = int(counts.sum())
            if total == 0 or counts.empty:
                continue

            threshold = self._threshold(min_count, min_frequency, total)
            if threshold <= 0:
                continue

            rare_counts = counts[counts < threshold]
            if rare_counts.empty:
                continue

            rare_labels = set(rare_counts.index.tolist())
            self.rare_categories[column] = rare_labels
            rare_total = int(rare_counts.sum())
            self.explanations.append(
                f"Grouped {len(rare_labels)} rare categories in `{column}` into "
                f"{other_label!r} ({rare_total} rows)."
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.rare_categories:
            return X

        other_label = self.get_config('other_label')
        for column, rare_values in self.rare_categories.items():
            if column not in X.columns or not rare_values:
                continue

            series = X[column]
            rare_mask = series.isin(rare_values)
            if not rare_mask.any():
                continue

            if pd.api.types.is_categorical_dtype(series):
                if other_label not in series.cat.categories:
                    series = series.cat.add_categories([other_label])
                series = series.where(~rare_mask, other_label)
                X[column] = series
            else:
                X.loc[rare_mask, column] = other_label

        return X

    def suitable(self, dataset: Dataset) -> bool:
        columns = self._select_columns(dataset)
        if not columns or dataset.X.empty:
            return False

        min_count = max(0, int(self.get_config('min_count')))
        min_frequency = max(0.0, float(self.get_config('min_frequency')))
        if min_count <= 0 and min_frequency <= 0:
            return False

        for column in columns:
            counts = dataset.X[column].value_counts(dropna=True)
            total = int(counts.sum())
            if total == 0 or counts.empty:
                continue

            threshold = self._threshold(min_count, min_frequency, total)
            if threshold > 0 and (counts < threshold).any():
                return True

        return False

    def priorize(self, candidate: Candidate = None) -> float:
        if candidate is None or candidate.dataset.X.empty:
            return 0.0

        columns = self._select_columns(candidate.dataset)
        if not columns:
            return 0.0

        min_count = max(0, int(self.get_config('min_count')))
        min_frequency = max(0.0, float(self.get_config('min_frequency')))

        total_values = 0
        rare_values = 0
        for column in columns:
            counts = candidate.dataset.X[column].value_counts(dropna=True)
            total = int(counts.sum())
            if total == 0 or counts.empty:
                continue
            threshold = self._threshold(min_count, min_frequency, total)
            if threshold <= 0:
                continue
            rare_counts = counts[counts < threshold]
            total_values += total
            rare_values += int(rare_counts.sum())

        if total_values == 0:
            return 0.0

        return min(1.0, max(0.0, rare_values / total_values))

    def _threshold(self, min_count: int, min_frequency: float, total: int) -> int:
        ratio_threshold = 0
        if min_frequency > 0 and total > 0:
            ratio_threshold = int(math.ceil(min_frequency * total))
        return max(min_count, ratio_threshold)

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
