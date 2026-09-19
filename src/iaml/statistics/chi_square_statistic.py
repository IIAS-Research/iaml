"""[STATISTIC] Chi-square."""
from __future__ import annotations

import textwrap
import pandas as pd
from scipy.stats import chi2_contingency

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class ChiSquareStatistic(Statistic):
    """[STATISTIC] Chi-square."""

    name: str = "Chi-square"
    _description: str = textwrap.dedent("""\
        Chi-square measures association between categorical columns and classes.
        """)
    _description_long: str = textwrap.dedent("""\
        Chi-square computes the chi-square test statistic for each categorical column
        against the target classes. The `_all` suffix reports the multi-class statistic,
        while per-class values compare a class against the rest.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'chi_square'

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

    def _chi2_statistic(self, feature: pd.Series, target: pd.Series) -> float | None:
        if not isinstance(target, pd.Series):
            target = pd.Series(target, index=feature.index)
        mask = feature.notna() & target.notna()
        if not mask.any():
            return None
        table = pd.crosstab(feature[mask], target[mask])
        if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
            return None
        stat, _, _, _ = chi2_contingency(table, correction=False)
        if pd.isna(stat):
            return None
        return float(stat)

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute chi-square statistics for categorical columns."""
        if dataset.type_of_target in ['survival', 'continuous']:
            return pd.DataFrame()

        columns = self._select_columns(dataset)
        if not columns:
            return pd.DataFrame()

        class_labels = list(pd.unique(dataset.y))
        data = []
        df_columns = []

        for col in columns:
            for label in ['all'] + class_labels:
                df_columns.append(f"{col}_{label}")
                if label == 'all':
                    data.append(self._chi2_statistic(dataset.X[col], dataset.y))
                else:
                    binary_target = pd.Series(dataset.y == label, index=dataset.X.index)
                    data.append(self._chi2_statistic(dataset.X[col], binary_target))

        return pd.DataFrame([data], index=[str(self)], columns=df_columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target not in ['survival', 'continuous']
