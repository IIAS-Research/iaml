"""[STATISTIC] Correlation with target."""
from __future__ import annotations

import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class CorrelationWithTargetStatistic(Statistic):
    """[STATISTIC] Correlation with target."""

    name: str = "Correlation with target"
    _description: str = textwrap.dedent("""\
        Correlation (continuous targets) or eta (classification) for numeric columns.
        """)
    _description_long: str = textwrap.dedent("""\
        For continuous targets, this reports Pearson correlations between each numeric
        feature and the target. For classification targets, this reports the eta
        correlation ratio between each numeric feature and the target classes. The
        `_all` suffix reports the multi-class statistic, while per-class values compare
        a class against the rest.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'correlation_with_target'

    def _pearson_corr(self, feature: pd.Series, target: pd.Series) -> float | None:
        if not isinstance(target, pd.Series):
            target = pd.Series(target, index=feature.index)
        values = pd.to_numeric(feature, errors='coerce')
        target_values = pd.to_numeric(target, errors='coerce')
        mask = values.notna() & target_values.notna()
        if mask.sum() < 2:
            return None
        if values[mask].nunique() < 2 or target_values[mask].nunique() < 2:
            return None
        corr = values[mask].corr(target_values[mask])
        if pd.isna(corr):
            return None
        return float(corr)

    def _eta(self, feature: pd.Series, target: pd.Series) -> float | None:
        if not isinstance(target, pd.Series):
            target = pd.Series(target, index=feature.index)
        mask = feature.notna() & target.notna()
        if not mask.any():
            return None
        values = feature[mask]
        groups = target[mask]
        if groups.nunique() < 2:
            return None
        overall_mean = values.mean()
        ss_total = ((values - overall_mean) ** 2).sum()
        if ss_total == 0:
            return None
        ss_between = 0.0
        for _, group_values in values.groupby(groups):
            count = group_values.size
            if count == 0:
                continue
            mean = group_values.mean()
            ss_between += count * (mean - overall_mean) ** 2
        eta_sq = ss_between / ss_total
        if pd.isna(eta_sq):
            return None
        return float(eta_sq) ** 0.5

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute correlations or eta with the target for numeric columns."""
        if dataset.type_of_target == 'survival':
            return pd.DataFrame()

        numeric_columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not numeric_columns:
            return pd.DataFrame()

        data = []
        columns = []

        if dataset.type_of_target == 'continuous':
            for col in numeric_columns:
                columns.append(col)
                data.append(self._pearson_corr(dataset.X[col], dataset.y))
            return pd.DataFrame([data], index=[str(self)], columns=columns)

        class_labels = list(pd.unique(dataset.y))
        for col in numeric_columns:
            overall_eta = self._eta(dataset.X[col], dataset.y)
            for label in ['all'] + class_labels:
                columns.append(f"{col}_{label}")
                if label == 'all':
                    data.append(overall_eta)
                else:
                    binary_target = pd.Series(dataset.y == label, index=dataset.X.index)
                    data.append(self._eta(dataset.X[col], binary_target))

        return pd.DataFrame([data], index=[str(self)], columns=columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target != 'survival'
