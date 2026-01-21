"""[STATISTIC] ANOVA."""
from __future__ import annotations

import textwrap
import pandas as pd
from scipy import stats

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class ANOVAStatistic(Statistic):
    """[STATISTIC] ANOVA."""

    name: str = "ANOVA"
    _description: str = textwrap.dedent("""\
        ANOVA tests whether numerical feature means differ across classes.
        """)
    _description_long: str = textwrap.dedent("""\
        ANOVA computes one-way p-values for numerical columns across classes.
        Per-class columns report ANOVA p-values comparing each class against the
        rest of the data, while the `_all` suffix reports the overall ANOVA p-value.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'anova'

    def _anova_pvalue(self, groups: list[pd.Series]) -> float | None:
        cleaned = []
        for values in groups:
            if values is None:
                continue
            series = values.dropna()
            if series.size > 1:
                cleaned.append(series.to_numpy())
        if len(cleaned) < 2:
            return None
        result = stats.f_oneway(*cleaned)
        pvalue = float(result.pvalue)
        if pd.isna(pvalue):
            return None
        return pvalue

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute ANOVA p-values for numerical columns."""
        if dataset.type_of_target in ['survival', 'continuous']:
            return pd.DataFrame()

        numeric_columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not numeric_columns:
            return pd.DataFrame()

        class_labels = list(pd.unique(dataset.y))
        class_masks = {label: (dataset.y == label) for label in class_labels}

        data = []
        columns = []

        for col in numeric_columns:
            col_values = dataset.X[col]
            overall_groups = [col_values[mask] for mask in class_masks.values()]
            overall_pvalue = self._anova_pvalue(overall_groups)

            for label in ['all'] + class_labels:
                columns.append(f"{col}_{label}")
                if label == 'all':
                    data.append(overall_pvalue)
                else:
                    mask = class_masks[label]
                    class_values = col_values[mask]
                    rest_values = col_values[~mask]
                    data.append(self._anova_pvalue([class_values, rest_values]))

        return pd.DataFrame([data], index=[str(self)], columns=columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target not in ['survival', 'continuous']
