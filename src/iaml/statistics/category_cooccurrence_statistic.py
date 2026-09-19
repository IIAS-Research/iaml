"""[STATISTIC] Category Cooccurrence."""
from __future__ import annotations

import itertools
import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class CategoryCooccurrenceStatistic(Statistic):
    """[STATISTIC] Category Cooccurrence."""

    name: str = "Category Cooccurrence"
    _description: str = textwrap.dedent("""\
        Category cooccurrence counts value pairs between categorical column pairs.
        """)
    _description_long: str = textwrap.dedent("""\
        Category cooccurrence counts value pairs between categorical column pairs,
        optionally per class for classification.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'category_cooccurrence'

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

    def _cooccurrence(self, frame: pd.DataFrame, col_a: str, col_b: str) -> list[tuple[tuple, int]]:
        values = frame[[col_a, col_b]].dropna()
        if values.empty:
            return []
        counts = values.groupby([col_a, col_b], sort=True).size()
        return [((idx_a, idx_b), int(count)) for (idx_a, idx_b), count in counts.items()]

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute category cooccurrence for categorical column pairs."""
        if dataset.type_of_target == 'survival':
            return pd.DataFrame()

        columns = self._select_columns(dataset)
        if len(columns) < 2:
            return pd.DataFrame()

        pairs = list(itertools.combinations(columns, 2))
        data = []
        df_columns = []

        if dataset.type_of_target == 'continuous':
            for col_a, col_b in pairs:
                df_columns.append(f"{col_a}__{col_b}")
                data.append(self._cooccurrence(dataset.X, col_a, col_b))
            return pd.DataFrame([data], index=[str(self)], columns=df_columns)

        class_labels = list(pd.unique(dataset.y))
        for col_a, col_b in pairs:
            for label in ['all'] + class_labels:
                df_columns.append(f"{col_a}__{col_b}_{label}")
                if label == 'all':
                    frame = dataset.X
                else:
                    frame = dataset.X.loc[dataset.y == label]
                data.append(self._cooccurrence(frame, col_a, col_b))
        return pd.DataFrame([data], index=[str(self)], columns=df_columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target != 'survival'
