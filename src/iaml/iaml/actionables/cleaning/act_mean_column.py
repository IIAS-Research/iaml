"""
[STEP] Fill missing values with mean
"""
import textwrap
import pandas as pd
import numpy as np
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step
from ...data_type import DataType


@is_step('cleaning', 'baseline_cleaning')
class ActMeanColumn(Actionable):
    """
    [STEP] Fill missing values with the mean.
    """
    name = 'Fill missing values'
    description = 'Fill missing values with the mean of non-missing values.'
    description_long = description
    can_be_disabled = False

    def __init__(self):
        self.columns: list[str] = None

    def fit(self, dataset:Dataset) -> Actionable:
        self.columns = []
        explain = []

        for column in dataset.get_columns_names_by_type(DataType.NUMERIC):
            values = dataset.X[column]
            nan_values_count = values.isnull().sum()

            mean = values.mean()
            if np.isnan(mean):
                mean = 0

            self.columns.append((column, mean))
            explain.append((
                nan_values_count,
                len(values),
                nan_values_count / len(values) * 100,
            ))

        self.explanations = [
            f"""Filled missing values of column **`{c}`** with **{mean:.2f}**
                (**{v[0]}** out of **{v[1]}** values (**{v[2]:.2f}**%)
                were missing in train data)."""
            for (c, mean), v in zip(self.columns, explain)
            if v[0] > 0 # hide processings that affected no values
        ]

        return self

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Fill NA values with the mean.

        :param pd.DataFrame x: DataFrame to transform
        :return: Transformed dataset
        """
        for name, mean in self.columns:
            X[name] = X[name].fillna(mean)

        return X

    def priorize(self, candidate:Candidate=None) -> float:
        return 1 - (candidate.dataset.X.isnull().sum().min() / len(candidate.dataset.X))
