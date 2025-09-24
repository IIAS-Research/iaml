"""[STEP] Fill missing values with mean"""
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
    """[STEP] Fill missing values with the mean."""

    name: str = 'Fill missing values'
    _description: str = textwrap.dedent('''\
        Fill missing values with the mean of non-missing values
        when the proportion of empty rows is lower than {empty_threshold:.0%}.''')
    _description_long: str = textwrap.dedent('''\
        Fill a column missings values with the mean of the columns
        when the proportion of empty rows is lower than {empty_threshold}.
        Work only for numerical columns.''')
    can_be_disabled: bool = False

    def __init__(self):
        self.columns: list[str] = None
        self.configuration:dict = {
            'empty_threshold': {
                'description': textwrap.dedent('''\
                    Column with less or equal proportion of empty row will be
                    fill with mean value. 1 will always fill void values'''),
                'default': 1 # TODO Review when adding new kind of imputer
            }
        }

    def fit(self, dataset: Dataset) -> Actionable:
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

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill NA values with the mean.

        :param pd.DataFrame x: DataFrame to transform.
        :return: Transformed dataset.
        """
        for name, mean in self.columns:
            X[name] = X[name].fillna(mean).infer_objects(copy=False)

        return X

    def priorize(self, candidate: Candidate = None) -> float:
        return 1 - (candidate.dataset.X.isnull().sum().min() / len(candidate.dataset.X))
