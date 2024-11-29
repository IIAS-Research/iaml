"""
[STEP] Drop Numerical Column
"""
import textwrap
import pandas as pd
from ...actionable import Actionable
from ...dataset import Dataset
from ...data_type import DataType
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('cleaning')
class ActDropNumericalColumn(Actionable):
    """
    [STEP] Drop Numerical Column
    """
    name = 'Remove numerical columns'
    description = textwrap.dedent('''\
        Remove numerical columns where the proportion of empty rows
        in the dataset is higher than {empty_threshold}.''')
    description_long = textwrap.dedent('''\
        Remove numerical columns from the dataset where the proportion of empty
        rows in the dataset is higher than {empty_threshold}. This ensure that every columns will
        be relevant for the model to train on.''')

    def __init__(self):
        self.columns_to_drop: list[str] = None
        self.configuration = {
            'empty_threshold': {
                'description': 'Column with more or equal proportion of empty row \
                    will dropped. 1 will drop all columns',
                'default': 0.5
            }
        }

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns_to_drop = []
        explain = []

        for column in dataset.get_columns_names_by_type(DataType.NUMERIC):
            values = dataset.X[column]
            nan_values_count = values.isnull().sum()

            if nan_values_count / len(values) >= self.get_config('empty_threshold'):
                self.columns_to_drop.append(column)
                explain.append((nan_values_count, len(values)))

        self.explanations = [
            f"""Dropped column **`{c}`** because **{v[0]}** values out of
                **{v[1]}** (**{(v[0] / v[1] * 100):.2f}%**) are empty."""
            for c, v in zip(self.columns_to_drop, explain)
        ]

        return self

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns.

        :param pd.DataFrame X: DataFrame to transform
        :return: Transformed DataFrame
        """
        return X.drop(self.columns_to_drop, axis=1)

    def priorize(self, candidate:Candidate=None) -> float:
        return 0

    def suitable(self, dataset:Dataset) -> bool:
        for column in dataset.get_columns_names_by_type(DataType.NUMERIC):
            values = dataset.X[column]
            nan_values_count = values.isnull().sum()

            if nan_values_count / len(values) >= self.get_config('empty_threshold'):
                return True

        return False
