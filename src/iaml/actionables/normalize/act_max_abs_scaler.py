"""[STEP] Max Abs Scaler"""
import textwrap
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step
from ...data_type import DataType


@is_step('normalize')
class ActMaxAbsScaler(Actionable):
    """[STEP] Max Abs Scaler"""

    name: str = "Max Abs Scaler"
    _description: str = textwrap.dedent('''\
        MaxAbsScaler rescales numeric data by dividing by the maximum absolute value,
        keeping values within a [-1, 1] range.''')
    _description_long: str = textwrap.dedent('''\
        MaxAbsScaler scales each numeric feature by its maximum absolute value
        observed in the training data. This keeps values within [-1, 1] while
        preserving sparsity because it does not center the data.
        It is a good fit for sparse datasets where zeros should remain zeros.''')
    _usage: str = "Use when numeric features are sparse and you want scale to [-1, 1] without centering; compare ActMinMaxScaler. Applicable to numeric data with many zeros or sparse matrices. Avoid when you need centering or heavy outlier handling; consider ActNormalizer or ActRobustScaler."

    def __init__(self):
        self.columns: list[str] = None
        self.scaler: MaxAbsScaler = None

        self.configuration = {
            'copy': {
                'description': 'Set to False to perform scaling in-place when possible.',
                'default': True,
                'categorical': [True, False]
            }
        }

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if self.columns and not dataset.X.empty:
            values = dataset.X[self.columns]
            self.scaler = MaxAbsScaler(**self.passthrough_parameters())
            self.scaler.fit(values)
        else:
            self.scaler = None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply max abs scaler

        :param pd.DataFrame X: DataFrame to transform
        :return: Transformed dataset
        """
        if self.scaler and self.columns:
            X[self.columns] = self.scaler.transform(X[self.columns])
        return X

    def suitable(self, dataset: Dataset) -> bool:
        columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        return bool(columns) and not dataset.X.empty

    def priorize(self, candidate: Candidate = None) -> float:
        if candidate is None:
            return 0.0
        columns = candidate.dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not columns or candidate.dataset.X.empty:
            return 0.0
        values = candidate.dataset.X[columns]
        total = values.size - values.isna().sum().sum()
        if total <= 0:
            return 0.0
        zeros = (values == 0).sum().sum()
        zero_ratio = zeros / total
        return min(1.0, zero_ratio * 1.5)
