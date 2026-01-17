"""[STEP] Robust Scaler"""
import textwrap
import pandas as pd
from sklearn.preprocessing import RobustScaler
from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('normalize')
class ActRobustScaler(Actionable):
    """[STEP] Robust Scaler"""

    name: str = "Robust Scaler"
    _usage: str = "Use when numeric features have outliers or skew and you want robust scaling vs ActMinMaxScaler or ActMaxAbsScaler. Applicable to continuous numeric columns. Avoid when you need unit-norm vectors or the data has no outliers."
    _description: str = textwrap.dedent('''\
        RobustScaler centers and scales numeric data using the median and IQR
        to reduce the impact of outliers.''')
    _description_long: str = textwrap.dedent('''\
        RobustScaler is a scaling technique that uses the median to center each
        feature and the interquartile range (IQR) to scale it.
        Because these statistics are resilient to extreme values, the transformation
        is well suited for data sets that contain outliers.''')

    def __init__(self):
        self.columns: list[str] = None
        self.scaler: RobustScaler = None

        self.configuration = {
            'with_centering': {
                'description': 'Center data before scaling.',
                'default': True,
                'categorical': [True, False]
            },
            'with_scaling': {
                'description': 'Scale data to the IQR.',
                'default': True,
                'categorical': [True, False]
            },
            'quantile_range_low': {
                'description': 'Lower quantile used to compute the IQR.',
                'default': 25.0,
                'range': [0.0, 50.0]
            },
            'quantile_range_high': {
                'description': 'Upper quantile used to compute the IQR.',
                'default': 75.0,
                'range': [50.0, 100.0]
            },
            'unit_variance': {
                'description': 'Scale data so that scaled features have unit variance.',
                'default': False,
                'categorical': [True, False]
            }
        }

    def _build_scaler(self) -> RobustScaler:
        params = self.passthrough_parameters()
        low = float(params.pop('quantile_range_low'))
        high = float(params.pop('quantile_range_high'))
        if low >= high:
            low, high = 25.0, 75.0
        params['quantile_range'] = (low, high)
        return RobustScaler(**params)

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if self.columns:
            values = dataset.X[self.columns]
            self.scaler = self._build_scaler()
            self.scaler.fit(values)
        else:
            self.scaler = None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply robust scaler

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
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if (iqr == 0).all():
            return 0.1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((values < lower) | (values > upper)).sum().sum()
        total = values.size or 1
        return min(1.0, outliers / total * 5.0)
