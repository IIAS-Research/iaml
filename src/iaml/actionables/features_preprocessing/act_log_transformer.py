"""[STEP] Apply log1p to skewed numeric features."""
import textwrap
import numpy as np
import pandas as pd
from ...actionable import Actionable
from ...candidate import Candidate
from ...dataset import Dataset
from ...data_type import DataType
from ...decorators.all import is_step


@is_step('features_preprocessing')
class ActLogTransformer(Actionable):
    """[STEP] Apply log1p to skewed numeric features."""

    name: str = "Log1p Transformer"
    _description: str = "Apply log1p to highly skewed numeric columns"
    _usage: str = "Use when numeric features are highly right-skewed and >= -1, often before ActKBinsDiscretizer or ActKernelPCA. Applicable to continuous numeric columns with long-tailed distributions. Avoid when values are <= -1 or already log/scale transformed."
    _description_long: str = textwrap.dedent('''\
        This step detects numeric features with strong positive skew
        and applies a log1p (log(1+x)) transformation to compress
        extreme values. The transformation is automatic and only
        applied to columns that meet the skewness threshold and have
        values above the configured minimum.
    ''')

    def __init__(self):
        self.columns: list[str] = []

        self.configuration = {
            'skew_threshold': {
                'description': 'Minimum skewness required to apply log1p.',
                'default': 1.5,
                'range': [0.5, 5.0]
            },
            'min_value': {
                'description': 'Minimum allowed value for applying log1p.',
                'default': 0.0,
                'range': [-0.99, 1.0]
            }
        }

        self.optimizable: bool = True

    @staticmethod
    def _coerce_float(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _resolve_skew_threshold(self) -> float:
        threshold = self._coerce_float(self.get_config('skew_threshold'), 1.5)
        if threshold < 0:
            threshold = 0.0
        return threshold

    def _resolve_min_value(self) -> float:
        min_value = self._coerce_float(self.get_config('min_value'), 0.0)
        if min_value <= -1.0:
            min_value = -0.999
        return min_value

    def _select_columns(self, values: pd.DataFrame) -> list[str]:
        if values.empty:
            return []
        skewness = values.skew().fillna(0.0)
        min_values = values.min(skipna=True)
        threshold = self._resolve_skew_threshold()
        min_value = self._resolve_min_value()
        eligible = (skewness >= threshold) & (min_values >= min_value)
        return [col for col in values.columns if bool(eligible.get(col, False))]

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = []
        columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not columns or dataset.X.empty:
            return self
        values = dataset.X[columns]
        self.columns = self._select_columns(values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply log1p

        :param pd.DataFrame X: DataFrame to transform
        :return: Transformed dataset
        """
        if not self.columns:
            return X
        columns = [col for col in self.columns if col in X.columns]
        if not columns:
            return X
        min_value = self._resolve_min_value()
        values = X[columns].clip(lower=min_value)
        X[columns] = np.log1p(values)
        return X

    def suitable(self, dataset: Dataset) -> bool:
        columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not columns or dataset.X.empty:
            return False
        values = dataset.X[columns]
        return bool(self._select_columns(values))

    def priorize(self, candidate: Candidate = None) -> float:
        if candidate is None:
            return 0.0
        dataset = candidate.dataset
        columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not columns or dataset.X.empty:
            return 0.0
        values = dataset.X[columns]
        selected = self._select_columns(values)
        if not selected:
            return 0.0
        skewness = values[selected].skew().fillna(0.0)
        threshold = self._resolve_skew_threshold()
        if threshold <= 0:
            return 1.0
        mean_skew = float(skewness.mean()) if not skewness.empty else 0.0
        return min(1.0, mean_skew / (threshold * 2.0))
