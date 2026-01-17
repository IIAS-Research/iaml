"""[STEP] VIF Selector"""
import textwrap

import numpy as np
import pandas as pd

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('features_selection')
class ActVIFSelector(Actionable):
    """[STEP] VIF Selector"""

    name: str = 'VIF Selector'
    _description: str = textwrap.dedent('''\
        Drop numeric columns with VIF above {threshold}.''')
    _usage: str = "Use when reducing multicollinearity among numeric predictors, as a simpler alternative to ActRFE or ActSelectFromModel. Applicable to numeric tabular data with correlated features. Avoid when features are non-numeric, too few columns, or you need target-driven selection."
    _description_long: str = textwrap.dedent('''\
        Variance Inflation Factor (VIF) measures multicollinearity among
        numeric predictors. This step computes VIF from the correlation
        matrix and removes columns whose VIF exceeds the configured
        threshold to improve model stability.''')

    def __init__(self):
        self.configuration = {
            'threshold': {
                'description': 'Drop columns with VIF greater than this value.',
                'default': 10.0,
                'range': [1.0, 1000.0]
            }
        }
        self.columns: list[str] = []
        self.to_drop: list[str] = []
        self.vif_scores: dict[str, float] = {}

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        self.to_drop = []
        self.vif_scores = {}
        self.explanations = []

        if len(self.columns) < 2 or dataset.X.empty:
            return self

        self.to_drop, self.vif_scores = self.__get_columns(dataset)

        threshold = self.get_config('threshold')
        for column in self.to_drop:
            vif_value = self.vif_scores.get(column)
            if vif_value is None or not np.isfinite(vif_value):
                message = (
                    f"Dropped column **`{column}`** because its VIF was not finite."
                )
            else:
                message = (
                    f"Dropped column **`{column}`** because its VIF ({vif_value:.2f}) "
                    f"exceeded {threshold}."
                )
            self.explanations.append(message)

        return self

    def __get_columns(self, dataset: Dataset) -> tuple[list[str], dict[str, float]]:
        columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if len(columns) < 2:
            return [], {}

        X_numeric = dataset.X[columns]
        if X_numeric.empty:
            return [], {}

        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
        X_numeric = X_numeric.dropna(axis=0, how='any')
        if X_numeric.shape[0] < 2:
            return [], {}

        variances = X_numeric.var(ddof=0)
        eligible_columns = variances[variances > 0].index.tolist()
        if len(eligible_columns) < 2:
            return [], {}

        corr = X_numeric[eligible_columns].corr()
        if corr.empty:
            return [], {}

        inv_corr = np.linalg.pinv(corr.values)
        vif_values = np.diag(inv_corr)
        vif_values = np.maximum(vif_values, 1.0)

        vifs = {
            column: float(vif)
            for column, vif in zip(eligible_columns, vif_values)
        }

        threshold = self.get_config('threshold')
        to_drop = [
            column for column, vif in vifs.items()
            if not np.isfinite(vif) or vif > threshold
        ]

        return to_drop, vifs

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop columns with high VIF from the DataFrame.

        :param pd.DataFrame X: The dataset to transform.
        :return: Transformed dataset without high VIF columns.
        """
        if not self.to_drop:
            return X

        drop_cols = [column for column in self.to_drop if column in X.columns]
        if not drop_cols:
            return X

        return X.drop(columns=drop_cols)

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5

    def suitable(self, dataset: Dataset) -> bool:
        columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if len(columns) < 2 or dataset.X.empty:
            return False

        return bool(self.__get_columns(dataset)[0])
