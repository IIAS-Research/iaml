"""[STEP] Normalizer"""
import textwrap
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('normalize')
class ActNormalizer(Actionable):
    """[STEP] Normalizer"""

    name: str = "Normalizer"
    _description: str = textwrap.dedent('''\
        Normalizer scales each sample so its L1 or L2 norm equals one,
        keeping per-sample magnitudes comparable.''')
    _description_long: str = textwrap.dedent('''\
        Normalizer rescales each row independently by dividing its values
        by the L1 or L2 norm. This preserves the direction of each sample
        while making their magnitudes comparable, which is helpful when
        features represent frequencies, counts, or embeddings.''')
    _usage: str = "Use when you need per-sample L1/L2 normalization for row magnitude comparability, especially for counts or embeddings. Applicable to numeric features where each row should be unit norm. Avoid when feature-wise scaling is needed; consider ActMinMaxScaler or ActRobustScaler."

    def __init__(self):
        self.columns: list[str] = None
        self.normalizer: Normalizer = None

        self.configuration = {
            'norm': {
                'description': 'Normalization type to apply to each sample.',
                'default': 'l2',
                'categorical': ['l1', 'l2']
            },
            'copy': {
                'description': 'Set to False to perform normalization in-place when possible.',
                'default': True,
                'categorical': [True, False]
            }
        }

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if self.columns and not dataset.X.empty:
            values = dataset.X[self.columns]
            self.normalizer = Normalizer(**self.passthrough_parameters())
            self.normalizer.fit(values)
        else:
            self.normalizer = None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization

        :param pd.DataFrame X: DataFrame to transform
        :return: Transformed dataset
        """
        if self.normalizer and self.columns:
            X[self.columns] = self.normalizer.transform(X[self.columns])
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
        if values.empty:
            return 0.0
        matrix = values.to_numpy(dtype=float, copy=True)
        if np.isnan(matrix).any():
            matrix = np.nan_to_num(matrix, nan=0.0)
        if self.get_config('norm') == 'l1':
            norms = np.sum(np.abs(matrix), axis=1)
        else:
            norms = np.linalg.norm(matrix, axis=1)
        if norms.size == 0:
            return 0.0
        nonzero = norms > 0
        if not nonzero.any():
            return 0.0
        norms = norms[nonzero]
        mean_deviation = float(np.mean(np.abs(norms - 1.0)))
        return min(1.0, mean_deviation)
