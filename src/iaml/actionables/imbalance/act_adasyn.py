"""[STEP] ADASYN"""
import textwrap
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('imbalance')
class ActADASYN(Actionable):
    """[STEP] ADASYN"""

    name: str = "ADASYN"
    _description: str = textwrap.dedent('''\
        ADASYN creates synthetic samples in hard to learn regions
        to adaptively balance classes.''')
    _description_long: str = textwrap.dedent('''\
        ADASYN (Adaptive Synthetic Sampling) focuses on minority samples that are
        difficult to learn. It generates more synthetic data where class overlap is
        higher, improving decision boundaries while keeping runtime small.''')
    _usage: str = "Use when numeric imbalance needs adaptive synthetic focus rather than ActSMOTE or ActRandomOverSampling. Applicable to binary or multiclass numeric features with sufficient minority samples. Avoid when categorical/text/date features exist or minority class is extremely small."
    refs: list[dict[str, Any]] = [
        {
            'year': 2008,
            'name': 'ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning',
            'authors': [
                'Haibo He',
                'Yang Bai',
                'Edwardo A. Garcia',
                'Sheng Ma'
            ],
            'doi': 'https://doi.org/10.1109/IJCNN.2008.4633969',
            'publisher': 'IEEE International Joint Conference on Neural Networks'
        }
    ]

    def __init__(self):
        self.configuration = {
            'sampling_strategy': {
                'description': 'Sampling strategy to balance classes.',
                'default': 'auto',
                'categorical': ['minority', 'auto']
            },
            'n_neighbors': {
                'description': 'Number of nearest neighbors used to create synthetic samples.',
                'default': 5,
                'range': [1, 20]
            },
            'random_state': {
                'description': 'Random seed used for reproducibility.',
                'default': 42
            }
        }
        self.resampler: ADASYN | None = None
        self.categorical_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self._effective_n_neighbors: int | None = None

    def fit(self, dataset: Dataset) -> Actionable:
        self.resampler = None
        self.categorical_columns = dataset.get_columns_names_by_type(DataType.CATEGORICAL)
        self.numeric_columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        self._effective_n_neighbors = None

        if dataset.X.empty or dataset.type_of_target not in ['binary', 'multiclass']:
            return self

        if dataset.y is None or len(dataset.y) == 0:
            return self

        unsupported = dataset.get_columns_names_by_type(
            [DataType.TEXT, DataType.SHORT_TEXT, DataType.DATE]
        )
        if unsupported:
            return self

        if self.categorical_columns:
            return self

        if not self.numeric_columns:
            return self

        _, counts = np.unique(dataset.y, return_counts=True)
        if len(counts) < 2:
            return self

        min_count = int(counts.min())
        if min_count <= 1:
            return self

        max_neighbors = min_count - 1
        n_neighbors = min(int(self.get_config('n_neighbors')), max_neighbors)
        n_neighbors = max(1, n_neighbors)
        self._effective_n_neighbors = n_neighbors

        params = self.passthrough_parameters()
        params['n_neighbors'] = n_neighbors

        self.resampler = ADASYN(**params)
        self.resampler.fit(dataset.X, dataset.y)
        return self

    def resample(self, X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply ADASYN.

        :param pd.DataFrame X: Features to resample
        :param pd.DataFrame y: Labels to resample
        :return: Resampled X and y
        """
        if self.resampler is None:
            return X, y
        return self.resampler.fit_resample(X, y)

    def priorize(self, candidate: Candidate = None) -> float:
        if candidate is None or candidate.dataset.y is None:
            return 0.0
        y = candidate.dataset.y
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        if len(counts) < 2:
            return 0.0
        imbalance = 1.0 - (counts.min() / counts.max())
        return float(min(1.0, max(0.0, imbalance)))

    def suitable(self, dataset: Dataset) -> bool:
        if dataset.type_of_target not in ['binary', 'multiclass']:
            return False
        if dataset.X.empty or dataset.y is None or len(dataset.y) == 0:
            return False
        if dataset.get_columns_names_by_type(DataType.CATEGORICAL):
            return False
        if not dataset.get_columns_names_by_type(DataType.NUMERIC):
            return False
        unsupported = dataset.get_columns_names_by_type(
            [DataType.TEXT, DataType.SHORT_TEXT, DataType.DATE]
        )
        if unsupported:
            return False
        _, counts = np.unique(dataset.y, return_counts=True)
        if len(counts) < 2:
            return False
        return counts.min() > 1
