"""[STEP] Near Miss"""
import inspect
import textwrap
from typing import Any

import numpy as np
import pandas as pd
from imblearn.under_sampling import NearMiss

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('imbalance')
class ActNearMiss(Actionable):
    """[STEP] Near Miss"""

    name: str = "Near Miss"
    _description: str = textwrap.dedent('''\
        NearMiss under-samples by keeping majority samples
        that are closest to the minority class.''')
    _description_long: str = textwrap.dedent('''\
        NearMiss reduces class imbalance by selecting majority samples near
        the minority class. Different versions use nearest-neighbor distances
        to keep samples that are harder to separate from the minority class.''')
    _usage: str = "Use when you want distance-based under-sampling on numeric data; compare ActRandomUnderSampler or ActSMOTE. Applicable to imbalanced binary or multiclass numeric datasets. Avoid when you have categorical/text/date features or need to retain most majority samples."
    refs: list[dict[str, Any]] = []

    def __init__(self):
        self.configuration = {
            'sampling_strategy': {
                'description': 'Sampling strategy to reduce the majority class.',
                'default': 'auto',
                'categorical': ['auto', 'majority']
            },
            'version': {
                'description': 'NearMiss version to use (1, 2, or 3).',
                'default': 1,
                'categorical': [1, 2, 3]
            },
            'n_neighbors': {
                'description': 'Number of minority neighbors used to compute distances.',
                'default': 3,
                'range': [1, 20]
            },
            'n_neighbors_ver3': {
                'description': 'Number of majority neighbors kept per minority sample for version 3.',
                'default': 3,
                'range': [1, 20]
            }
        }
        self.resampler: NearMiss | None = None
        self.categorical_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self.imbalance_ratio: float = 0.0
        self._effective_n_neighbors: int | None = None
        self._effective_n_neighbors_ver3: int | None = None

    def fit(self, dataset: Dataset) -> Actionable:
        self.resampler = None
        self.categorical_columns = dataset.get_columns_names_by_type(DataType.CATEGORICAL)
        self.numeric_columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        self.imbalance_ratio = 0.0
        self._effective_n_neighbors = None
        self._effective_n_neighbors_ver3 = None

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

        max_count = int(counts.max())
        min_count = int(counts.min())
        if max_count <= min_count or min_count <= 0:
            return self

        self.imbalance_ratio = 1.0 - (min_count / max_count)

        n_neighbors = min(int(self.get_config('n_neighbors')), min_count)
        n_neighbors = max(1, n_neighbors)
        self._effective_n_neighbors = n_neighbors

        n_neighbors_ver3 = min(int(self.get_config('n_neighbors_ver3')), max_count)
        n_neighbors_ver3 = max(1, n_neighbors_ver3)
        self._effective_n_neighbors_ver3 = n_neighbors_ver3

        params = self.passthrough_parameters()
        params['n_neighbors'] = n_neighbors
        params['n_neighbors_ver3'] = n_neighbors_ver3

        sig_params = inspect.signature(NearMiss).parameters
        params = {key: value for key, value in params.items() if key in sig_params}

        self.resampler = NearMiss(**params)
        self.resampler.fit(dataset.X, dataset.y)
        return self

    def resample(self, X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply NearMiss.

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
        return counts.max() > counts.min()
