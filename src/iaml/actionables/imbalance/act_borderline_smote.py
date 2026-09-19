"""[STEP] Borderline SMOTE"""
import inspect
import textwrap
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('imbalance')
class ActBorderlineSMOTE(Actionable):
    """[STEP] Borderline SMOTE"""

    name: str = "Borderline SMOTE"
    _description: str = textwrap.dedent('''\
        BorderlineSMOTE generates synthetic samples for minority points that
        lie close to the decision boundary.''')
    _description_long: str = textwrap.dedent('''\
        BorderlineSMOTE is a SMOTE variant that targets minority samples in
        the danger zone near the majority class. By focusing on boundary
        samples, it can reduce noise introduced by oversampling safe regions.''')
    _usage: str = "Use when minority points lie near the boundary and you want targeted oversampling vs ActSMOTE or ActRandomOverSampling. Applicable to numeric-only binary or multiclass datasets. Avoid when categorical/text/date features exist or the minority class is too small for neighbors."
    refs: list[dict[str, Any]] = [
        {
            'year': 2005,
            'name': 'Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning',
            'authors': [
                'Hui Han',
                'Wen-Yuan Wang',
                'Bing-Huan Mao'
            ],
            'publisher': 'ICIC 2005, Lecture Notes in Computer Science Vol.3644 page 878--887'
        }
    ]

    def __init__(self):
        self.configuration = {
            'sampling_strategy': {
                'description': 'Sampling strategy to balance classes.',
                'default': 'minority',
                'categorical': ['minority', 'auto']
            },
            'k_neighbors': {
                'description': 'Number of nearest neighbors used to create synthetic samples.',
                'default': 5,
                'range': [1, 20]
            },
            'm_neighbors': {
                'description': 'Number of neighbors used to detect borderline samples.',
                'default': 10,
                'range': [1, 20]
            },
            'kind': {
                'description': 'Borderline variant to use when supported by imbalanced-learn.',
                'default': 'borderline-1',
                'categorical': ['borderline-1', 'borderline-2'],
                'passthrough': False
            },
            'random_state': {
                'description': 'Random seed used for reproducibility.',
                'default': 42
            }
        }
        self.resampler: BorderlineSMOTE | None = None
        self.categorical_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self._effective_k_neighbors: int | None = None
        self._effective_m_neighbors: int | None = None

    def fit(self, dataset: Dataset) -> Actionable:
        self.resampler = None
        self.categorical_columns = dataset.get_columns_names_by_type(DataType.CATEGORICAL)
        self.numeric_columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        self._effective_k_neighbors = None
        self._effective_m_neighbors = None

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

        max_k = min_count - 1
        k_neighbors = min(int(self.get_config('k_neighbors')), max_k)
        k_neighbors = max(1, k_neighbors)
        self._effective_k_neighbors = k_neighbors

        total_count = int(len(dataset.y))
        max_m = max(1, total_count - 1)
        m_neighbors = min(int(self.get_config('m_neighbors')), max_m)
        m_neighbors = max(1, m_neighbors)
        self._effective_m_neighbors = m_neighbors

        params = self.passthrough_parameters()
        params['k_neighbors'] = k_neighbors
        params['m_neighbors'] = m_neighbors
        if 'kind' in inspect.signature(BorderlineSMOTE).parameters:
            params['kind'] = self.get_config('kind')

        self.resampler = BorderlineSMOTE(**params)
        self.resampler.fit(dataset.X, dataset.y)
        return self

    def resample(self, X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply Borderline SMOTE.

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
