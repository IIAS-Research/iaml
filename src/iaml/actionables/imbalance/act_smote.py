"""[STEP] SMOTE"""
import textwrap
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('imbalance')
class ActSMOTE(Actionable):
    """[STEP] SMOTE"""

    name: str = "SMOTE"
    _usage: str = "Use when imbalanced classification needs synthetic minority samples; compare ActADASYN for adaptive oversampling. Applicable to binary or multiclass with numeric features (categoricals via SMOTENC). Avoid when non-classification, text/date-only, or minority count <= 1."
    _description: str = textwrap.dedent('''\
        SMOTE balances the minority class by creating synthetic samples
        through interpolation.''')
    _description_long: str = textwrap.dedent('''\
        SMOTE (Synthetic Minority Over-sampling Technique) addresses class
        imbalance by generating new minority samples between nearest neighbors.
        This keeps the original data while reducing bias toward the majority class.''')
    refs: list[dict[str, Any]] = [
        {
            'year': 2002,
            'name': 'SMOTE: Synthetic Minority Over-sampling Technique',
            'authors': [
                'Nitesh V. Chawla',
                'Kevin W. Bowyer',
                'Lawrence O. Hall',
                'W. Philip Kegelmeyer'
            ],
            'doi': 'https://doi.org/10.1613/jair.953',
            'publisher': 'Journal of Artificial Intelligence Research Vol.16 page 321--357'
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
            'random_state': {
                'description': 'Random seed used for reproducibility.',
                'default': 42
            }
        }
        self.resampler: SMOTE | SMOTENC | None = None
        self.categorical_columns: list[str] = []
        self.categorical_indices: list[int] = []
        self.numeric_columns: list[str] = []
        self._effective_k_neighbors: int | None = None

    def fit(self, dataset: Dataset) -> Actionable:
        self.resampler = None
        self.categorical_columns = dataset.get_columns_names_by_type(DataType.CATEGORICAL)
        self.numeric_columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        self.categorical_indices = []
        self._effective_k_neighbors = None

        if dataset.X.empty or dataset.type_of_target not in ['binary', 'multiclass']:
            return self

        if dataset.y is None or len(dataset.y) == 0:
            return self

        unsupported = dataset.get_columns_names_by_type(
            [DataType.TEXT, DataType.SHORT_TEXT, DataType.DATE]
        )
        if unsupported:
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

        params = self.passthrough_parameters()
        params['k_neighbors'] = k_neighbors

        if self.categorical_columns:
            self.categorical_indices = [
                dataset.X.columns.get_loc(column)
                for column in self.categorical_columns
                if column in dataset.X.columns
            ]
            if self.categorical_indices:
                self.resampler = SMOTENC(
                    categorical_features=self.categorical_indices,
                    **params
                )
            else:
                self.resampler = SMOTE(**params)
        else:
            self.resampler = SMOTE(**params)

        self.resampler.fit(dataset.X, dataset.y)
        return self

    def resample(self, X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply SMOTE.

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
