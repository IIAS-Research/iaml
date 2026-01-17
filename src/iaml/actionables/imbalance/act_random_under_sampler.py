"""[STEP] Random Under Sampling"""
import inspect
import textwrap
from typing import Any

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('imbalance')
class ActRandomUnderSampler(Actionable):
    """[STEP] Random Under Sampling"""

    name: str = "Random Under Sampling"
    _description: str = textwrap.dedent('''\
        RandomUnderSampler balances data by randomly removing
        samples from the majority class.''')
    _description_long: str = textwrap.dedent('''\
        RandomUnderSampler reduces class imbalance by randomly dropping
        samples from the majority class until class sizes are closer.
        It is simple, fast, and keeps the original minority samples intact.''')
    _usage: str = "Use when fast random majority downsampling is acceptable; compare ActNearMiss for guided removal. Applicable to binary or multiclass tabular data with a clear majority class. Avoid when minority data is scarce or information loss hurts; consider ActRandomOverSampling."
    refs: list[dict[str, Any]] = [
        {
            'year': 2012,
            'name': 'Training and assessing classification rules with imbalanced data',
            'authors': [
                'Giovanna Menardi',
                'Nicola Torelli'
            ],
            'doi': 'https://doi.org/10.1007/s10618-012-0295-5',
            'publisher': 'Data Mining and Knowledge Discovery Vol.28 page 92--122'
        }
    ]

    def __init__(self):
        self.configuration = {
            'sampling_strategy': {
                'description': 'Sampling strategy to reduce the majority class.',
                'default': 'auto',
                'categorical': ['auto', 'majority']
            },
            'random_state': {
                'description': 'Random seed used for reproducibility.',
                'default': 42
            },
            'replacement': {
                'description': 'Sample with replacement when under-sampling.',
                'default': False,
                'categorical': [True, False]
            }
        }
        self.resampler: RandomUnderSampler | None = None
        self.feature_columns: list[str] = []
        self.imbalance_ratio: float = 0.0

    def fit(self, dataset: Dataset) -> Actionable:
        self.resampler = None
        self.feature_columns = self.__candidate_columns(dataset)
        self.imbalance_ratio = 0.0

        if dataset.X.empty or not self.feature_columns:
            return self
        if dataset.type_of_target not in ['binary', 'multiclass']:
            return self
        if dataset.y is None or len(dataset.y) == 0:
            return self

        _, counts = np.unique(dataset.y, return_counts=True)
        if len(counts) < 2:
            return self

        max_count = int(counts.max())
        min_count = int(counts.min())
        if max_count <= min_count:
            return self

        self.imbalance_ratio = 1.0 - (min_count / max_count)

        params = self.passthrough_parameters()
        sig_params = inspect.signature(RandomUnderSampler).parameters
        params = {key: value for key, value in params.items() if key in sig_params}

        self.resampler = RandomUnderSampler(**params)
        self.resampler.fit(dataset.X, dataset.y)
        return self

    def resample(self, X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply Random Under Sampling.

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
        if not self.__candidate_columns(dataset):
            return False
        _, counts = np.unique(dataset.y, return_counts=True)
        if len(counts) < 2:
            return False
        return counts.max() > counts.min()

    def __candidate_columns(self, dataset: Dataset) -> list[str]:
        columns = dataset.get_columns_names_by_type(list(DataType))
        if len(columns) != dataset.X.shape[1]:
            missing = [column for column in dataset.X.columns if column not in columns]
            columns.extend(missing)
        return columns
