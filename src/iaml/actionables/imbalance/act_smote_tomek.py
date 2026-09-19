"""[STEP] SMOTETomek"""
import inspect
import textwrap
from typing import Any

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('imbalance')
class ActSMOTETomek(Actionable):
    """[STEP] SMOTETomek"""

    name: str = "SMOTE Tomek"
    _description: str = textwrap.dedent('''\
        SMOTETomek balances data by creating synthetic minority samples
        and removing Tomek links from overlapping classes.''')
    _description_long: str = textwrap.dedent('''\
        SMOTETomek combines SMOTE oversampling with Tomek links cleaning.
        It generates synthetic minority samples, then removes nearest neighbor
        pairs from different classes to reduce overlap and noise.''')
    _usage: str = "Use when imbalanced numeric data has overlap and want SMOTE plus Tomek cleanup vs ActSMOTE. Applicable to binary or multiclass, all-numeric features. Avoid when categorical/text/date fields exist, minority has <2 samples, or you want pure undersampling like ActRandomUnderSampler."
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
        },
        {
            'year': 1976,
            'name': 'Two Modifications of CNN',
            'authors': [
                'Ivan Tomek'
            ],
            'publisher': 'IEEE Transactions on Systems, Man, and Cybernetics Vol.6 No.11 '
                         'page 769--772'
        }
    ]

    def __init__(self):
        self.configuration = {
            'sampling_strategy': {
                'description': 'Sampling strategy to balance classes.',
                'default': 'auto',
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
        self.resampler: SMOTETomek | None = None
        self.categorical_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self._effective_k_neighbors: int | None = None

    def fit(self, dataset: Dataset) -> Actionable:
        self.resampler = None
        self.categorical_columns = dataset.get_columns_names_by_type(DataType.CATEGORICAL)
        self.numeric_columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
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

        smote_params = {
            'k_neighbors': k_neighbors
        }
        smote_sig = inspect.signature(SMOTE).parameters
        if 'sampling_strategy' in smote_sig:
            smote_params['sampling_strategy'] = self.get_config('sampling_strategy')
        if 'random_state' in smote_sig:
            smote_params['random_state'] = self.get_config('random_state')
        smote = SMOTE(**smote_params)

        tomek = TomekLinks()

        smotetomek_params: dict[str, Any] = {}
        smotetomek_sig = inspect.signature(SMOTETomek).parameters
        if 'smote' in smotetomek_sig:
            smotetomek_params['smote'] = smote
        if 'tomek' in smotetomek_sig:
            smotetomek_params['tomek'] = tomek
        if 'sampling_strategy' in smotetomek_sig and 'smote' not in smotetomek_params:
            smotetomek_params['sampling_strategy'] = self.get_config('sampling_strategy')
        if 'random_state' in smotetomek_sig and 'smote' not in smotetomek_params:
            smotetomek_params['random_state'] = self.get_config('random_state')

        self.resampler = SMOTETomek(**smotetomek_params)
        self.resampler.fit(dataset.X, dataset.y)
        return self

    def resample(self, X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply SMOTETomek.

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
