"""[STEP] SMOTEENN"""
import inspect
import textwrap
from typing import Any

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('imbalance')
class ActSMOTEENN(Actionable):
    """[STEP] SMOTEENN"""

    name: str = "SMOTE ENN"
    _description: str = textwrap.dedent('''\
        SMOTEENN balances data by creating synthetic minority samples and
        removing ambiguous samples with Edited Nearest Neighbors.''')
    _description_long: str = textwrap.dedent('''\
        SMOTEENN combines SMOTE oversampling with Edited Nearest Neighbors
        cleaning. It adds synthetic minority samples, then removes samples
        that disagree with their neighbors to reduce noise and class overlap.''')
    _usage: str = "Use when imbalance with noisy borders needs SMOTE plus cleaning vs ActSMOTE. Applicable to numeric-only binary or multiclass data with >=2 samples per class. Avoid when categorical/text features, very small minorities, or you want pure under-sampling like ActNearMiss."
    refs: list[dict[str, Any]] = [
        {
            'year': 2004,
            'name': 'A Study of the Behavior of Several Methods for Balancing '
                    'Machine Learning Training Data',
            'authors': [
                'Gustavo E. A. P. A. Batista',
                'Ronaldo C. Prati',
                'Maria Carolina Monard'
            ],
            'doi': 'https://doi.org/10.1145/1007730.1007735',
            'publisher': 'ACM SIGKDD Explorations Newsletter Vol.6 No.1 page 20--29'
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
            'n_neighbors': {
                'description': 'Number of neighbors used by Edited Nearest Neighbors.',
                'default': 3,
                'range': [1, 20]
            },
            'kind_sel': {
                'description': 'Rule used by Edited Nearest Neighbors to select samples.',
                'default': 'all',
                'categorical': ['all', 'mode'],
                'passthrough': False
            },
            'random_state': {
                'description': 'Random seed used for reproducibility.',
                'default': 42
            }
        }
        self.resampler: SMOTEENN | None = None
        self.categorical_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self._effective_k_neighbors: int | None = None
        self._effective_n_neighbors: int | None = None

    def fit(self, dataset: Dataset) -> Actionable:
        self.resampler = None
        self.categorical_columns = dataset.get_columns_names_by_type(DataType.CATEGORICAL)
        self.numeric_columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        self._effective_k_neighbors = None
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

        max_k = min_count - 1
        k_neighbors = min(int(self.get_config('k_neighbors')), max_k)
        k_neighbors = max(1, k_neighbors)
        self._effective_k_neighbors = k_neighbors

        total_count = int(len(dataset.y))
        max_neighbors = max(1, total_count - 1)
        n_neighbors = min(int(self.get_config('n_neighbors')), max_neighbors)
        n_neighbors = max(1, n_neighbors)
        self._effective_n_neighbors = n_neighbors

        smote = SMOTE(
            sampling_strategy=self.get_config('sampling_strategy'),
            k_neighbors=k_neighbors,
            random_state=self.get_config('random_state')
        )

        enn_params = {
            'n_neighbors': n_neighbors
        }
        if 'kind_sel' in inspect.signature(EditedNearestNeighbours).parameters:
            enn_params['kind_sel'] = self.get_config('kind_sel')
        enn = EditedNearestNeighbours(**enn_params)

        smoteenn_params = {}
        smoteenn_sig = inspect.signature(SMOTEENN).parameters
        if 'smote' in smoteenn_sig:
            smoteenn_params['smote'] = smote
        if 'enn' in smoteenn_sig:
            smoteenn_params['enn'] = enn
        if 'sampling_strategy' in smoteenn_sig and 'smote' not in smoteenn_params:
            smoteenn_params['sampling_strategy'] = self.get_config('sampling_strategy')
        if 'random_state' in smoteenn_sig and 'smote' not in smoteenn_params:
            smoteenn_params['random_state'] = self.get_config('random_state')

        self.resampler = SMOTEENN(**smoteenn_params)
        self.resampler.fit(dataset.X, dataset.y)
        return self

    def resample(self, X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply SMOTEENN.

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
