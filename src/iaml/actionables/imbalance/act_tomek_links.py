"""[STEP] Tomek Links"""
import inspect
import textwrap
from typing import Any

import numpy as np
import pandas as pd
from imblearn.under_sampling import TomekLinks

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('imbalance')
class ActTomekLinks(Actionable):
    """[STEP] Tomek Links"""

    name: str = "Tomek Links"
    _description: str = textwrap.dedent('''\
        TomekLinks cleans class boundaries by removing samples that form
        nearest-neighbor pairs across classes.''')
    _description_long: str = textwrap.dedent('''\
        TomekLinks identifies pairs of samples from different classes that are
        each other's nearest neighbors (Tomek links). Removing the majority
        samples in these pairs reduces overlap and cleans noisy boundaries.''')
    _usage: str = "Use when you want light boundary cleaning instead of heavier ActNearMiss. Applicable to numeric-only binary or multiclass data. Avoid when data includes categorical/text/date or you need to add samples (ActSMOTE)."
    refs: list[dict[str, Any]] = [
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
                'description': 'Sampling strategy to remove Tomek link pairs.',
                'default': 'auto',
                'categorical': ['auto', 'majority', 'all']
            }
        }
        self.resampler: TomekLinks | None = None
        self.categorical_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self.imbalance_ratio: float = 0.0

    def fit(self, dataset: Dataset) -> Actionable:
        self.resampler = None
        self.categorical_columns = dataset.get_columns_names_by_type(DataType.CATEGORICAL)
        self.numeric_columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        self.imbalance_ratio = 0.0

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
        if max_count <= 0 or min_count <= 0:
            return self

        self.imbalance_ratio = 1.0 - (min_count / max_count)

        params = self.passthrough_parameters()
        sig_params = inspect.signature(TomekLinks).parameters
        params = {key: value for key, value in params.items() if key in sig_params}

        self.resampler = TomekLinks(**params)
        self.resampler.fit(dataset.X, dataset.y)
        return self

    def resample(self, X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply Tomek Links.

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
        return True
