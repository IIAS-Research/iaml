"""[STEP] Random Over Sampling"""
import textwrap
from typing import Any
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('imbalance')
class ActRandomOverSampling(Actionable):
    """[STEP] Random Over Sampling"""

    name: str = "Random Over Sampling"
    _description: str = textwrap.dedent('''\
        RandomOverSampler is a tool that helps balance data by copying
        and pasting samples from minority groups.''')
    _description_long: str = textwrap.dedent('''\
        RandomOverSampler is a technique used to handle imbalanced datasets.
        It works by randomly copying and pasting samples from the minority class
        (the group with fewer samples) until it has the same number of samples as the majority class.
        This helps ensure that all classes are represented equally in the dataset.''')
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
        self.resampler: RandomOverSampler = None

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.resampler = RandomOverSampler(sampling_strategy='minority')
        self.resampler.fit(dataset.X, dataset.y)
        return self

    def resample(self, X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply Random Over Sampling

        :param pd.DataFrame X: Features to resample
        :param pd.DataFrame y: Labels to resample
        :return: Resampled X and y
        """
        return self.resampler.fit_resample(X, y)

    def priorize(self, candidate: Candidate = None) -> float:
        return 1

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target in ['binary', 'multiclass']
