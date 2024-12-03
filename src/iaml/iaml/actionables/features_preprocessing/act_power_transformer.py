"""
[STEP] Preprocess with PowerTransformer
"""
from typing import Any
import textwrap
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('features_preprocessing')
class ActPowerTransformer(Actionable):
    """[STEP] Preprocess with PowerTransformer"""

    name: str = "Preprocess with PowerTransformer"
    _description: str = textwrap.dedent('''\
        PowerTransformer changes data to make it more "bell-curve" shaped.
        It uses special math tricks to flatten out irregular distributions and make the data
        behave more like a normal distribution.''')
    _description_long: str = textwrap.dedent('''\
        PowerTransformer is a preprocessing technique that applies a power
        transformation to make data more Gaussian-like. PowerTransformer is useful when you want to apply
        machine learning models that assume normal distribution,
        even if your original data doesn't meet this assumption.
        It helps make your data more compatible with many common ML algorithms.''')
    refs: list[dict[str, Any]] = [
        {
            'year': 1964,
            'name': 'An Analysis of Transformations',
            'authors': [
                'G. E. P. Box',
                'D. R. Cox'
            ],
            'doi': 'https://doi.org/10.1111/j.2517-6161.1964.tb00553.x',
            'publisher': 'Journal of the Royal Statistical Society: Series B (Methodological), \
                Vol.26, No.2 page 211--243'
        },
        {
            'year': 2000,
            'name': 'A New Family of Power Transformations to Improve Normality or Symmetry',
            'authors': [
                'In-Kwon Yeo',
                'Richard A. Johnson'
            ],
            'doi': 'https://doi.org/10.1093/biomet/87.4.954',
            'publisher': 'Oxford University Press, Biometrika Vol.87 No.4 page 954--959'
        },
    ]

    def __init__(self):
        self.configuration = {
            'method': {
                'description': 'The power transform method.',
                'default': 'yeo-johnson',
                'categorical': ['yeo-johnson', 'box-cox']
                },
            'standardize': {
                'description': 'Set to True to apply zero-mean, \
                    unit-variance normalization to the transformed output.',
                'default': True
                }
            }

        self.optimizable = True
        self.preprocessor = None

    def fit(self, dataset: Dataset) -> Actionable:
        self.preprocessor = PowerTransformer(**self.passthrough_parameters())
        self.preprocessor.fit(dataset.X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply PowerTransformer

        :param pd.DataFrame X: DataFrame to transform
        :return: Transformed dataset
        """

        return pd.DataFrame(self.preprocessor.transform(X))

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5

    def suitable(self, dataset: Dataset) -> bool:
        if self.get_config('method') == 'box-cox' and not(dataset.X < 0).any().any():
            self.configure('method', 'yeo-johnson')

        return True
