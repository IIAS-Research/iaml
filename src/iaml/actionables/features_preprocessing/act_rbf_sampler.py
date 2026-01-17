"""[STEP] Decompose features with RBFSampler"""
from typing import Any
import textwrap
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('features_preprocessing')
class ActRBFSampler(Actionable):
    """[STEP] Approximate with RBFSampler"""

    name: str = "Approximate with RBFSampler"
    _usage: str = "Use when you want a fast nonlinear kernel approximation for numeric features, as a lighter alternative to ActKernelPCA. Applicable to dense tabular data where scaling is reasonable. Avoid when data are categorical heavy, very sparse, or when you need exact kernel features."
    _description: str = textwrap.dedent('''\
        RBFSampler is a tool that helps computers understand complex relationships
        between things by turning them into simpler numbers.''')
    _description_long: str = textwrap.dedent('''\
        RBFSampler is a machine learning technique that transforms data into
        a higher-dimensional space where it's easier for algorithms to find patterns.
        It works by creating random projections of the original data onto a new set of axes.
        This allows it to approximate the effects of a radial basis function kernel, which is a
        mathematical way of measuring similarity between data points.''')
    refs: list[dict[str, Any]] =[
        {
            'year': 2008,
            'name': 'Weighted Sums of Random Kitchen Sinks: Replacing minimization with \
                randomization in learning',
            'authors': [
                'Ali Rahimi',
                'Benjamin Recht'
            ],
            'doi': None,
            'publisher': 'Advances in Neural Information Processing Systems 21 page 1313--1320'
        }
    ]

    def __init__(self):
        self.configuration = {
            'n_components': {
                'description': 'Number of components to keep.',
                'default': 100,
                'range': [50, 10000]
            },
            'random_state': {
                'description': 'Random State',
                'default': 42
            }
        }

        self.optimizable: bool = True
        self.preprocessor: bool = None

    def fit(self, dataset: Dataset) -> Actionable:
        self.preprocessor = RBFSampler(**self.passthrough_parameters())
        self.preprocessor.fit(dataset.X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply RBFSampler

        :param pd.DataFrame X: DataFrame to transform
        :return: Transformed dataset
        """
        return pd.DataFrame(self.preprocessor.transform(X))

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5
