"""
[STEP] Decompose features with RBFSampler
"""
import textwrap
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('features_preprocessing')
class ActRBFSampler(Actionable):
    """
    [STEP] Approximate with RBFSampler
    """
    name = "Approximate with RBFSampler"
    _description = textwrap.dedent('''\
        RBFSampler is a tool that helps computers understand complex relationships
        between things by turning them into simpler numbers.''')
    _description_long = textwrap.dedent('''\
        RBFSampler is a machine learning technique that transforms data into
        a higher-dimensional space where it's easier for algorithms to find patterns.
        It works by creating random projections of the original data onto a new set of axes.
        This allows it to approximate the effects of a radial basis function kernel, which is a
        mathematical way of measuring similarity between data points.''')
    refs=[
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
        
        self.optimizable = True
        self.preprocessor = None

    def fit(self, dataset: Dataset) -> Actionable:
        """
        Fit Features agglomerations

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        
        self.preprocessor = RBFSampler(**self.passthrough_parameters())
        self.preprocessor.fit(dataset.X)
        
        return self
    
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply RBFSampler

        Args:
            x (pd.DataFrame): DataFrame to transform

        Returns:
            pd.DataFrame: Transformed dataset
        """
        return pd.DataFrame(self.preprocessor.transform(X))
        
    
    def priorize(self, candidate: Candidate = None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5
