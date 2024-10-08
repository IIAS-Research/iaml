"""
[STEP] Decompose features with KernelPCA
"""
import pandas as pd
from sklearn.decomposition import KernelPCA
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step
import textwrap

@is_step('features_preprocessing')
class ActKernelPCA(Actionable):
    """
    [STEP] Decompose features with KernelPCA
    """
    name = textwrap.dedent('Decompose features with KernelPCA')
    description = textwrap.dedent('''Reduce datasets number of features by using
        Kernel Principal Component Analysis Algorithm.''')
    description_long = textwrap.dedent('''The dataset is reduced to {n_components} components
        using {kernel} kernel''')
    refs = [
        {
            'year': 1997,
            'name': 'Kernel principal component analysis',
            'authors': [
                'Bernhard Schölkopf',
                'Alexander Smola',
                'Klaus-Robert Müller'    
            ],
            'doi': 'https://doi.org/10.1007/BFb0020217',
            'publisher': 'Springer, Berlin, Heidelberg'
        },
        {
            'year': 2003,
            'name': 'Learning to find pre-images',
            'authors': [
                'Jason Weston',
                'Bernhard Schölkopf',
                'Gökhan Bakir'
            ],
            'doi': None,
            'publisher': 'Advances in neural information processing systems 16 (2004) page 449--456'
        },
        {
            'year': 2009,
            'name': 'Finding structure with randomness: Probabilistic algorithms for constructing \
                approximate matrix decompositions',
            'authors': [
                'Nathan Halko',
                'Per-Gunnar Martinsson',
                'Joel A. Tropp'
            ],
            'doi': 'https://doi.org/10.48550/arXiv.0909.4061',
            'publisher': 'SIAM Rev., Survey and Review section, Vol.53, No.2 page 217--288'
        },
        {
            'year': 2011,
            'name': 'A randomized algorithm for the decomposition of matrices',
            'authors': [
                'Per-Gunnar Martinsson',
                'Vladimir Rokhlin',
                'Mark Tygert'
            ],
            'doi': 'https://doi.org/10.1016/j.acha.2010.02.003',
            'publisher': 'Applied and Computational Harmonic Analysis, Vol.30, No.1 page 47--68'
        }
    ]
    def __init__(self):
        self.configuration:dict = {
            'kernel': {
                'description': 'Kernel used for PCA.',
                'default': 'rbf',
                'categorical': ['poly', 'rbf', 'sigmoid', 'cosine']
                },
            'n_components': {
                'description': 'Number of components to keep.',
                'default': 100,
                'range': [10, 2000]
                },
            'coef0': {
                'description': 'Independent term in poly and sigmoid kernels. \
                    Ignored by other kernels.',
                'default': 1.0,
                'range': [-1.0, 1.0]
                },
            'degree': {
                'description': 'Degree for poly kernels. Ignored by other kernels.',
                'default': 3,
                'range': [2, 5]
                },
            'random_state': {
                'description': 'Random State',
                'default': 42
                }
            }
        
        self.optimizable = True
        self.preprocessor = None


    def fit(self, dataset:Dataset) -> Actionable:
        """
        Find columns to convert

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        
        
        try:
            self.preprocessor = KernelPCA(**self.passthrough_parameters())
            self.preprocessor.fit(dataset.X)
        except ValueError: 
            higher_gamma = 1/dataset.X.shape[1] + 0.05
            self.preprocessor = KernelPCA(gamma=higher_gamma, **self.passthrough_parameters())
            self.preprocessor.fit(dataset.X)
            
        return self
    
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Apply KernelPCA

        Args:
            x (pd.DataFrame): DataFrame to transform

        Returns:
            pd.DataFrame: Transformed dataset
        """
        
        return pd.DataFrame(self.preprocessor.transform(X))
        
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5
