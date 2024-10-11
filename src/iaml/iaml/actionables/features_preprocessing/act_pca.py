"""
[STEP] Decompose features with PCA
"""
import textwrap
import pandas as pd
from sklearn.decomposition import PCA
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('features_preprocessing')
class ActPCA(Actionable):
    """
    [STEP] Reduce dimensions with PCA
    """
    name = "PCA"
    description = "Apply PCA for dimensionality reduction over a list of columns"
    description_long = textwrap.dedent('''\
        PCA, or Principal Component Analysis, is a dimensionality reduction technique. 
        It transforms the data into a set of linearly uncorrelated components, capturing 
        the maximum variance in the data with each successive component. 
        This method is unsupervised, meaning it does not require labeled data, 
        and is particularly useful for simplifying datasets while retaining 
        as much of the underlying structure as possible.
    ''')
    
    def __init__(self):
        self.configuration:dict = {
            'n_components': {
                'description': 'Number of components to keep.',
                'default': 0.999,
                'range': [0.5, 0.999]
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
        self.preprocessor = PCA(**self.passthrough_parameters())
        self.preprocessor.fit(dataset.X)
        return self
    
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Apply PCA

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
