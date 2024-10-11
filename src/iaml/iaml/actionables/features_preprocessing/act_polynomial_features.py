"""
[STEP] Decompose features with PolynomialFeatures
"""
import textwrap
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('features_preprocessing')
class ActPolynomialFeatures(Actionable):
    """
    [STEP] Preprocess with PolynomialFeatures
    """
    name= "Preprocess with PolynomialFeatures"
    description = textwrap.dedent('''\
        PolynomialFeatures creates new features by combining existing
        features mathematically. It squares, cubes, and multiplies features to 
        create more complex patterns.''')
    description_long = textwrap.dedent('''\
        PolynomialFeatures is a preprocessing technique that
        generates new features based on polynomial relationships between existing
        features. This helps capture non-linear relationships in the data that may
        not be apparent from the original features alone. PolynomialFeatures is
        particularly useful when you suspect the underlying relationship in your data
        might not be straightforward or linear.''')
    
    def __init__(self):
        self.configuration:dict = {
            'include_bias': {
                'description': 'If True (default), then include a bias \
                    column, the feature in which all polynomial powers are zero',
                'default': True
                },
            'interaction_only': {
                'description': 'If True, only interaction features are produced',
                'default': False
                },
            'degree': {
                'description': 'Degree of the polynomial kernel.',
                'default': 3,
                'range': [2, 5]
                }
            }
        
        self.optimizable = True
        self.preprocessor = None


    def fit(self, dataset:Dataset) -> Actionable:
        """
        Fit Features agglomerations

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        
        self.preprocessor = PolynomialFeatures(**self.passthrough_parameters())
        self.preprocessor.fit(dataset.X)
        
        return self
    
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Apply PolynomialFeatures

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
