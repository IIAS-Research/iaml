"""
[STEP] Random Over Sampling
"""
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('normalize')
class ActRandomOverSampling(Actionable):
    """
    [STEP] Random Over Sampling
    """
    name = "Random Over Sampling"
    refs = [
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
        self.configuration:dict = {}
        self.resampler:RandomOverSampler = None
    
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Add resample random over sampling to Candidate

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.resampler = RandomOverSampler(sampling_strategy='minority')
        self.resampler.fit(dataset.X, dataset.y)
        return self
    
    def resample(self, X:pd.DataFrame, y:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply Random Over Sampling

        Args:
            X (pd.DataFrame): Features to resample
            y (pd.DataFrame): Labels to resample

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: _description_
        """
        return self.resampler.fit_resample(X, y)
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 1
    
    def suitable(self, dataset:Dataset) -> bool:
        return dataset.type_of_target in ['binary', 'multiclass']
