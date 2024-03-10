"""
[STEP] Random Over Sampling
"""
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from ...actionable import Actionable
from ...dataset import Dataset
from ...output import Input
from ...decorators.all import is_step


@is_step('normalize')
class ActRandomOverSampling(Actionable):
    """
    [STEP] Random Over Sampling
    """
    name = "Random Over Sampling"
    def __init__(self):
        self.configuration:dict = {}
        self.resampler:RandomOverSampler = None
    
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Add resample random over sampling to Input

        Args:
            input_data (Input): Fit data
            callback (callable, optional): Call after each step. Defaults to None.

        Returns:
            Output: Transformed input
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
    
    def priorize(self, input_data:Input=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 1
    
    def suitable(self, input_data: Input) -> bool:
        return input_data.dataset.type_of_target in ['binary', 'multiclass']
