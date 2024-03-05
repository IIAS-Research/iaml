"""
[STEP] Random Over Sampling
"""
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from ...actionable import Actionable
from ...output import Output, Input
from ...step import is_step, runner


@is_step('normalize')
class ActRandomOverSampling(Actionable):
    """
    [STEP] Random Over Sampling
    """
    name = "Random Over Sampling"
    def __init__(self):
        self.configurations = [{}]
    
    @runner
    def run(self, input_data: Input, callback:callable=None) -> Output: # pylint: disable=unused-argument
        """
        Add resample random over sampling to Input

        Args:
            input_data (Input): Fit data
            callback (callable, optional): Call after each step. Defaults to None.

        Returns:
            Output: Transformed input
        """
        return input_data.add_resample(self)
    
    def resample(self, X:pd.DataFrame, y:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply Random Over Sampling

        Args:
            X (pd.DataFrame): Features to resample
            y (pd.DataFrame): Labels to resample

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: _description_
        """
        return RandomOverSampler(sampling_strategy='minority').fit_resample(X, y)
    
    def priorize(self, input_data:Input=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 1
    
    def suitable(self, input_data: Input) -> bool:
        return input_data.dataset.type_of_target in ['binary', 'multiclass']
