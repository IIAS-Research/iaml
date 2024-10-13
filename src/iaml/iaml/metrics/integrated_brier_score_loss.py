"""
[METRIC] Integrated Brier Score Loss for Survival Models
"""
import textwrap
import pandas as pd
import numpy as np
from ..dataset import Dataset
from .integrated_brier_score import IntegratedBrierScoreMetric


class IntegratedBrierScoreLossMetric(IntegratedBrierScoreMetric):
    """
    [METRIC] Integrated Brier Score Loss for Survival Models
    """
    
    name = 'Integrated Brier Score Loss for Survival Models'
    description = textwrap.dedent('''\
        The Integrated Brier Score (IBS) is a measure used to evaluate how well survival 
        models predict the likelihood of an event happening over time. Here we compute 1 - IBS''')
    description_long = textwrap.dedent('''\
        The Integrated Brier Score (IBS) is a tool used to check how accurately survival models predict events, 
        like the time until a patient experiences a certain outcome. It compares the model's predictions with 
        real-life results over a period of time, taking into account situations where some data may be missing 
        or incomplete. By looking at these differences, the IBS provides a single score that summarizes the model's 
        performance. A lower IBS score indicates that the model is doing a better job at making accurate predictions, 
        which is important for making informed decisions in healthcare and research. Here we compute 1 - IBS
        ''')
    
    def __str__(self):
        return 'integrated_brier_score_loss'
    
    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame, 
                y_train:pd.DataFrame=None, **kwargs) -> float:
        """
        Compute the Integrated Brier Score Loss (IBS) with the predicted data.

        Args:
            y (pd.DataFrame): Ground truth data (duration and event status)
            y_pred (pd.DataFrame): Predicted data (risk scores or predicted survival times)
            kwargs: Additional arguments, including y_train and X_train

        Returns:
            float: Computed Integrated Brier Score
        """
        return 1 - super().compute(y, y_pred, y_train, **kwargs)
