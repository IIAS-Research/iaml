"""[METRIC] Integrated Brier Score Loss for Survival Models"""
import textwrap
import pandas as pd
from .integrated_brier_score import IntegratedBrierScoreMetric


class IntegratedBrierScoreLossMetric(IntegratedBrierScoreMetric):
    """[METRIC] Integrated Brier Score Loss for Survival Models"""

    name: str = 'Reverse Integrated Brier Score'
    description: str = textwrap.dedent('''\
        The Integrated Brier Score (IBS) is a measure used to evaluate how well survival 
        models predict the likelihood of an event happening over time. Here we compute 1 - IBS''')
    description_long: str = textwrap.dedent('''\
        The Integrated Brier Score (IBS) is a tool used to check how accurately survival models predict events, 
        like the time until a patient experiences a certain outcome. It compares the model's predictions with 
        real-life results over a period of time, taking into account situations where some data may be missing 
        or incomplete. By looking at these differences, the IBS provides a single score that summarizes the model's 
        performance. A lower IBS score indicates that the model is doing a better job at making accurate predictions, 
        which is important for making informed decisions in healthcare and research. Here we compute 1 - IBS
        ''')

    def __str__(self) -> str:
        return 'integrated_brier_score_loss'

    def compute(
        self,
        y: pd.DataFrame,
        y_pred: pd.DataFrame,
        y_train: pd.DataFrame = None,
        **kwargs) -> float:
        """Compute the Integrated Brier Score Loss (IBS) with the predicted data.

        :param pd.DataFrame y: Ground truth data (duration and event status).
        :param pd.DataFrame y_pred: Predicted data (risk scores or predicted survival times).
        :param pd.DataFrame y_train: Training data (duration and event status).
        :param dict, optional \\**kwargs: Additional parameters
        :return: Computed value
        """
        return 1 - super().compute(y, y_pred, y_train, **kwargs)
