"""[METRIC] Integrated Brier Score for Survival Models"""
from typing import Any
import textwrap
import pandas as pd
import numpy as np
from sksurv.metrics import integrated_brier_score
from ..metric import Metric
from ..dataset import Dataset


class IntegratedBrierScoreMetric(Metric):
    """[METRIC] Integrated Brier Score for Survival Models"""

    name: str = 'Integrated Brier Score'
    _description: str = textwrap.dedent('''\
        The Integrated Brier Score (IBS) is a measure used to evaluate how well survival 
        models predict the likelihood of an event happening over time. It looks at the 
        difference between what the model predicts and what actually happens, including cases 
        where data is incomplete. A lower IBS score means the model is more accurate.''')
    _description_long: str = textwrap.dedent('''\
        The Integrated Brier Score (IBS) is a tool used to check how accurately survival models predict events, 
        like the time until a patient experiences a certain outcome. It compares the model's predictions with 
        real-life results over a period of time, taking into account situations where some data may be missing 
        or incomplete. By looking at these differences, the IBS provides a single score that summarizes the model's 
        performance. A lower IBS score indicates that the model is doing a better job at making accurate predictions, 
        which is important for making informed decisions in healthcare and research.''')
    refs: list[dict[str, Any]] = [
        {
            'year': 1999,
            'name': textwrap.dedent("""\
                Assessment and comparison of prognostic classification schemes for survival data
                """),
            'authors': [
                'E. Graf',
                'C. Schmoor',
                'W. Sauerbrei',
                'M. Schumacher'
            ],
            'doi': textwrap.dedent("""\
                https://doi.org/10.1002/(SICI)1097-0258(19990915/30)18:17/18%3C2529::AID-SIM274%3E3.0.CO;2-5
                """),
            'publisher': ' Statistics in Medicine, vol. 18, no. 17-18, pp. 2529–2545'
        }
    ]

    def __str__(self) -> str:
        return 'integrated_brier_score'

    @property
    def needed_prediction(self) -> str:
        return 'predict_survival_function'

    def suitable(self, X: pd.DataFrame, y: pd.DataFrame, type_of_target: str) -> bool:
        return type_of_target == 'survival'

    def compute(
        self,
        y: pd.DataFrame,
        y_pred: pd.DataFrame,
        y_train: pd.DataFrame = None,
        **kwargs) -> float:
        """Compute the Integrated Brier Score (IBS) with the predicted data.

        :param pd.DataFrame y: Ground truth data (duration and event status).
        :param pd.DataFrame y_pred: Predicted data (risk scores or predicted survival times).
        :param pd.DataFrame y_train: Training data (duration and event status).
        :param dict, optional \\**kwargs: Additional parameters
        :return: Computed value
        """
        y_train_samples = Dataset.normalize_survival_target(y_train)
        y_samples = Dataset.fix_y_survival(y, y_train_samples)

        y_train_struct = np.array(
            y_train_samples,
            dtype=[('event', 'bool'), ('time', 'float')]
        )
        y_struct = np.array(
            y_samples,
            dtype=[('event', 'bool'), ('time', 'float')]
        )

        # Extract time from y test
        _, time = zip(*y_samples)
        times = np.arange(min(time), max(time))

        # Extract risk score for each time point
        predictions = np.asarray([[fn(t) for t in times] for fn in y_pred])

        # Calculate integrated Brier score using sksurv function
        return integrated_brier_score(y_train_struct, y_struct, predictions, times)
