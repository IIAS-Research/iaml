"""[METRIC] Cumulative Dynamic AUC for Survival Models"""
from typing import Any
import textwrap
import pandas as pd
import numpy as np
from sksurv.metrics import cumulative_dynamic_auc
from ..metric import Metric
from ..dataset import Dataset

class CumulativeDynamicAUCMetric(Metric):
    """[METRIC] Cumulative Dynamic AUC for Survival Models"""
    name: str = 'Cumulative AUC'
    description: str = textwrap.dedent('''\
        The Cumulative Dynamic AUC (Area Under the Curve) for Survival Models measures the accuracy of a survival model 
        in predicting the probability of an event over time. A higher AUC indicates better predictive performance.
        ''')
    description_long: str = textwrap.dedent('''\
        The Cumulative Dynamic AUC for Survival Models evaluates how well a model predicts 
        the likelihood of an event, such as death or disease, at various time points. Unlike traditional AUC, which 
        assesses binary classification, the cumulative dynamic AUC accounts for time-dependent predictions 
        in survival analysis. This metric calculates the area under the curve of the time-dependent receiver 
        operating characteristic (ROC) curve, providing a comprehensive view of model performance over time. 
        Values range from 0 to 1, where 0.5 indicates no predictive ability and 1 indicates perfect prediction. 
        The Cumulative Dynamic AUC is particularly useful for researchers and clinicians in assessing the effectiveness
        of survival models in real-world scenarios.
        ''')
    refs: list[dict[str, Any]]=[
        {
            'year': 2007,
            'name': textwrap.dedent("""\
                Evaluating prediction rules for t-year survivors with censored regression models
                """),
            'authors': [
                'H. Uno',
                'T. Cai.',
                ' L. Tian',
                'L. J. Wei'
            ],
            'doi': 'https://doi.org/10.1198/016214507000000149',
            'publisher': 'Journal of the American Statistical Association, vol. 102, pp. 527–537'
        },
        {
            'year': 2010,
            'name': 'Estimation methods for time-dependent AUC models with survival data',
            'authors': [
                'H. Hung',
                'C. T. Chiang'
            ],
            'doi': '',
            'publisher': 'Canadian Journal of Statistics, vol. 38, no. 1, pp. 8–26'
        },
        {
            'year': 2014,
            'name': textwrap.dedent("""\
                Summary measure of discrimination in survival models based on cumulative/dynamic time-dependent ROC curves
                """),
            'authors': [
                'J. Lambert',
                'S. Chevret'
            ],
            'doi': '',
            'publisher': 'Statistical Methods in Medical Research'
        }
    ]

    def __str__(self) -> str:
        return 'cumulative_dynamic_auc'

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
        """Compute the cumulative dynamic AUC using the predicted data.

        :param pd.DataFrame y: Ground truth data (duration and event status).
        :param pd.DataFrame y_pred: Predicted risk scores or survival probabilities.
        :param pd.DataFrame y_train: Training data (duration and event status).
        :param dict, optional \\**kwargs: Additional parameters
        :return: Computed value
        """
        y_train = np.array(y_train, dtype=[('event', 'bool'), ('time', 'float')])
        y = Dataset.fix_y_survival(y, y_train)
        y = np.array(y, dtype=[('event', 'bool'), ('time', 'float')])

        # Extract time from y test
        _, time = zip(*y)
        times = np.arange(min(time), max(time))

        # Extract risk score for each time point
        predictions = np.asarray([[fn(t) for t in times] for fn in y_pred])

        # Calculate cumulative dynamic AUC using sksurv function
        all_points, _ = cumulative_dynamic_auc(y_train, y, predictions, times)

        # Return mean AUC across time points
        return np.nanmean(all_points)
