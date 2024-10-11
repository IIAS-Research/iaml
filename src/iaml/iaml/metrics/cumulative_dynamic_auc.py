"""
[METRIC] Cumulative Dynamic AUC for Survival Models
"""
import textwrap
import pandas as pd
import numpy as np
from sksurv.metrics import cumulative_dynamic_auc
from ..metric import Metric
from ..dataset import Dataset

class CumulativeDynamicAUCMetric(Metric):
    """
    [METRIC] Cumulative Dynamic AUC for Survival Models
    """
    name = 'Concordance Index for Survival Models'
    description = textwrap.dedent('''\
        The Cumulative Dynamic AUC (Area Under the Curve) for Survival Models measures the accuracy of a survival model 
        in predicting the probability of an event over time. A higher AUC indicates better predictive performance.
        ''')
    description_long = textwrap.dedent('''\
        The Cumulative Dynamic AUC for Survival Models evaluates how well a model predicts 
        the likelihood of an event, such as death or disease, at various time points. Unlike traditional AUC, which 
        assesses binary classification, the cumulative dynamic AUC accounts for time-dependent predictions 
        in survival analysis. This metric calculates the area under the curve of the time-dependent receiver 
        operating characteristic (ROC) curve, providing a comprehensive view of model performance over time. 
        Values range from 0 to 1, where 0.5 indicates no predictive ability and 1 indicates perfect prediction. 
        The Cumulative Dynamic AUC is particularly useful for researchers and clinicians in assessing the effectiveness
        of survival models in real-world scenarios.
        ''')
    
    refs=[
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

    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'Computes the time-dependent Cumulative Dynamic Area Under the ROC Curve (AUC) \
                for survival models. The time-dependent AUC measures the model’s ability to \
                distinguish between subjects who experience the event before a given time and \
                those who do not, considering censored data. Higher values indicate \
                better discrimination, with 1 being perfect.'

    def __str__(self):
        return 'cumulative_dynamic_auc'
    
    
    @property
    def needed_prediction(self):
        return 'predict_survival_function'

    def suitable(self, X: pd.DataFrame, y: pd.DataFrame, type_of_target: str) -> bool:
        """
        Is this metric suitable for this candidate? Must be survival analysis.

        Args:
            X (pd.DataFrame): Features
            y (pd.DataFrame): Labels (with two columns: 'duration' and 'event')
            type_of_target (str): Type of target (must be 'survival')

        Returns:
            bool: Suitable?
        """
        return type_of_target == 'survival'
        
    def compute(self, y: pd.DataFrame, y_pred: pd.DataFrame, 
                y_train: pd.DataFrame = None, **kwargs) -> float:
        """
        Compute the cumulative dynamic AUC using the predicted data.

        Args:
            y (pd.DataFrame): Ground truth data (duration and event status)
            y_pred (pd.DataFrame): Predicted risk scores or survival probabilities
            y_train (pd.DataFrame): Training data (duration and event status)
            time_points (array-like): Time points at which to compute the AUC
            kwargs: Additional arguments, including y_train and X_train

        Returns:
            float: Mean Cumulative Dynamic AUC across the specified time points
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
