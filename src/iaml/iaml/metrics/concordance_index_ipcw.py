"""
[METRIC] Concordance Index with Inverse Probability of Censoring 
Weights (IPCW) for Survival Models
"""
import textwrap
import pandas as pd
import numpy as np
from sksurv.metrics import concordance_index_ipcw
from ..metric import Metric
from ..dataset import Dataset

class ConcordanceIndexIPCWMetric(Metric):
    """
    [METRIC] Concordance Index with Inverse Probability of Censoring 
    Weights (IPCW) for Survival Models
    """
    name = 'Concordance Index with Inverse Probability of Censoring Weights for Survival Models'
    description = textwrap.dedent('''\
        The Concordance Index with Inverse Probability of Censoring Weights (IPCW) 
        evaluates survival models by measuring how well they predict the order of events, 
        like survival times, while accounting for censored data. A higher index indicates 
        better predictive accuracy.''')
    description_long = textwrap.dedent('''\
        The Concordance Index with Inverse Probability of Censoring Weights (IPCW) 
        assesses survival models by focusing on the ranking of survival times. It addresses the issue of censored 
        data—when some outcomes are not fully observed—by applying weights based on the probability of censoring. 
        The index ranges from 0 to 1, with 0.5 indicating no predictive ability and 1 indicating perfect prediction. 
        By incorporating IPCW, this metric provides a more accurate evaluation of a model's performance, making it 
        essential for researchers in fields like medicine and epidemiology.''')
    
    refs=[
        {
            'year': 2011,
            'name': textwrap.dedent("""\
                On the C-statistics for evaluating overall adequacy of risk prediction 
                procedures with censored survival data"""),
            'authors': [
                'Hajime Uno',
                'Tianxi Cai',
                'Michael J. Pencina',
                'Ralph B. D\'Agostino',
                'L. J. Wei'
            ],
            'doi': 'https://doi.org/10.1002/sim.4154',
            'publisher': 'Statistics in Medicine, vol. 18, no. 17-18, pp. 2529–2545'
        }
    ]

    def __str__(self):
        return 'concordance_index_ipcw'
    
    
    @property
    def needed_prediction(self):
        return 'predict'

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
        Compute the Concordance Index (C-index) with IPCW using the predicted data.

        Args:
            y (pd.DataFrame): Ground truth data (duration and event status)
            y_pred (pd.DataFrame): Predicted data (risk scores or predicted survival times)
            kwargs: Additional arguments, including y_train and X_train

        Returns:
            float: Computed Concordance Index (C-index) using IPCW
        """
        y_train = np.array(y_train, dtype=[('event', 'bool'), ('time', 'float')])
        y = Dataset.fix_y_survival(y, y_train)        
        y = np.array(y, dtype=[('event', 'bool'), ('time', 'float')])

        # Calculate concordance index using sksurv function
        return concordance_index_ipcw(y_train, y, y_pred)[0]
