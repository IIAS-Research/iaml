"""
[METRIC] Specificity Multilabel
"""
import textwrap
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np
from ..metric import Metric

class SpecificityMultilabelMetric(Metric):
    """
    [METRIC] Specificity Multilabel
    """

    name= "Specificity Multilabel"
    _description = textwrap.dedent('''\
        Multilabel specificity is a metric used to evaluate the performance of a classification model that 
        predicts multiple labels for each instance. It measures how well the model identifies negative cases 
        for each label individually.''')
    _description_long = textwrap.dedent('''\
        Multilabel specificity assesses how effectively a classification model identifies negative cases for 
        multiple labels. It calculates specificity for each label separately by determining the true negatives 
        and false positives for that label. After calculating the specificity for all labels, these values are 
        averaged to obtain an overall measure. A high multilabel specificity indicates that the model is good at 
        correctly identifying non-target labels, while a low score suggests it may misclassify negative cases. 
        In summary, multilabel specificity helps evaluate a model's ability to accurately recognize negative outcomes 
        across various labels.''')
    
    refs=[
        {
            'year': 2021,
            'name': 'Comprehensive Comparative Study of Multi-Label Classification Methods',
            'authors': [
                'Jasmin Bogatinovski', 
                'Ljupčo Todorovski', 
                'Sašo Džeroski', 
                'Dragi Kocev',
            ],
            'doi': 'https://doi.org/10.48550/arXiv.2102.07113',
            'publisher': ''
        }
    ]

    def _str__(self):
        return 'specificity_multilabel'
    
    def suitable(self, X:pd.DataFrame, y:pd.DataFrame, type_of_target:str) -> bool:
        """
        Does this metric is suitable for this candidate ?
        Must be multilabel classification 

        Args:
            X (pd.DataFrame): Features
            y (pd.DataFrame): labels
            type_of_target (str): Type of target

        Returns:
            bool: Suitable ?
        """
        return type_of_target in ['multilabel-indicator']
    
    # Specificity is calculated for each label separately, 
    # then averaged to obtain an overall measure. 
    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame, **kwargs) -> float:
        """
        Compute metric with predicted data

        Args:
            y (pd.DataFrame): Ground truth data
            y_pred (pd.DataFrame): Predicted data

        Returns:
            float: computed value 
        """
        # Generates a series of confusion matrices,  one for each label
        mcm = multilabel_confusion_matrix(y, y_pred)
        specificity_per_label = []
        for i in range(mcm.shape[0]):
            tn, fp, _, _ = mcm[i].ravel()
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
            specificity_per_label.append(specificity)
                
        mean_specificity = np.mean(specificity_per_label)
        return mean_specificity
