"""
[METRIC] Accuracy
"""
from collections import Counter
from sklearn.metrics import accuracy_score
import pandas as pd
from ..metric import Metric
import textwrap

class AccuracyMetric(Metric):
    """
    [METRIC] Accuracy
    """
    name = 'Accuracy'
    description = textwrap.dedent('''\
        Accuracy measures how well a model predicts outcomes by calculating the percentage 
        of correct predictions out of the total predictions. Higher accuracy means better performance.''')
    description_long = textwrap.dedent('''\
        Accuracy is a tool to evaluate how well a predictive 
        model works, especially in healthcare. 
        It shows the percentage of correct predictions made by the model.
        To calculate it, you add the number of correct positive and negative predictions, 
        then divide by the total number of predictions. 
        For example, if a model is correct 80 times out of 100, its accuracy is 80%.''')
    refs = [
        {
            'year': 2006,
            'name': 'Understanding the meaning of accuracy, trueness and precision',
            'authors': [
                'Antonio Menditto',
                'Marina Patriarca',
                'Bertil Magnusson'    
            ],
            'doi': 'https://doi.org/10.1007/s00769-006-0191-z',
            'publisher': ' Accreditation and Quality Assurance, Volume 12, pages 45--47'
        }
    ]

    def __str__(self) -> str:
        return 'accuracy'
    
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'Accuracy classification score.'
    
    # Get one label (numpy.array or pd.series) and return true is classes is balanced
    def __is_balanced(self, y):
        class_count = Counter(y)
        total_samples = y.shape[0]
        ideal_count = total_samples/len(class_count)
        threshold = 0.20 * ideal_count
        return not any(abs(count - ideal_count) > threshold for count in class_count.values())
    
    def suitable(self, X:pd.DataFrame, y:pd.DataFrame, type_of_target:str) -> bool:
        """
        Does this metric is suitable for this candidate ?
        Must be classification with balanced labels

        Args:
            X (pd.DataFrame): Features
            y (pd.DataFrame): labels
            type_of_target (str): Type of target

        Returns:
            bool: Suitable ?
        """
        return type_of_target in ['binary', 'multiclass'] and not self.__is_balanced(y)
        
    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame, **kwargs) -> float:
        """
        Compute metric with predicted data

        Args:
            y (pd.DataFrame): Ground truth data
            y_pred (pd.DataFrame): Predicted data

        Returns:
            float: computed value 
        """
        return accuracy_score(y, y_pred) 
