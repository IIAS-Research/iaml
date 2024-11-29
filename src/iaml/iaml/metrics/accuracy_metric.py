"""[METRIC] Accuracy"""
from typing import Any
import textwrap
from collections import Counter
from sklearn.metrics import accuracy_score
import pandas as pd
from numpy import ndarray
from ..metric import Metric

class AccuracyMetric(Metric):
    """[METRIC] Accuracy"""
    name: str = 'Accuracy'
    description: str = textwrap.dedent('''\
        Accuracy measures how well a model predicts outcomes by calculating the percentage 
        of correct predictions out of the total predictions. Higher accuracy means better performance.
        ''')
    description_long: str = textwrap.dedent('''\
        Accuracy is a tool to evaluate how well a predictive 
        model works, especially in healthcare. 
        It shows the percentage of correct predictions made by the model.
        To calculate it, you add the number of correct positive and negative predictions, 
        then divide by the total number of predictions. 
        For example, if a model is correct 80 times out of 100, its accuracy is 80%.
        ''')
    refs: list[dict[str, Any]] = [
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

    def __is_balanced(self, y: ndarray | pd.Series) -> bool:
        """Get one label (numpy.array or pd.series) and 
        return true if classes is balanced
        
        :return: Balanced ?
        """
        class_count = Counter(y)
        total_samples = y.shape[0]
        ideal_count = total_samples/len(class_count)
        threshold = 0.20 * ideal_count
        return not any(abs(count - ideal_count) > threshold for count in class_count.values())

    def suitable(self, X: pd.DataFrame, y: pd.DataFrame, type_of_target: str) -> bool:
        return type_of_target in ['binary', 'multiclass'] and not self.__is_balanced(y)

    def compute(self, y: pd.DataFrame, y_pred: pd.DataFrame, **kwargs) -> float:
        return accuracy_score(y, y_pred)
