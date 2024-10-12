"""
[METRIC] Balanced Accuracy
"""
import textwrap
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
from ..metric import Metric

class BalancedAccuracyMetric(Metric):
    """
    [METRIC] Balanced Accuracy
    """
    name = 'Balanced Accuracy'
    description = textwrap.dedent('''\
        Balanced Accuracy Score is a metric that evaluates a model's 
        performance by considering both positive and negative classes equally. 
        It calculates the average accuracy for each class, making it useful for imbalanced dataset.
        ''')
    description_long = textwrap.dedent('''\
        Balanced Accuracy Score evaluates how well a predictive 
        model performs, giving equal importance to both positive and negative classes. 
        This is important in healthcare when data is imbalanced.
        To calculate it, you find the accuracy for each class and then average those values. 
        For example, if a model has 70% accuracy for positive cases and 90% for negative cases, 
        the balanced accuracy is (70% + 90%) / 2 = 80%. This metric ensures that the model is effective 
        for all classes, making it valuable for medical decision-making.
        ''')
    refs=[
        {
            'year': 2010,
            'name': 'The Balanced Accuracy and Its Posterior Distribution',
            'authors': [
                'Kay Henning Brodersen',
                'Cheng Soon Ong',
                'Klaas Enno Stephan',
                'Joachim M. Buhmann'
            ],
            'doi': 'https://doi.org/10.1109/ICPR.2010.764',
            'publisher': textwrap.dedent("""\
                Proceedings of the 20th International Conference on Pattern Recognition, 3121-24.
                """)
        },
        {
            'year': 2015,
            'name': textwrap.dedent("""\
                Fundamentals of Machine Learning for Predictive Data Analytics: Algorithms, Worked Examples, and Case Studies
                """),
            'authors': [
                'John D. Kelleher',
                'Brian Mac Namee',
                'Aoife D\'Arcy'
            ],
            'doi': None,
            'publisher': textwrap.dedent("""\
                Fundamentals of Machine Learning for Predictive Data Analytics: Algorithms, Worked Examples, and Case Studies
                """)
        }
    ]
    
    def __str__(self):
        return 'balanced_accuracy'
    
    def suitable(self, X:pd.DataFrame, y:pd.DataFrame, type_of_target:str) -> bool:
        """
        Does this metric is suitable for this candidate ?
        Must be classification

        Args:
            X (pd.DataFrame): Features
            y (pd.DataFrame): labels
            type_of_target (str): Type of target

        Returns:
            bool: Suitable ?
        """
        return type_of_target in ['binary', 'multiclass']
    
    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame, **kwargs) -> float:
        """
        Compute metric with predicted data

        Args:
            y (pd.DataFrame): Ground truth data
            y_pred (pd.DataFrame): Predicted data

        Returns:
            float: computed value 
        """
        return balanced_accuracy_score(y, y_pred)
