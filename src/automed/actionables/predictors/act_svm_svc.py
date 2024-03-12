"""
[STEP] Learn :  SVM Classifier
"""
from sklearn import svm
import pandas as pd
from ...predictor import Predictor
from ...candidate import Candidate
from ...dataset import Dataset
from ...decorators.all import is_step

@is_step('predictor', 'tabular')
class ActSVMSVC(Predictor):
    """
    [STEP] Learn :  SVM Classifier
    """
    name = "Learn : SVM Classification"
    def __init__(self):
        self.configuration:dict = {
            'kernel': {
                'description': 'Kernel to use in the SVM',
                'default': 'rbf',
                'categorical': ['linear', 'poly', 'rbf', 'sigmoid']
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            },
            'probability': {
                'description': 'If true, the candidate will be a probability. If false, \
                    it will be Binary',
                'default': False
            },
            'class_weight': {
                'description': 'Can be set on "balenced" to improve results on unbalenced data',
                'default': None,
                'categorical': [None, 'balanced']
            }
        }
        self.model:svm.SVC = None
        
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit SVM classifier on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = svm.SVC(
            kernel = self.get_config('kernel'),
            class_weight = self.get_config('class_weight'),
            random_state = self.get_config('random_state'),
            probability = self.get_config('probability')
            )
        
        self.model.fit(dataset.X, dataset.y)
        
        return self
        
    def predict(self, X:pd.DataFrame) -> list[float]:
        """
        Apply prediction model on DataFrame

        Args:
            X (pd.DataFrame): DataFrame use to predict

        Returns:
            list[float]: Predicted values
        """
        return self.model.predict(X)
        
    
    def suitable(self, candidate) -> bool:
        return candidate.dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator']
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
