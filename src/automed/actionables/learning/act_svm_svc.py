"""
[STEP] Learn :  SVM Classifier
"""
from sklearn import svm
import pandas as pd
from ...actionable import Actionable
from ...output import Input
from ...step import is_step, runner

@is_step('learning', 'tabular')
class ActSVMSVC(Actionable):
    """
    [STEP] Learn :  SVM Classifier
    """
    name = "Learn : SVM Classification"
    def __init__(self):
        self.configurations:list[dict] = [{
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
                'description': 'If true, the output will be a probability. If false, \
                    it will be Binary',
                'default': False
            },
            'class_weight': {
                'description': 'Can be set on "balenced" to improve results on unbalenced data',
                'default': None,
                'categorical': [None, 'balanced']
            }
        }]
        self.model:svm.SVC = None
        
    @runner
    def run(self, input_data: Input, callback=None): # pylint: disable=unused-argument
        """
        Fit SVM classifier on Input.dataset

        Args:
            input_data (Input): Fit data
            callback (callable, optional): Call after each step. Defaults to None.

        Returns:
            Output: Transformed input
        """
        self.model = svm.SVC(
            kernel = self.get_config('kernel'),
            class_weight = self.get_config('class_weight'),
            random_state = self.get_config('random_state'),
            probability = self.get_config('probability')
            )
        
        self.model.fit(input_data.dataset.X, input_data.dataset.y)
        
        return input_data.set_model(self)
        
    def predict(self, X:pd.DataFrame) -> list[float]:
        """
        Apply prediction model on DataFrame

        Args:
            X (pd.DataFrame): DataFrame use to predict

        Returns:
            list[float]: Predicted values
        """
        return self.model.predict(X)
        
    
    def suitable(self, input_data) -> bool:
        return input_data.dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator']
    
    def priorize(self, input_data:Input=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
