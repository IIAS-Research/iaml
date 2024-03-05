"""
[STEP] Learn :  SVM Regressor
"""
from sklearn import svm
import pandas as pd
from ...actionable import Actionable
from ...output import Input
from ...step import is_step, runner

@is_step('learning', 'tabular')
class ActSVMSVR(Actionable):
    """
    [STEP] Learn :  SVM Regressor
    """
    name = "Learn : SVM Regression"
    def __init__(self):
        self.configuration:dict = {
            'kernel': {
                'description': 'Kernel to use in the SVM',
                'default': 'rbf',
                'categorical': ['linear', 'poly', 'rbf', 'sigmoid']
            }
        }
        self.model:svm.SVR = None
        
    @runner
    def run(self, input_data: Input, callback=None): # pylint: disable=unused-argument
        """
        Fit SVM Regressor on Input.dataset

        Args:
            input_data (Input): Fit data
            callback (callable, optional): Call after each step. Defaults to None.

        Returns:
            Output: Transformed input
        """
        self.model = svm.SVR(
            kernel = self.get_config('kernel')
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
        
    
    def suitable(self, input_data: Input) -> bool:
        return input_data.dataset.type_of_target in ['continuous']
    
    def priorize(self, input_data:Input=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
