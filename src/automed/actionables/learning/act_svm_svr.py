"""
[STEP] Learn :  SVM Regressor
"""
from sklearn import svm
from .act_base_learning import ActBaseLearning
from ...output import Input
from ...step import is_step, runner

@is_step('learning', 'tabular')
class ActSVMSVR(ActBaseLearning):
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
        self.model = svm.SVR(kernel=self.get_config('kernel'))
        self.model.fit(input_data.dataset.X, input_data.dataset.y)
        
        return input_data.set_model(self)
    
    def suitable(self, input_data: Input) -> bool:
        return input_data.dataset.type_of_target == 'continuous'
