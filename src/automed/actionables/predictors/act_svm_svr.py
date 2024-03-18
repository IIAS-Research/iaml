"""
[STEP] Learn :  SVM Regressor
"""
from sklearn import svm
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('predictor', 'tabular')
class ActSVMSVR(Predictor):
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
        
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit SVM Regressor on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = svm.SVR(
            **self.model_parameters()
            )
        
        self.model.fit(dataset.X, dataset.y)
        
        return self
    
    
    def suitable(self, candidate: Candidate) -> bool:
        return candidate.dataset.type_of_target in ['continuous']
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
