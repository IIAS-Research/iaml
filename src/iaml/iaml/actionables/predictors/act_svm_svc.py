"""
[STEP] Learn :  SVM Classifier
"""
from sklearn import svm
from ...predictor import Predictor
from ...candidate import Candidate
from ...dataset import Dataset
from ...decorators.all import is_step

@is_step('predictor', 'tabular', 'classifier')
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
            'class_weight': {
                'description': 'Can be set on "balenced" to improve results on unbalenced data',
                'default': None,
                'categorical': [None, 'balanced']
            },
            'tol': {
                'description': 'The stopping criterion.',
                'default': 0.001,
                'range': [1e-05, 0.1]
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
            probability = True, # Needed to predict_proba (thus MetaLearner)
            **self.passthrough_parameters()
            )
        
        self.model.fit(dataset.X, dataset.y)
        
        return self
    
    def suitable(self, dataset:Dataset) -> bool:
        return dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator']
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
