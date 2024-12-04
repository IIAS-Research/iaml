"""
[STEP]  SVM Regressor
"""
import textwrap
from sklearn import svm
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step

@is_step('predictor', 'tabular', 'regressor')
class ActSVMSVR(Predictor):
    """
    [STEP]  SVM Regressor
    """
    name = "SVM Regression"
    _description = textwrap.dedent('''\
        SVM Regressor is a machine learning algorithm that models the relationship
        between input features and a continuous output variable using a support vector machine
        (SVM). It can handle non-linearly separable data by using a kernel function to map the data
        into a higher-dimensional space.''')
    _description_long = textwrap.dedent('''\
        SVM Regressor is a type of regression algorithm that models the
        relationship between input features and a continuous output variable using a support
        vector machine (SVM). It works by finding the optimal hyperplane or boundary that predicts
        the output variable with the minimum error.''')
    
    refs = [
        {
            'year': 1999,
            'name': 'Probabilistic Outputs for Support Vector Machines and Comparisons to \
                Regularized Likelihood Methods',
            'authors': [
                'John C. Platt'
            ],
            'doi': 'https://api.semanticscholar.org/CorpusID:5656387',
            'publisher': 'Microsoft Research'
        },
        {
            'year': 2001,
            'name': 'LIBSVM: A Library for Support Vector Machines',
            'authors': [
                'Chih-Chung Chang',
                'Chih-Jen Lin'
            ],
            'doi': 'https://doi.org/10.1145/1961189.1961199',
            'publisher': 'ACM Transactions on Intelligen Systems and Technology Vol.2 page 1--27'
        },
    ]

    def __init__(self):
        self.configuration = {
            'kernel': {
                'description': 'Kernel to use in the SVM',
                'default': 'rbf',
                'categorical': ['linear', 'poly', 'rbf', 'sigmoid']
            },
            'epsilon': {
                'description': 'Epsilon in the epsilon-SVR model.',
                'default': 0.1,
                'range': [1e-05, 0.1]
            },
            'tol': {
                'description': 'The stopping criterion.',
                'default': 0.001,
                'range': [1e-05, 0.1]
            }
        }
        self.model: svm.SVR = None
        
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit SVM Regressor on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = svm.SVR(
            **self.passthrough_parameters()
            )
        
        self.model.fit(dataset.X, dataset.y)
        
        return self
    
    
    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target in ['continuous']
    
    def priorize(self, candidate: Candidate = None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
