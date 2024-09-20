"""
[STEP] Learn :  Logistic Regression Classifier
"""
from sklearn.linear_model import LogisticRegression
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('predictor', 'tabular', 'fast_predictor', 'classifier')
class ActLogisticRegression(Predictor):
    """
    [STEP] Learn :  Logistic Regression Classifier
    """
    name = "Learn : Logistic Regression Classifier"
    description = '''LogisticRegression is a machine learning algorithm 
        that models the relationship between input features and a binary 
        output variable using a logistic function.'''
    description_long = '''LogisticRegression is a type of classification algorithm that models 
        the relationship between input features and a binary 
        output variable using a logistic function. 
        It works by finding the best-fitting line or hyperplane that 
        maximizes the likelihood of the observed output variables given the input features.'''
    refs = [
        {
            'year': 1944,
            'name': 'Application of the Logistic Function to Bio-Essay',
            'authors': [
                'Joseph Berkson'
            ],
            'doi': 'https://doi.org/10.2307/2280041',
            'publisher': (
                'Journal of the American Statistical Association '
                'Vol. 39, No. 227, page 357--365'
            )
        },
        {
            'year': 1951,
            'name': 'Why I Prefer Logits to Probits',
            'authors': [
                'Joseph Berkson'
            ],
            'doi': 'https://doi.org/10.2307/3001655',
            'publisher': (
                'Biometrics '
                'Vol. 7, No. 4, page 327--339'
            )
        }
    ]
    def __init__(self):
        self.configuration:dict = {
            'random_state': {
                'description': 'random_state',
                'default': 42
            },
            'penalty': {
                'description': 'Specify the norm of the penalty',
                'default': 'l2',
                'categorical': ['l2', None]
            },
            'tol': {
                'description': 'The stopping criterion.',
                'default': 0.0001,
                'range': [1e-05, 0.1]
            },
            'class_weight': {
                'description': 'The “balanced” mode uses the values of y to \
                    automatically adjust weights inversely proportional to class frequencies ',
                'default': None,
                'categorical': [None, 'balanced']
            }
        }
        self.model:LogisticRegression = None
    
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit Logistic Regression on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = LogisticRegression(
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
