"""
[STEP] SGD Regressor
"""
import textwrap
from sklearn.linear_model import SGDRegressor
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step

@is_step('predictor', 'tabular', 'regressor')
class ActSGDRegressor(Predictor):
    """
    [STEP] SGD Regressor
    """
    name = "SGD Regressor"
    description = textwrap.dedent('''\
        SGDRegressor is a machine learning algorithm that models 
        the relationship between input features and a continuous output variable 
        using stochastic gradient descent.''')
    description_long = textwrap.dedent('''\
        SGDRegressor is a type of linear model that models the relationship 
        between input features and a continuous output variable using stochastic gradient descent. 
        It works by iteratively updating the model parameters in the direction of the negative 
        gradient of the loss function with respect to the parameters, using a single example 
        at a time.''')
    refs = [
        {
            'year': 1951,
            'name': 'A Stochastic Approximation Method',
            'authors': [
                'Herbert Robbins',
                'Sutton Monro'    
            ],
            'doi': 'https://doi.org/10.1214/aoms/1177729586',
            'publisher': 'The annals of Mathematical Statistics Vol.22 No.3 page 400--407'
        }
    ]
    def __init__(self):
        self.configuration:dict = {
            'alpha': {
                'description': 'Constant that multiplies the regularization term.',
                'default': 0.0001,
                'range': [1e-07, 0.1]
                },
            'tol': {
                'description': 'The stopping criterion.',
                'default': 0.0001,
                'range': [1e-05, 0.1]
                },
            'epsilon': {
                'description': 'Epsilon in the epsilon-insensitive loss functions',
                'default': 0.1,
                'range': [1e-05, 0.1]
                },
            'eta0': {
                'description': 'The initial learning rate for the ‘constant’, ‘invscaling’ \
                    or ‘adaptive’ schedules.',
                'default': 0.01,
                'range': [1e-07, 0.1]
                },
            'l1_ratio': {
                'description': 'The Elastic Net mixing parameter',
                'default': 0.15,
                'range': [1e-09, 1.0]
                },
            'power_t': {
                'description': 'The exponent for inverse scaling learning rate.',
                'default': 0.25,
                'range': [1e-05, 1.0]
                },
            'average': {
                'description': 'When set to True, computes the averaged SGD \
                    weights across all updates and stores the result in the coef_ attribute. ',
                'default': False
                },
            'loss': {
                'description': 'The loss function to be used.',
                'default': "squared_error",
                'categorical': ["squared_error",
                                "huber",
                                "epsilon_insensitive",
                                "squared_epsilon_insensitive"]
                },
            'penalty': {
                'description': 'The penalty (aka regularization term) to be used.',
                'default': "l2",
                'categorical': ["l1", "l2", "elasticnet"]
                }
            }
        self.model:SGDRegressor = None
    
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit SGDRegressor on Candidate.dataset

        Args:
            dataset (Candidate): Fit data

        Returns:
            Fitted step
        """
        self.model = SGDRegressor(**self.passthrough_parameters())
        
        self.model.fit(dataset.X, dataset.y)
        
        return self
    
    
    def suitable(self, dataset:Dataset) -> bool:
        """
        Does this step suitable for this candidate

        Args:
            candidate (Candidate): Suitable for this candidate

        Returns:
            bool: Suitable ?
        """
        return dataset.type_of_target == 'continuous'
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
