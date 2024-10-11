"""
[STEP] Learn :  XGBoost
"""
import textwrap
from sklearn.ensemble import GradientBoostingClassifier
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step

@is_step('predictor', 'tabular', 'classifier')
class ActXGBoost(Predictor):
    """
    [STEP] Learn :  XGBoost
    """
    name = "Learn : XGBoost"
    description = textwrap.dedent('''\
        GradientBoostingClassifier is a machine learning algorithm that models the 
        relationship between input features and a categorical output variable using 
        gradient boosting.''')
    description_long = textwrap.dedent('''\
        It works by building multiple decision trees in a sequential manner, 
        where each tree is trained to correct the errors made by the previous tree. 
        The final prediction is made by summing the predictions of all the trees.''')
    
    refs = [
        {
            'name': 'Stochastic Gradient Boosting',
            'year': 1999,
            'authors': [
              'Jerome H. Friedman'  
            ],
            'doi': 'https://doi.org/10.1016/S0167-9473(01)00065-2',
            'publisher': 'Computational Statistics & Data Analysis, Vol.38, No.4 page 367--378'
        },
        {
            'year': 2001,
            'name': 'Greedy Function Approximation: A Gradient Boosting Machine',
            'authors': [
              'Jerome H. Friedman'  
            ],
            'doi': 'https://doi.org/10.1214/aos/1013203451',
            'publisher': 'The Annals of Statistics, Vol.29, No.5 page 1189--1232'
        },
        {
            'year': 2009,
            'name': 'The Elements of Statistical Learning',
            'authors': [
                'Trevor Hastie',
                'Robert Tibshirani',
                'Jerome H. Friedman'  
            ],
            'doi': 'https://doi.org/10.1007/978-0-387-84858-7',
            'publisher': 'Springer New York'
        }
    ]
    def __init__(self):
        self.configuration:dict = {
            'max_depth': {
                'description': 'Max depth of each tree',
                'default': 15,
                'range': [1, 100]
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            },
            'learning_rate': {
                'description': 'Learning rate',
                'default': 0.1,
                'range': [0.0000001, 5]
            },
            'subsample': {
                'description': 'he fraction of samples to be used for fitting \
                    the individual base learners.',
                'default': 1.0,
                'range': [0.1, 1.0]
            },
            'n_estimators': {
                'description': 'Number of estimators',
                'default': 100,
                'range': [1, 500]
            },
            'loss': {
                'description': 'The loss function to use in the boosting process.',
                'default': "log_loss",
                'categorical': ['log_loss', 'exponential']
            },
            'criterion': {
                'description': 'The function to measure the quality of a split',
                'default': "friedman_mse",
                'categorical': ['friedman_mse', 'squared_error']
            },
            'min_samples_leaf': {
                'description': 'The minimum number of samples required to be at a leaf node.',
                'default': 1,
                'range': [1, 15]
            },
            'max_features': {
                'description': 'The number of features to consider when looking for the best split',
                'default': 1,
                'range': [0.1, 1]
            },
            'min_samples_split': {
                'description': 'The minimum number of samples required to split an internal node',
                'default': 2,
                'range': [2, 20]
            }
        }
        self.model:GradientBoostingClassifier = None
        
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit XgBoost on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        if dataset.type_of_target == 'binary':
            self.configuration['loss']['categorical'] = ['log_loss', 'exponential']
        else:
            self.configuration['loss']['categorical'] = ['log_loss']
        self.check_configuration()
        
        self.model = GradientBoostingClassifier(**self.passthrough_parameters())
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
    
