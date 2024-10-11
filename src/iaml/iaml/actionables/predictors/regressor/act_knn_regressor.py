"""
[STEP] Learn : KNN
"""
import textwrap
from sklearn.neighbors import KNeighborsRegressor
from ....predictor import Predictor
from ....candidate import Candidate
from ....dataset import Dataset
from ....decorators.all import is_step

@is_step('predictor', 'tabular', 'fast_predictor', 'regressor')
class ActKNNRegressor(Predictor):
    """
    [STEP] Learn : KNN
    """
    name = "Learn : KNN"
    description = textwrap.dedent('''\
        KNeighborsRegressor is a machine learning algorithm that makes 
        predictions for regression tasks using k-nearest neighbors.''')
    description_long = textwrap.dedent('''\
        KNeighborsRegressor is a type of instance-based learning 
        algorithm that makes predictions for new input features based on the values of 
        the k-nearest neighbors in the training data. It works by calculating the distance 
        between the new input features and all the training data, and then selecting 
        the k-nearest neighbors based on that distance. The output variable for the new 
        input features is then calculated as the average of the output variables 
        for the k-nearest neighbors.''')
    
    refs = [
        {
            'year': 1951,
            'name': 'Discriminatory Analysis, Nonparametric Discrimination: Consistency Properties',
            'authors': [
                'Evelyn Fix',
                'Joseph Lawson Hodges Jr.'
            ],
            'doi': '',
            'publisher': 'Technical Report 4, USAF School of Aviation Medicine, Randolph Field'
        },
        {
            'year': 1967,
            'name': 'Nearest neighbor pattern classification',
            'authors': [
                'Thomas M. Cover',
                'Peter E. Hart'
            ],
            'doi': 'https://doi.org/10.1109/TIT.1967.1053964',
            'publisher': 'IEEE Transactions on Information Theory. 13: page 21--27'
        }
    ]
    def __init__(self):
        self.configuration:dict = {
            'metric': {
                'description': 'Can be minkowski or manhattan',
                'default': 'minkowski',
                'categorical': ['minkowski', 'manhattan']
            },
            'n_neighbors': {
                'description': 'Number of neighbors',
                'default': 5,
                'range': [1, 200],
                'passthrough': False
            },
            'weights': {
                'description': 'Weight function used in prediction.',
                'default': 'uniform',
                'categorical': ['uniform', 'distance']
            }
        }
        self.model:KNeighborsRegressor = None
        
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit Knn regressor on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = KNeighborsRegressor(
            n_neighbors = min(self.get_config('n_neighbors'), dataset.X.shape[0]),
            **self.passthrough_parameters()
            )
        
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
        return dataset.type_of_target in ['continuous']

    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
