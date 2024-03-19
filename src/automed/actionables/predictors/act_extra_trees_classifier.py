"""
[STEP] Learn :  Extra Trees Classifier
"""
from sklearn.ensemble import ExtraTreesClassifier
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('predictor', 'tabular')
class ActExtraTreesClassifier(Predictor):
    """
    [STEP] Learn :  Extra Trees Classifier
    """
    name = "Learn : Extra Trees Classifier" 
    def __init__(self):
        self.configuration:dict = {
            'max_depth': {
                'description': 'Max depth of each tree',
                'default': 15,
                'range': [1, 100]
            },
            'min_samples_leaf': {
                'description': 'The minimum number of samples required to be at a leaf node.',
                'default': 1,
                'range': [1, 15]
            },
            'min_samples_split': {
                'description': 'The minimum number of samples required to split an internal node',
                'default': 2,
                'range': [2, 20]
            },
            'bootstrap': {
                'description': 'Whether bootstrap samples are used when building trees. \
                    If False, the whole dataset is used to build each tree.',
                'default': False
            },
            'max_features': {
                'description': 'The number of features to consider when looking for the best split',
                'default': 'sqrt',
                'categorical': ['sqrt', 'log2']
            },
            'criterion': {
                'description': 'The function to measure the quality of a split.',
                'default': "gini",
                'categorical': ['gini', 'entropy']
            },
            'n_estimators': {
                'description': 'Number of threes',
                'default': 100,
                'range': [1, 500]
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            }
        }
        self.model:ExtraTreesClassifier = None
        
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit Extra Trees Classifier on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = ExtraTreesClassifier(**self.model_parameters())
        
        self.model.fit(dataset.X, dataset.y)
        
        return self
    
    def suitable(self, candidate) -> bool:
        return candidate.dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator']

    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
