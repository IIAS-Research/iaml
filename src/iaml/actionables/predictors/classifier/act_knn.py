
"""[STEP]  KNN"""
import textwrap
from typing import Any
from sklearn.neighbors import KNeighborsClassifier
from ....predictor import Predictor
from ....candidate import Candidate
from ....dataset import Dataset
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'classifier')
class ActKNN(Predictor):
    """[STEP]  KNN"""

    name: str = "KNN"
    _description: str = textwrap.dedent('''\
        KNeighborsClassifier is a machine learning algorithm that makes
        predictions for classification tasks using k-nearest neighbors.''')
    _description_long: str = textwrap.dedent('''\
        KNeighborsClassifier is a type of instance-based learning
        algorithm that makes predictions for new input features based on the labels
        of the k-nearest neighbors in the training data.
        It works by calculating the distance between the new input features and all
        the training data, and then selecting the k-nearest neighbors based on that distance
        The label for the new input features is then determined by a majority vote of the
        labels of the k-nearest neighbors.''')
    refs: list[dict[str, Any]] = [
        {
            'year': 1951,
            'name': 'Discriminatory Analysis, Nonparametric Discrimination: Consistency Properties',
            'authors': [
                'Evelyn Fix',
                'Joseph Lawson Hodges Jr.'
            ],
            'doi': 'https://doi.org/10.2307/1403797',
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
        self.configuration = {
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
        self.model: KNeighborsClassifier = None

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.model = KNeighborsClassifier(
            n_neighbors = min(self.get_config('n_neighbors'), dataset.X.shape[0]),
            **self.passthrough_parameters()
            )
        self.model.fit(dataset.X, dataset.y)
        return self

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator']

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
