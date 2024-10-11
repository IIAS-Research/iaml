"""
[STEP] Learn : SurvivalTree
"""
import textwrap
from sksurv.tree import SurvivalTree
import numpy as np

from ....predictor import Predictor
from ....candidate import Candidate
from ....dataset import Dataset
from ....decorators.all import is_step

@is_step('predictor', 'tabular', 'survival')
class ActSurvivalTree(Predictor):
    """
    [STEP] Learn : SurvivalTree
    """
    name = "Learn : SurvivalTree"
    description = textwrap.dedent('''\
        SurvivalTree is a decision tree algorithm tailored for survival analysis. 
        It constructs a tree structure based on the log-rank test, where each split 
        is designed to separate data by survival times. The method handles censored data 
        and outputs risk scores, cumulative hazard functions, and survival functions 
        based on the tree's terminal nodes.''')
    description_long = textwrap.dedent('''\
        SurvivalTree builds a decision tree based on survival data, using the 
        log-rank splitting rule to determine the best splits. It is a non-parametric model that
        is particularly suited for survival analysis with right-censored data. The model provides
        both cumulative hazard and survival functions at each terminal node, making it useful for
        clinical risk prediction and other applications where time-to-event outcomes are crucial.
        ''')
    
    refs = [
        {
            'year': 1993,
            'name': 'Survival Trees by Goodness of Split',
            'authors': [
                'M. Leblanc',
                'J. Crowley'
            ],
            'doi': 'https://doi.org/10.1080/01621459.1993.10476284',
            'publisher': 'Journal of the American Statistical Association, 88(422), 457-467'
        }
    ]

    def __init__(self):
        self.configuration: dict = {
            'splitter': {
                'description': 'The strategy used to split at each node. \
                    Supported: "best", "random".',
                'default': 'best',
                'options': ['best', 'random'],
                'passthrough': False
            },
            'max_depth': {
                'description': 'The maximum depth of the tree.',
                'default': None,
                'range': [1, None],
                'passthrough': False
            },
            'min_samples_split': {
                'description': 'The minimum number of samples required to split an internal node.',
                'default': 6,
                'range': [2, 20],
                'passthrough': False
            },
            'min_samples_leaf': {
                'description': 'The minimum number of samples required to be at a leaf node.',
                'default': 3,
                'range': [1, 20],
                'passthrough': False
            },
            'min_weight_fraction_leaf': {
                'description': 'The minimum weighted fraction of the input samples \
                    required to be at a leaf node.',
                'default': 0.0,
                'range': [0.0, 0.5],
                'passthrough': False
            },
            'max_features': {
                'description': 'The number of features to consider when looking for the best \
                    split.',
                'default': None,
                'options': [None, 'auto', 'sqrt', 'log2'],
                'passthrough': False
            },
            'random_state': {
                'description': 'Controls the randomness of the estimator.',
                'default': None,
                'options': [None, 'int'],
                'passthrough': False
            },
            'max_leaf_nodes': {
                'description': 'Grow a tree with a maximum number of leaf nodes.',
                'default': None,
                'range': [None, 1000],
                'passthrough': False
            },
            'low_memory': {
                'description': 'Reduce memory usage but disable some prediction functions.',
                'default': False,
                'options': [True, False],
                'passthrough': False
            }
        }
        self.model: SurvivalTree = None
        
    def fit(self, dataset: Dataset):  # pylint: disable=unused-argument
        """
        Fit Survival Tree on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = SurvivalTree(
            **self.passthrough_parameters()
        )
        
        y = np.array(dataset.y, dtype=[('event', 'bool'), ('time', 'float')])
        self.model.fit(dataset.X, y)
        
        return self
    
    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'survival'

    def priorize(self, candidate: Candidate = None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5  # neutral
