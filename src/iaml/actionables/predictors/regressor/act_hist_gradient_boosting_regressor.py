"""[STEP] HistGradient Boosting Regressor"""
import textwrap
from typing import Any
from sklearn.ensemble import HistGradientBoostingRegressor
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step

# @is_step('predictor', 'tabular', 'regressor')
@is_step('disabled')
class ActHistGradientBoostingRegressor(Predictor):
    """[STEP] HistGradient Boosting Regressor"""

    name: str = "HistGradient Boosting Regressor"
    _description: str = textwrap.dedent('''\
        HistGradientBoostingRegressor is a machine learning algorithm
        that makes predictions for regression tasks using histogram-based gradient boosting.''')
    _description_long: str = textwrap.dedent('''\
        HistGradientBoostingRegressor is a type of gradient boosting
        algorithm that uses histogram-based decision trees to model the relationship between
        the input features and the output variable. It works by iteratively adding decision
        trees to the model, where each tree is trained to correct the errors made by the
        previous tree. The decision trees are constructed using histograms of the input
        features, which allows for faster computation and more efficient memory
        usage compared to other tree-based algorithms.
        HistGradientBoostingRegressor also includes options for regularization,
        such as L1 and L2 regularization, to prevent overfitting.''')
    _usage: str = "Use when you need fast nonlinear tabular regression; leaner than ActCatBoostRegressor or ActExtraTreesRegressor. Applicable to medium to large tabular continuous targets with mostly numeric features. Avoid when data is tiny, mostly linear, or interpretability is critical."
    refs: list[dict[str, Any]] = [
        {
            'year': 2006,
            'name': 'Gaussian Processes for Machine Learning',
            'authors': [
                'Carl Edward Rasmussen',
                'Christopher K. I. Williams'
            ],
            'doi': 'https://doi.org/10.7551/mitpress/3206.001.0001',
            'publisher': 'MIT Press 2006'
        }
    ]

    def __init__(self):
        self.configuration = {
            'l2_regularization': {
                'description': 'The L2 regularization parameter. \
                    Use 0 for no regularization (default).',
                'default': 1e-10,
                'range': [1e-10, 1.0]
                },
            'quantile': {
                'description': 'If loss is “quantile”, this parameter specifies which quantile to \
                    be estimated and must be between 0 and 1.',
                'default': 0.5,
                'range': [0.1, 1.0]
                },
            'learning_rate': {
                'description': 'The learning rate, also known as shrinkage.',
                'default': 0.1,
                'range': [0.01, 1.0]
                },
            'max_leaf_nodes': {
                'description': 'The maximum number of leaves for each tree.',
                'default': 31,
                'range': [3, 2048]
                },
            'min_samples_leaf': {
                'description': 'The minimum number of samples per leaf.',
                'default': 20,
                'range': [1, 200]
                },
            'loss': {
                'description': 'The loss function to use in the boosting process.',
                'default': "squared_error",
                'categorical': ["absolute_error", "poisson", "quantile", "squared_error"]
                },
            'n_iter_no_change': {
                'description': 'Used to determine when to “early stop”.',
                'default': 4,
                'range': [2, 15]
                },
            'tol': {
                'description': 'The absolute tolerance to use when comparing \
                    scores during early stopping',
                'default': 1e-4,
                'range': [1e-8, 1e-2]
                },
            'max_depth': {
                'description': 'The maximum depth of each tree',
                'default': 10,
                'range': [8, 25]
                }
            }
        self.model: HistGradientBoostingRegressor = None

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.model = HistGradientBoostingRegressor(early_stopping=True,
                                                   **self.passthrough_parameters())
        self.model.fit(dataset.X, dataset.y)
        return self

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'continuous'

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
