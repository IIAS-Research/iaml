"""[STEP] MLP Regressor"""
from typing import Any
import textwrap
from sklearn.neural_network import MLPRegressor
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step

@is_step('predictor', 'tabular', 'regressor')
class ActMLPRegressor(Predictor):
    """[STEP] MLP Regressor"""

    name: str = "MLP Regressor"
    _description: str = textwrap.dedent('''\
        MLPRegressor is a machine learning algorithm that models the
        relationship between input features and a continuous output variable using
        a multi-layer perceptron neural network.''')
    _description_long: str = textwrap.dedent('''\
        MLPRegressor is a type of neural network algorithm that models
        the relationship between input features and a continuous output variable using a
        multi-layer perceptron (MLP) neural network.
        It works by transforming the input features through one or more hidden
        layers with non-linear activation functions, and then using a final layer with
        a linear activation function to output a continuous value.''')
    refs: list[dict[str, Any]] = [
        {
            'year': 1989,
            'name': 'Connectionist Learning Procedures',
            'authors': [
                'Geoffrey E. Hinton'
            ],
            'doi': 'https://doi.org/10.1016/0004-3702(89)90049-0',
            'publisher': 'Artificial intelligence Vol. 40.1 page 185--234'
        },
        {
            'year': 2010,
            'name': 'Understanding the difficulty of training deep feedforward neural networks',
            'authors': [
                'Xavier Glorot',
                'Yoshua Bengio'
            ],
            'doi': '',
            'publisher': (
                'Proceedings of the Thirteenth International Conference on '
                'Artificial Intelligence and Statistics page 249--256'
            )
        }
    ]

    def __init__(self):
        self.configuration = {
            'activation': {
                'description': 'Activation function for the hidden layer.',
                'default': 'relu',
                'categorical': ["tanh", "relu"]
            },
            'alpha': {
                'description': textwrap.dedent('''\
                    Strength of the L2 regularization term. The L2
                    regularization term is divided by the sample size when
                    added to the loss.'''),
                'default': 0.0001,
                'range': [1e-07, 0.1]
            },
            'hidden_layer_count': {
                'description': 'Number of hidden layer',
                'default': 1,
                'range': [1, 4],
                'passthrough': False
            },
            'node_per_layer': {
                'description': 'Number of node per layer',
                'default': 32,
                'range': [16, 256],
                'passthrough': False
            },
            'learning_rate_init': {
                'description': 'Learning rate schedule for weight updates',
                'default': 0.001,
                'range': [0.0001, 0.5]
            }
        }

        self.model: MLPRegressor = None

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.model = MLPRegressor(
            hidden_layer_sizes=[self.get_config('node_per_layer') \
                for i in range(self.get_config('hidden_layer_count'))],
            early_stopping=True,
            max_iter=400,
            **self.passthrough_parameters())

        self.model.fit(dataset.X, dataset.y)

        return self

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == "continuous"

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
