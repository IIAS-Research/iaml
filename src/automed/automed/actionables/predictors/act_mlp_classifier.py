"""
[STEP] Learn : MLP Classifier
"""

from sklearn.neural_network import MLPClassifier
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('predictor', 'tabular', 'classifier')
class ActMLPClassifier(Predictor):
    """
    [STEP] Learn : MLP Classifier
    """
    name = "Learn : MLP Classifier"
    def __init__(self):
        self.configuration:dict = {
            'activation': {
                'description': 'Activation function for the hidden layer.',
                'default': 'relu',
                'categorical': ["tanh", "relu"]
                },
            'alpha': {
                'description': 'Strength of the L2 regularization term. The L2 regularization \
                    term is divided by the sample size when added to the loss.',
                'default': 0.0001,
                'range': [1e-07, 0.1]
                },
            'hidden_layer_count': {
                'description': 'NUmber of hidden layer',
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
        self.model:MLPClassifier = None
    
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit MLP Classifier on Candidate.dataset

        Args:
            dataset (Candidate): Fit data

        Returns:
            Fitted step
        """
        self.model = MLPClassifier(
            hidden_layer_sizes=[self.get_config('node_per_layer') \
                for i in range(self.get_config('hidden_layer_count'))],
            early_stopping=True,
            max_iter=400,
            **self.passthrough_parameters())
        
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
        return dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator']
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
