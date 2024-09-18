"""
[STEP] Learn : Gaussian NB
"""

from sklearn.naive_bayes import GaussianNB
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('predictor', 'tabular', 'classifier')
class ActGaussianNb(Predictor):
    """
    [STEP] Learn : Gaussian NB
    """
    name = "Learn : Gaussian NB"
    refs = [
        {
            'year': 1763,
            'name': None,
            'authors': [
            ],
            'doi': None,
            'publisher': None
        }
    ]
    def __init__(self):
        self.configuration:dict = {
            'var_smoothing': {
                'description': 'Portion of the largest variance of all \
                    features that is added to variances for calculation stability.',
                'default': 1e-9,
                'range': [1e-11, 1e-4]
            },
        }
        self.model:GaussianNB = None
    
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit GaussianNB on Candidate.dataset

        Args:
            dataset (Candidate): Fit data

        Returns:
            Fitted step
        """
        self.model = GaussianNB(**self.passthrough_parameters())
        
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
