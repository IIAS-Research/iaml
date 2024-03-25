"""
[STEP] Learn : Multinomial NB
"""

from sklearn.naive_bayes import MultinomialNB
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('predictor', 'tabular')
class ActMultinomialNB(Predictor):
    """
    [STEP] Learn : Multinomial NB
    """
    name = "Learn : Multinomial NB"
    def __init__(self):
        self.configuration:dict = {
            'alpha': {
                'description': 'Additive (Laplace/Lidstone) \
                    smoothing parameter (set alpha=0 and force_alpha=True, for no smoothing).',
                'default': 1.0,
                'range': [0.01, 100.0]
                },
            'fit_prior': {
                'description': 'Whether to learn class prior probabilities \
                    or not. If false, a uniform prior will be used.',
                'default': True,
                'categorical': [True, False]
                }
            }
        self.model:MultinomialNB = None
    
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit MultinomialNB on Candidate.dataset

        Args:
            dataset (Candidate): Fit data

        Returns:
            Fitted step
        """
        self.model = MultinomialNB(**self.passthrough_parameters())
        
        self.model.fit(dataset.X, dataset.y)
        
        return self
    
    
    def suitable(self, candidate: Candidate) -> bool:
        """
        Does this step suitable for this candidate ?
        Dataset must contain only positive values

        Args:
            candidate (Candidate): Suitable for this candidate

        Returns:
            bool: Suitable ?
        """
        # Negative values are not supported
        return not((candidate.dataset.X < 0).any().any() or (candidate.dataset.y < 0).any()) \
            and candidate.dataset.type_of_target in ['binary', 'multiclass']
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
