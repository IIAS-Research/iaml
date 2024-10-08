"""
[STEP] Learn : Bernoulli NB
"""

from sklearn.naive_bayes import BernoulliNB
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step
import textwrap

@is_step('predictor', 'tabular', 'classifier')
class ActBernoulliNb(Predictor):
    """
    [STEP] Learn : Bernoulli NB
    """
    name = textwrap.dedent("Learn : Bernoulli NB")
    description = textwrap.dedent('''BernoulliNB is a tool that helps computers predict categories 
        by analyzing binary features, even if the input isn't strictly binary.''')
    description_long = textwrap.dedent('''BernoulliNB is a type of Naive Bayes classifier specifically 
        designed for binary features. While it's primarily meant for binary inputs,
        scikit-learn implements it in a way that can handle non-binary data.''')
    refs = [
        {
            'year': 1998,
            'name': 'A Comparison of Event Models for Naive Bayes Text Classification',
            'authors': [
                'Andrew McCallum',
                'Kamal Nigam'
            ],
            'doi': None,
            'publisher': (
                'AAAI-98 workshop on learning for text categorization, '
                '752, page 41--48. (1998)'
            )
        },
        {
            'year': 2006,
            'name': 'Spam Filtering with Naive Bayes - Which Naive Bayes?',
            'authors': [
                'Vangelis Metsis',
                'Ion Androutsopoulos',
                'Georgios Paliouras'
            ],
            'doi': '',
            'publisher': (
                'The Third Conference on Email and Anti-Spam 2006 (CEAS)'
            )
        }
    ]
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
        self.model:BernoulliNB = None

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit BernoulliNB on Candidate.dataset

        Args:
            dataset (Candidate): Fit data

        Returns:
            Fitted step
        """
        self.model = BernoulliNB(**self.passthrough_parameters())
        
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
