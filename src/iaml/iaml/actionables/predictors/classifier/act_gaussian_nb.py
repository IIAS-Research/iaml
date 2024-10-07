"""
[STEP] Learn : Gaussian NB
"""

from sklearn.naive_bayes import GaussianNB
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step

@is_step('predictor', 'tabular', 'classifier')
class ActGaussianNb(Predictor):
    """
    [STEP] Learn : Gaussian NB
    """
    name = "Learn : Gaussian NB"
    description = '''GaussianNB is a machine learning algorithm that makes predictions 
        based on the Gaussian (normal) distribution of the input features.'''
    description_long = '''GaussianNB is a type of naive Bayes classifier that assumes the 
        input features are independent and follow a Gaussian (normal) distribution. 
        It uses Bayes' theorem to calculate the probability of each class given the 
        input features and then makes a prediction based on the highest probability. 
        GaussianNB is particularly useful when the input features have a continuous 
        distribution and can be modeled well by a normal distribution. 
        It is a simple and fast algorithm that works well for many classification problems, 
        especially when the number of features is much larger than the number of samples.'''    
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
