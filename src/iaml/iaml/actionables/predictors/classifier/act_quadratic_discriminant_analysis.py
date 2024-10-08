"""
[STEP] Learn : Quadratic Discriminant Analysis
"""

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step
import textwrap

@is_step('predictor', 'tabular', 'classifier')
class ActQuadraticDiscriminantAnalysis(Predictor):
    """
    [STEP] Learn : Quadratic Discriminant Analysis
    """
    name = textwrap.dedent("Learn : Quadratic Discriminant Analysis")
    description = textwrap.dedent('''QuadraticDiscriminantAnalysis is a machine learning algorithm 
        that models the relationship between input features and a categorical 
        output variable using a quadratic function.''')
    description_long = textwrap.dedent('''QuadraticDiscriminantAnalysis is a type of discriminant analysis 
        algorithm that models the relationship between input features and a categorical output 
        variable using a quadratic function. 
        It works by estimating the covariance matrices of the classes and using 
        them to calculate the probability density functions for each class. 
        The class with the highest probability density function is then used to make 
        the prediction.''')
    refs = [
        {
            'year': 1965,
            'name': 'Geometrical and Statistical Properties of Systems of Linear Inequalities \
                with Applications in Pattern Recognition',
            'authors': ['Thomas M. Cover'],
            'doi': 'https://doi.org/10.1109/PGEC.1965.264137',
            'publisher': 'IEEE Transactions on Electronic Computers Vol.EC-14 page 326--334'
            
        },
        {
            'year': 2016,
            'name': 'Linear vs. quadratic discriminant analysis classifier: a tutorial',
            'authors': ['Alaa Tharwat'],
            'doi': 'https://www.inderscienceonline.com/doi/abs/10.1504/IJAPR.2016.079050',
            'publisher': 'International Journal of Applied Pattern Recognition \
                Vol.3, No.2 page 145--180'
        }
    ]
    def __init__(self):
        self.configuration:dict = {
            'reg_param': {
                'description': 'Regularizes the per-class covariance estimates by transforming S2',
                'default': 0.0001,
                'range': [0.0001, 1.0]
                }
            }
        self.model:QuadraticDiscriminantAnalysis = None
    
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit Quadratic Discriminant Analysis on Candidate.dataset

        Args:
            dataset (Candidate): Fit data

        Returns:
            Fitted step
        """
        self.model = QuadraticDiscriminantAnalysis(**self.passthrough_parameters())
        
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
