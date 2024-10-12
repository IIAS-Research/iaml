"""
[STEP] Learn :  Linear Regression
"""
import textwrap
from sklearn.linear_model import LinearRegression
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step

@is_step('predictor', 'tabular', 'fast_predictor', 'regressor', 'baseline_predictor')
class ActLinearRegression(Predictor):
    """
    [STEP] Learn :  Linear Regression
    """
    name = "Learn : Linear Regression"
    description = textwrap.dedent('''\
        LinearRegression is a machine learning algorithm that models the 
        relationship between input features and a continuous output variable using 
        a linear function.''')
    description_long = textwrap.dedent('''\
        LinearRegression is a type of regression algorithm that models 
        the relationship between input features and a continuous output variable using 
        a linear function. It works by finding the best-fitting line or hyperplane 
        that minimizes the sum of the squared differences between the predicted 
        and actual output variables.''')
    
    refs = [
        {
            'year': 1875,
            'name': 'No information provided',
            'authors': [
                'Sir Francis Galton'
            ],
            'doi': '',
            'publisher': ''
        }
    ]
    def __init__(self):
        self.configuration:dict = {}
        self.model:LinearRegression = None
        
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit Linear regression on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = LinearRegression()
        self.model.fit(dataset.X, dataset.y)
        
        return self
    
    
    def suitable(self, dataset:Dataset) -> bool:
        return dataset.type_of_target in ['continuous']
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
