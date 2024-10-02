"""
[STEP] Learn : Linear Discriminant Analysis
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step

@is_step('predictor', 'tabular', 'classifier')
class ActLinearDiscriminantAnalysis(Predictor):
    """
    [STEP] Learn : Linear Discriminant Analysis
    """
    name = "Learn : Linear Discriminant Analysis"
    description = '''LinearDiscriminantAnalysis is a machine learning algorithm that finds
        a linear combination of features that maximizes the separation between 
        classes for classification tasks.'''
    description_long = '''LinearDiscriminantAnalysis is a type of dimensionality reduction 
        algorithm that finds a linear combination of features that maximizes the 
        separation between classes for classification tasks. 
        It works by calculating the within-class and between-class scatter matrices, 
        and then finding the directions in the feature space that maximize the ratio of 
        the between-class scatter to the within-class scatter.'''
    refs = [
        {
            'year': 1936,
            'name': 'The Use of Multiple Measurements in Taxonomic Problems',
            'authors': [
                'Sir Ronald Aylmer Fisher'
            ],
            'doi': '',
            'publisher': 'Annals of Eugenics Vol.7 page 179--188'
        }
    ]
    def __init__(self):
        self.configuration:dict = {
            'tol': {
                'description': 'Absolute threshold for a singular value of X to be considered \
                    significant, used to estimate the rank of X. ',
                'default': 0.0001,
                'range': [1e-05, 0.1]
                }
            }
        self.model:LinearDiscriminantAnalysis = None
    
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit Linear Discriminant Analysis on Candidate.dataset

        Args:
            dataset (Candidate): Fit data

        Returns:
            Fitted step
        """
        self.model = LinearDiscriminantAnalysis(**self.passthrough_parameters())
        
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
