"""
[STEP] Learn : Gaussian NB
"""

from sklearn.naive_bayes import GaussianNB
from .act_base_learning import ActBaseLearning
from ...output import Input
from ...step import is_step, runner

@is_step('learning', 'tabular')
class ActGaussianNb(ActBaseLearning):
    """
    [STEP] Learn : Gaussian NB
    """
    name = "Learn : Gaussian NB"
    def __init__(self):
        self.configuration:dict = {}
        self.model:GaussianNB = None
        
    @runner
    def run(self, input_data: Input, callback=None): # pylint: disable=unused-argument
        """
        Fit GaussianNB on Input.dataset

        Args:
            input_data (Input): Fit data
            callback (callable, optional): Call after each step. Defaults to None.

        Returns:
            Output: Transformed input
        """
        self.model = GaussianNB()
        
        self.model.fit(input_data.dataset.X, input_data.dataset.y)
        
        return input_data.set_model(self)
    
    def suitable(self, input_data:Input) -> bool:
        return input_data.dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator']
