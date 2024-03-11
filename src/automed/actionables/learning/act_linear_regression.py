"""
[STEP] Learn :  Linear Regression
"""
from sklearn.linear_model import LinearRegression
from .act_base_learning import ActBaseLearning
from ...output import Input
from ...step import is_step, runner


@is_step('learning', 'tabular', 'fast_learning')
class ActLinearRegression(ActBaseLearning):
    """
    [STEP] Learn :  Linear Regression
    """
    name = "Learn : Linear Regression"
    def __init__(self):
        self.configuration:dict = {}
        self.model:LinearRegression = None
        
    @runner
    def run(self, input_data: Input, callback=None): # pylint: disable=unused-argument
        """
        Fit Linear regression on Input.dataset

        Args:
            input_data (Input): Fit data
            callback (callable, optional): Call after each step. Defaults to None.

        Returns:
            Output: Transformed input
        """
        self.model = LinearRegression()
        self.model.fit(input_data.dataset.X, input_data.dataset.y)
        
        return input_data.set_model(self)
    
    def suitable(self, input_data: Input) -> bool:
        return input_data.dataset.type_of_target == 'continuous'
