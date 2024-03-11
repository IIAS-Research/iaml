"""
[STEP] Learn : KNN
"""
from sklearn.neighbors import KNeighborsRegressor
from .act_base_learning import ActBaseLearning
from ...output import Input
from ...step import is_step, runner


@is_step('learning', 'tabular', 'fast_learning')
class ActKNNRegressor(ActBaseLearning):
    """
    [STEP] Learn : KNN
    """
    name = "Learn : KNN"
    def __init__(self):
        self.configuration:dict = {
            'metric': {
                'description': 'Can be minkowski or manhattan',
                'default': 'minkowski',
                'categorical': ['minkowski', 'manhattan']
            },
            'n_neighbors': {
                'description': 'Number of neighbors',
                'default': 5,
                'range': [1, float('inf')]
            }
        }
        self.model:KNeighborsRegressor = None
        
    @runner
    def run(self, input_data: Input, callback=None): # pylint: disable=unused-argument
        """
        Fit Knn regressor on Input.dataset

        Args:
            input_data (Input): Fit data
            callback (callable, optional): Call after each step. Defaults to None.

        Returns:
            Output: Transformed input
        """
        self.model = KNeighborsRegressor(
            n_neighbors = self.get_config('n_neighbors'),
            metric = self.get_config('metric')
            )
        
        self.model.fit(input_data.dataset.X, input_data.dataset.y)
        
        return input_data.set_model(self)
    
    def suitable(self, input_data: Input) -> bool:
        return input_data.dataset.type_of_target == 'continuous'
