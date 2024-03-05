"""
[WRAPPER] Wrap learning step to apply Random Split on Dataset
"""
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from ...output import Input, Output
from ...step import is_step, runner, Step
from .wrap_dataset_wrapper import WrapDatasetWrapper


@is_step('wrapper')
class WrapRandomSplit(WrapDatasetWrapper):
    """
    [WRAPPER] Wrap learning step to apply Random Split on Dataset
    """
    name = "Split date to train and test set"
    def __init__(self, step: Step):
        self.configuration:dict = {
            'ratio': {
                'description': 'Split ratio',
                'default': 0.2,
                'no_gridsearch': True,
            },
            'random_state': {
                'description': 'Random state',
                'default': 42,
                'no_gridsearch': True,
            },
        }
        
    @runner
    def run(self, input_data: Input, callback:callable=None) -> Output: # pylint: disable=unused-argument
        test_size = self.get_config('ratio')
        random_state = self.get_config('random_state')
        
        def splitter(X:pd.DataFrame, y:pd.DataFrame) -> ShuffleSplit: # pylint: disable=unused-argument
            return ShuffleSplit(1, test_size=test_size, random_state=random_state).split(X)
        
        train_dataset, test_dataset = next(input_data.dataset.split(splitter))
        output = self.step.run(input_data.to_input(dataset=train_dataset), callback=callback)
        output.evaluate(test_dataset)
        
        return output
    
