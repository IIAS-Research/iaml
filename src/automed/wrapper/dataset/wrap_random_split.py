"""
[WRAPPER] Wrap learning step to apply Random Split on Dataset
"""
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from ...candidate import Candidate
from ...step import Step
from ...decorators.all import is_step, runner
from .wrap_dataset_wrapper import WrapDatasetWrapper


@is_step('wrapper')
class WrapRandomSplit(WrapDatasetWrapper):
    """
    [WRAPPER] Wrap learning step to apply Random Split on Dataset
    """
    name = "Splits the dataset into train and test sets randomly."
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
    def run(self, candidate: Candidate, callback:callable=None) -> Candidate: # pylint: disable=unused-argument
        test_size = self.get_config('ratio')
        random_state = self.get_config('random_state')
        
        def splitter(X:pd.DataFrame, y:pd.DataFrame) -> ShuffleSplit: # pylint: disable=unused-argument
            return ShuffleSplit(1, test_size=test_size, random_state=random_state).split(X)
        
        train_dataset, test_dataset = next(candidate.dataset.split(splitter))
        candidate = self.step.run(candidate.to_input(dataset=train_dataset), callback=callback)
        candidate.evaluate(test_dataset)

        self.explanations = ['Trained one model.']
        
        return super().run(candidate, callback)
    
