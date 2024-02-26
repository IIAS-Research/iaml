from sklearn.model_selection import ShuffleSplit

from ...output import Input, Output
from ...step import isStep, runner, Step
from .dataset_wrapper import DatasetWrapper


@isStep('wrapper')
class RandomSplit(DatasetWrapper):
    name = "Split date to train and test set"
    def __init__(self, step: Step):
        self.configurations = [{
            'ratio': {
                'description': 'Split ratio',
                'default': 0.2
            },
            'random_state': {
                'description': 'Random state',
                'default': 42
            }
        }]

        super().__init__(step)
        
    @runner
    def run(self, input: Input, callback=None) -> Output:
        test_size = self.get_config('ratio')
        random_state = self.get_config('random_state')

        training_input = next(input.to_training_inputs(
            ShuffleSplit(1, test_size=test_size, random_state=random_state).split,
            input.dataset.X
        ))

        return self.step.run(training_input)
    
    def priorize(self, input=None):
        return 1
