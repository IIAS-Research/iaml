from sklearn.model_selection import ShuffleSplit

from ...output import Input, Output
from ...step import is_step, runner, Step
from .wrap_dataset_wrapper import WrapDatasetWrapper


@is_step('wrapper')
class WrapRandomSplit(WrapDatasetWrapper):
    name = "Split date to train and test set"
    def __init__(self, step: Step):
        self.configurations = [{
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
        }]

        super().__init__(step)
        
    @runner
    def run(self, input_data: Input, callback=None) -> Output:
        test_size = self.get_config('ratio')
        random_state = self.get_config('random_state')
        
        splitter = lambda X, y: ShuffleSplit(1, test_size=test_size, random_state=random_state).split(X)
        
        train_dataset, test_dataset = next(input_data.dataset.split(splitter))
        output = self.step.run(input_data.to_input(dataset=train_dataset), callback=callback)
        output.evaluate(test_dataset)
        
        return output
    
