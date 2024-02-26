from sklearn.model_selection import KFold as SKKFold, StratifiedKFold

from ...output import Input, Output
from ...step import isStep, runner, Step
from .dataset_wrapper import DatasetWrapper


@isStep('wrapper')
class KFold(DatasetWrapper):
    name = "Split date to train and test set"
    def __init__(self, step: Step):
        self.configurations = [{
            'folds': {
                'description': 'Split ratio',
                'default': 5,
            },
            'stratify': {
                'description': 'Whether to run stratified K-Fold',
                'default': True,
            },
        }]

        super().__init__(step)
        
    @runner
    def run(self, input: Input, callback=None) -> Output:
        if self.get_config('stratify'):
            kfold = StratifiedKFold(self.get_config('folds'))
            inputs = input.to_training_inputs(kfold.split, input.dataset.X, input.dataset.Y)
        else:
            kfold = SKKFold(self.get_config('folds'))
            inputs = input.to_training_inputs(kfold.split, input.dataset.X)

        outputs = []
        for training_input in inputs:
            outputs.extend(self.step.run(training_input))

        return sorted(outputs, key=lambda o: o.evaluate())[-1]
    
    def priorize(self, input=None):
        return 1
