import numpy as np

from sklearn.model_selection import KFold as SKKFold, StratifiedKFold

from ...logger import Logger
from ...output import Input, Output
from ...step import isStep, runner, Step
from .wrap_dataset_wrapper import WrapDatasetWrapper


@isStep('wrapper')
class WrapKFold(WrapDatasetWrapper):
    name = "Split date to train and test set"
    def __init__(self, step: Step):
        self.configurations = [{
            'folds': {
                'description': 'Split ratio',
                'default': 5,
                'no_gridsearch': True,
            },
            'stratify': {
                'description': 'Whether to run stratified K-Fold',
                'default': True,
                'no_gridsearch': True,
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

        Logger().log(f"running k-folds: [b]{self.step.__class__.__name__}[/] ({', '.join(self.step.conf_to_rich_str_list())})")
        
        outputs: list[Output] = []
        metrics = []
        for training_input in inputs:
            output: Output = self.step.run(training_input)[0]
            outputs.append(output)

            # X_test will be None when training on the whole dataset,
            # which means we can't compute metrics.
            if training_input.dataset.X_test is not None:
                metrics.append(output.evaluate())

            output.dataset = input.dataset # back to Output (from TrainingInput)
        
        output = outputs[-1]
        output.computed_metrics = { k: np.mean([ metric[k] for metric in metrics ]) for k in outputs[0].computed_metrics.keys() }

        return output
    
    def priorize(self, input=None):
        return 1

    def conf_to_rich_str_list(self):
        return [f'step={self.step.__class__.__name__}', *super().conf_to_rich_str_list()]
