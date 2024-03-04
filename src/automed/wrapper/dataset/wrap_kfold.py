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
        if input.dataset.type_of_target in ['binary', 'multiclass'] and self.get_config('stratify'):
            kfold = StratifiedKFold(self.get_config('folds'))
        else:
            kfold = SKKFold(self.get_config('folds'))
        
        splitted_datasets = input.dataset.split(kfold.split)
        
        Logger().log(f"running k-folds: [b]{self.step.__class__.__name__}[/] ({', '.join(self.step.conf_to_rich_str_list())})")
        
        outputs: list[Output] = []
        metrics = []
        for train_ds, test_ds in splitted_datasets:
            training_input = input.to_input(dataset=train_ds)
            
            output: Output = self.step.run(training_input, callback=callback)[0]

            outputs.append(output)
            if test_ds is not None:
                metrics.append(output.evaluate(test_ds))

        output = outputs[-1]
        output.computed_metrics = { k: np.mean([ metric[k] for metric in metrics ]) for k, v in outputs[0].computed_metrics.items() if v is not None }
        output.dataset = input.dataset
        
        return output
    
    
    def count_steps(self):
        return 1 + self.step.count_steps()*self.get_config('folds') # 5 folds = 5*steps -> Outch!

    def conf_to_rich_str_list(self):
        return [f'step={self.step.__class__.__name__}', *super().conf_to_rich_str_list()]
