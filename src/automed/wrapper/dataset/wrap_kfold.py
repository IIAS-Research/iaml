"""
[WRAPPER] Warp learning step to implement cross validation
"""
import numpy as np
from sklearn.model_selection import KFold as SKKFold, StratifiedKFold
from ...logger import Logger
from ...output import Input, Output
from ...step import is_step, runner, Step
from .wrap_dataset_wrapper import WrapDatasetWrapper


@is_step('wrapper')
class WrapKFold(WrapDatasetWrapper):
    """
    [WRAPPER] Warp learning step to implement cross validation
    """
    name = "Split date to train and test set"
    def __init__(self, step: Step):
        self.configuration = {
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
        }
        
    @runner
    def run(self, input_data:Input, callback:callable=None) -> Output: # pylint: disable=unused-argument
        """
        Run wrapped step on kfold and merge results into one output

        Args:
            input_data (Input): Data to run on
            callback (callable, optional): Call after each step run. Defaults to None.

        Returns:
            Output: Transformed input
        """
        if input_data.dataset.type_of_target in ['binary', 'multiclass'] \
            and self.get_config('stratify'):
            kfold = StratifiedKFold(self.get_config('folds'))
        else:
            kfold = SKKFold(self.get_config('folds'))
        
        splitted_datasets = input_data.dataset.split(kfold.split)
        
        Logger().log(f"running k-folds: [b]{self.step.__class__.__name__}[/] \
            ({', '.join(self.step.conf_to_rich_str_list())})")
        
        outputs: list[Output] = []
        metrics = []
        for train_ds, test_ds in splitted_datasets:
            training_input = input_data.to_input(dataset=train_ds)
            output: Output = self.step.run(training_input, callback=callback)[0]

            if test_ds is not None:
                metrics.append(output.evaluate(test_ds, force=True))

            outputs.append(output)
        outputs.sort()
        output = outputs[-1]
        output.computed_metrics = { k: np.mean([ metric[k] or 0 for metric in metrics ]) \
            for k in outputs[0].computed_metrics.keys() }
        output.dataset = input_data.dataset # back to Output
        
        return output
    
    def count_steps(self) -> int:
        """
        Estimated count of remaining steps

        Returns:
            int: Step count
        """
        return 1 + self.step.count_steps()*self.get_config('folds') # 5 folds = 5*steps -> Outch!

    def conf_to_rich_str_list(self) -> list[str]:
        """
        Format Step for rich logger

        Returns:
            list[str]: Rich formatted strings
        """
        return [f'step={self.step.__class__.__name__}', *super().conf_to_rich_str_list()]
