"""
[WRAPPER] Warp learning step to implement cross validation
"""
import numpy as np
from sklearn.model_selection import KFold as SKKFold, StratifiedKFold
from ...logger import Logger
from ...candidate import Candidate
from ...step import Step
from ...decorators.all import is_step, runner
from .wrap_dataset_wrapper import WrapDatasetWrapper


@is_step('wrapper')
class WrapKFold(WrapDatasetWrapper):
    """
    [WRAPPER] Warp learning step to implement cross validation
    """
    name = "K-Fold cross validation"
    description = """Performs cross-validation on the dataset, splitting into
        train and test sets {folds} times."""

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
    def run(self, candidate:Candidate, callback:callable=None) -> Candidate: # pylint: disable=unused-argument
        """
        Run wrapped step on kfold and merge results into one candidate

        Args:
            candidate (Candidate): Data to run on
            callback (callable, optional): Call after each step run. Defaults to None.

        Returns:
            Candidate: Transformed candidate
        """
        if candidate.dataset.type_of_target in ['binary', 'multiclass'] \
            and self.get_config('stratify'):
            kfold = StratifiedKFold(self.get_config('folds'))
        else:
            kfold = SKKFold(self.get_config('folds'))
        
        splitted_datasets = candidate.dataset.split(kfold.split)
        
        Logger().log(f"running k-folds: [b]{self.step.__class__.__name__}[/] \
            ({', '.join(self.step.conf_to_rich_str_list())})")
        
        candidates: list[Candidate] = []
        metrics = []
        for train_ds, test_ds in splitted_datasets:
            training_candidate = candidate.to_input(dataset=train_ds)
            candidate: Candidate = self.step.run(training_candidate, callback=callback)[0]

            if test_ds is not None:
                metrics.append(candidate.evaluate(test_ds, force=True))

            candidates.append(candidate)

        candidates.sort()
        candidate = candidates[-1]
        candidate.computed_metrics = { k: np.mean([ metric[k] or 0 for metric in metrics ]) \
            for k in candidate.computed_metrics.keys() }
        candidate.dataset = candidate.dataset # back to Candidate

        self.explanations = [
            'Computed mean metrics.',
            f"""Trained {self.get_config('folds')} models, then one last model
                on the whole dataset, and returned it as the candidate."""
        ]
        
        return super().run(candidate, callback)
    
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
