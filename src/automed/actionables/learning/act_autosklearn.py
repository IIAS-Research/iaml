"""
[STEP] Learn : AutoSkLearn
"""
import autosklearn.classification # pylint: disable=import-error
from .act_base_learning import ActBaseLearning
from ...output import Input, Output
from ...step import is_step, runner


# @is_step('learning', 'tabular')
@is_step('to_compare')
class ActAutoSKLearn(ActBaseLearning):
    """
    [STEP] Learn : AutoSkLearn
    """
    name="Learn : AutoSkLearn"
    def __init__(self):
        self.configuration:dict = {
            'running_time': {
                'description': 'In seconds. Auto-SkLearn will search the best \
                    models during this time',
                'default': 30
            }
        }
        self.model:autosklearn.classification.AutoSklearnClassifier = None
    
    @runner
    def run(self, input_data: Input, callback:callable=None) -> Output: # pylint: disable=unused-argument
        """
        Fit AutoSkLearn on Input.dataset

        Args:
            input_data (Input): Fit data
            callback (callable, optional): Call after each step. Defaults to None.

        Returns:
            Output: Transformed input
        """
        self.model = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=self.get_config('running_time'),
            max_models_on_disc=5,
            memory_limit = 102400)
        
        self.model.fit(input_data.dataset.X, input_data.dataset.y)
        
        return input_data.set_model(self)

    def suitable(self, input_data:Input) -> bool:
        return input_data.dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator', 'continuous']
