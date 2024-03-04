"""
[STEP] Learn : AutoSkLearn
"""
import autosklearn.classification
import pandas as pd
from ...actionable import Actionable
from ...output import Input, Output
from ...step import is_step, runner


# @is_step('learning', 'tabular')
@is_step('to_compare')
class ActAutoSKLearn(Actionable):
    """
    [STEP] Learn : AutoSkLearn
    """
    name="Learn : AutoSkLearn"
    def __init__(self):
        self.configurations:list[dict] = [{
            'running_time': {
                'description': 'In seconds. Auto-SkLearn will search the best \
                    models during this time',
                'default': 30
            }
        }]
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
    

    def predict(self, X:pd.DataFrame) -> list[float]:
        """
        Apply prediction model on DataFrame

        Args:
            X (pd.DataFrame): DataFrame use to predict

        Returns:
            list[float]: Predicted values
        """
        return self.model.predict(X)

    def suitable(self, input_data:Input) -> bool:
        """
        Does this step suitable for this input

        Args:
            input_data (Input): Suitable for this input

        Returns:
            bool: Suitable ?
        """
        return input.dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator', 'continuous']
    
    def priorize(self, input_data:Input=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
