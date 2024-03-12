"""
[STEP] Learn : AutoSkLearn
"""
import autosklearn.classification # pylint: disable=import-error
import pandas as pd
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


# @is_step('predictor', 'tabular')
@is_step('to_compare')
class ActAutoSKLearn(Predictor):
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
    
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit AutoSkLearn on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=self.get_config('running_time'),
            max_models_on_disc=5,
            memory_limit = 102400)
        
        self.model.fit(dataset.X, dataset.y)
        
        return self
    

    def predict(self, X:pd.DataFrame) -> list[float]:
        """
        Apply prediction model on DataFrame

        Args:
            X (pd.DataFrame): DataFrame use to predict

        Returns:
            list[float]: Predicted values
        """
        return self.model.predict(X)

    def suitable(self, candidate:Candidate) -> bool:
        """
        Does this step suitable for this candidate

        Args:
            candidate (Candidate): Suitable for this candidate

        Returns:
            bool: Suitable ?
        """
        return candidate.dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator', 'continuous']
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
