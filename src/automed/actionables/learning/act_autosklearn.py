import autosklearn.classification

from ...actionable import *
from ...output import Input


# @isStep('learning', 'tabular')
@isStep('to_compare')
@assessable
class ActAutoSKLearn(Actionable):
    name="Learn : AutoSkLearn"
    def __init__(self):
        self.configurations = [{
            'running_time': {
                'description': 'In seconds. Auto-SkLearn will search the best models during this time',
                'default': 30
            }
        }]
    
    @runner
    def run(self, input: Input, callback=None):
        self.model = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=self.get_config('running_time'),
            max_models_on_disc=5,
            memory_limit = 102400)
        
        self.model.fit(input.dataset.X, input.dataset.y)
        
        return input.set_model(self)
    

    def predict(self, X):
        return self.model.predict(X)

    def suitable(self, input) -> bool:
        return input.dataset.type_of_target in ['binary', 'multiclass',  'multilabel-indicator', 'continuous']
    
    def priorize(self, input=None):
        return 0.5 # neutral