import autosklearn.classification

from ...actionable import *
from ...output import TrainingInput


def learn(model, X):
    return model.predict(X)


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
    def run(self, input: TrainingInput, callback=None):
        model = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=self.get_config('running_time'),
            max_models_on_disc=5,
            memory_limit = 102400)
        
        model.fit(input.dataset.X_train, input.dataset.Y_train)
        
        return input.set_model(model, learn)

        
    
    def priorize(self, input=None):
        return 0.5 # neutral