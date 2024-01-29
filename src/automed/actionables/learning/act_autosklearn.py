from ...actionable import *
from ...automed import Output, Metric
import autosklearn.classification


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
    def run(self, input:Output, callback=None):
        dataset = input.dataset
        metric = input.metric or Metric()
        model = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=self.get_config('running_time'),
            max_models_on_disc=5,
            memory_limit = 102400)
        
        model.fit(dataset.X_train, dataset.y_train)
        
        return input.set_model(model, learn)

        
    
    def priorize(self, input=None):
        return 0.5 # neutral