from ...actionable import *
from ...output import TrainingInput

from sklearn.ensemble import GradientBoostingClassifier
from skmultilearn.problem_transform import BinaryRelevance


def learn(model, X):
    return model.predict(X)


@isStep('learning', 'tabular')
@assessable
class ActXGBoost(Actionable):
    name = "Learn : XGBoost"
    def __init__(self):
        self.configurations = [{
            'max_depth': {
                'description': 'Max depth of each tree',
                'default': 15,
                'range': [1, float('inf')]
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            },
            'learning_rate': {
                'description': 'Learning rate',
                'default': 1.0,
                'range': [0.000000001, float('inf')]
            },
            'n_estimators': {
                'description': 'Number of estimators',
                'default': 100,
                'range': [1, float("inf")]
            }
        }]
        
    @runner
    def run(self, input: TrainingInput, callback=None):
        model = GradientBoostingClassifier(
                                        n_estimators=self.get_config('n_estimators'),
                                        learning_rate=self.get_config('learning_rate'),
                                        max_depth=self.get_config('max_depth'),
                                        random_state=self.get_config('random_state')
                                        )
        
        
        if input.dataset.is_multilabel:
            model = BinaryRelevance(classifier=model, require_dense=[False, True])
            
            
        model.fit(input.dataset.X_train, input.dataset.Y_train)
        
        return input.set_model(model, learn)
    

        
    
    def priorize(self, input=None):
        return 0.5 # neutral