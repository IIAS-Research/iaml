from actionable import *

@isStep('learning', 'tabular')
@assessable
class ActAutoSkLearn(Actionable):
    import autosklearn.classification
    
    configuration = {
        'running_time': {
            'description': 'In seconds. Auto-SkLearn will search the best models during this time',
            'default': 60
        }
    }
    
    @runner
    def run(self, dataset):
        cls = autosklearn.classification.AutoSklearnClassifier()
        cls.fit(X_train, y_train)
        predictions = cls.predict(X_test)
        
    
    def priorize(self, dataset=None):
        return 0.5 # neutral