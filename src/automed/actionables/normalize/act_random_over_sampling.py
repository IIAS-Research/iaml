from ...actionable import *
from ...automed import Output
from imblearn.over_sampling import RandomOverSampler


@isStep('normalize')
class ActRandomOverSampling(Actionable):
    name = "Random Over Sampling"
    def __init__(self):
        self.configurations = [{}]
    
    @runner
    def run(self, input: Input, callback=None) -> Output:
        return input.transform_dataset(self, before_train=True)
    
    def transform(self, x, y):
        return RandomOverSampler(sampling_strategy='minority').fit_resample(x, y)
    
    def priorize(self, input=None):
        return 1
    
    def suitable(self, input: Input) -> bool:
        return input.dataset.type_of_target in ['binary', 'multiclass']

