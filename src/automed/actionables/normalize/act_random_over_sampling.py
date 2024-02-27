from ...actionable import *
from ...automed import Output
from imblearn.over_sampling import RandomOverSampler


def transform(x, y):
    return RandomOverSampler(sampling_strategy='minority').fit_resample(x, y)


@isStep('normalize')
class ActRandomOverSampling(Actionable):
    name = "Random Over Sampling"
    def __init__(self):
        self.configurations = [{}]
    
    @runner
    def run(self, input: Input, callback=None) -> Output:
        return input.transform_dataset(transform, only_train=True)

    
    def priorize(self, input=None):
        return 1
    
    def suitable(self, input) -> bool:
        return input.dataset.type_of_target in ['binary', 'multiclass']

