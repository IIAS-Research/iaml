from ...actionable import *
from ...automed import Output
from imblearn.over_sampling import RandomOverSampler


@is_step('normalize')
class ActRandomOverSampling(Actionable):
    name = "Random Over Sampling"
    def __init__(self):
        self.configurations = [{}]
    
    @runner
    def run(self, input_data: Input, callback=None) -> Output:
        return input_data.resample(self)
    
    def resample(self, x, y):
        return RandomOverSampler(sampling_strategy='minority').fit_resample(x, y)
    
    def priorize(self, input_data=None):
        return 1
    
    def suitable(self, input_data: Input) -> bool:
        return input_data.dataset.type_of_target in ['binary', 'multiclass']

