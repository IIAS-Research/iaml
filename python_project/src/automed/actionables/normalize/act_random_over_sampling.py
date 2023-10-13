from ...actionable import *
from ...automed import Output
from pandas.api.types import is_numeric_dtype
import numpy as np
from imblearn.over_sampling import RandomOverSampler

import numpy as np

@isStep('normalize')
class ActRandomOverSampling(Actionable):
    name = "Random Over Sampling"
    def __init__(self):
        self.configurations = [{}]
    
    @runner
    def run(self, input, callback=None) -> Output:
        
        def transform(x, y):
            oversampler = RandomOverSampler(sampling_strategy='minority')
            return oversampler.fit_resample(x, y)
        
        input.dataset.apply(transform, only_train=True)
        
        return input.to_output(input.dataset, None, None)
    
        
    
    def priorize(self, input=None):
        return 1