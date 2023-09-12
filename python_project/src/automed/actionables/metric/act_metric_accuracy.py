from ...actionable import *
from ...automed import Output
from pandas.api.types import is_numeric_dtype

@isStep('metric')
class ActMetricAccuracy(Actionable):
    
    configuration = {}
    
    @runner
    def run(self, input, callback=None) -> Output:
        return input.to_output(input.dataset, None, None)
        
    
    def priorize(self, input=None):
        return 0 