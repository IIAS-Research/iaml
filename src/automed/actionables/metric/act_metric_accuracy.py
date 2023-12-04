from ...actionable import *
from ...automed import Output, Metric
from pandas.api.types import is_numeric_dtype

@isStep('metric')
class ActMetricAccuracy(Actionable):
    
    # configurations = [{}]
    
    @runner
    def run(self, input, callback=None) -> Output:
        return input.add_metric(Metric())
        
    
    def priorize(self, input=None):
        return 0 