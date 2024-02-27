from ...actionable import *
from ...automed import Output, Metric
from ...metrics import *
from pandas.api.types import is_numeric_dtype
from collections import Counter
import pandas as pd
@isStep('metric')
class MetricSelection(Actionable):
    
    # configurations = [{}]
    
    @runner
    def run(self, input, callback = None) -> Output:
        # Iterate through all subclasses that inherit from Metric
        for metric_sub_class in Metric.__subclasses__():
            # Instantiate a subclass
            metric = metric_sub_class()
            # Verify if a subclass is suitable or not
            if metric.suitable(input.dataset):
                input.add_metric(metric)
        return input.to_output()
            
    def priorize(self,  input = None):
        return 1