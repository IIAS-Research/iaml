from ...actionable import *
from ...automed import Output, Metric
from pandas.api.types import is_numeric_dtype
from collections import Counter
import pandas as pd
@isStep('metric')
class MetricSelection(Actionable):
    
    # configurations = [{}]
          
    @runner
    def run(self, input, callback = None) -> Output:
        y = input.dataset.y_train
        pertinent_metrics = []
        #if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        #    y = y.iloc[:,0]
        
        for metric_sub_class in Metric.__subclasses__():
            metric = metric_sub_class()
            if metric.suitable(y):
                #print(f"{metric_sub_class.__name__} is pertinent for this task and data")  
                pertinent_metrics.append(metric)   
            #else:
            #    print(f"{metric_sub_class.__name__} not pertinent for this task and data")               
    
        print('Pertinent metrics are:', pertinent_metrics)
        
        return input.add_metric(pertinent_metrics)
            
    def priorize(self,  input = None):
        return 0
    
   
                
            
   
                