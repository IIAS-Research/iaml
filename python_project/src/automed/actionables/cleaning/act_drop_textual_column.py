from ...actionable import *
from ...automed import Output
from pandas.api.types import is_string_dtype


@isStep('cleaning')
class ActDropTextualColumn(Actionable):
    name = "Drop textual column"
    
    @runner
    def run(self, input, callback=None) -> Output:
        
        def transform(x, y, column):
            x = x.drop(columns=[column])
            return x, y
            
        for column, values in input.dataset.train_data.items():
            if is_string_dtype(values):
                input.dataset.apply(transform, column=column)
        
        return input.to_output(input.dataset, None, None)
        
    
    def priorize(self, input=None):
        return 0 # Last cleaning action