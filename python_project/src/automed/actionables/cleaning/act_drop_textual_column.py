from ...actionable import *
from ...automed import Output
from pandas.api.types import is_string_dtype


@isStep('cleaning')
class ActDropTextualColumn(Actionable):
    name = "Drop textual column"
    
    @runner
    def run(self, input, callback=None) -> Output:
        for column, values in input.dataset.train_data.items():
            if is_string_dtype(values):
                input.dataset.train_data.drop(columns=[column], inplace=True)
                input.dataset.test_data.drop(columns=[column], inplace=True)
        
        return input.to_output(input.dataset, None, None)
        
    
    def priorize(self, input=None):
        return 0 # Last cleaning action