from ...actionable import *
from ...automed import Output
from pandas.api.types import is_datetime64_any_dtype as is_datetime


@isStep('cleaning')
class ActDropDate(Actionable):
    name = "Drop date column"
    
    @runner
    def run(self, input, callback=None) -> Output:
        
        def transform(x, y, column):
            x = x.drop(columns=[column])
            return x, y

        print("DDDRRRRROOOOPPPPPPPP¨")            
        print("DDDRRRRROOOOPPPPPPPP¨")            
        print("DDDRRRRROOOOPPPPPPPP¨")            
        print("DDDRRRRROOOOPPPPPPPP¨")            
        for column, values in input.dataset.train_data.items():
            print(column, is_datetime(values))
            if is_datetime(values):
                input.dataset.apply(transform, column=column)
        
        return input.to_output(input.dataset, None, None)
        
    
    def priorize(self, input=None):
        return 0 # Last cleaning action