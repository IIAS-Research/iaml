from ...actionable import *
from ...data_type import DataType
from ...automed import Output
from pandas.api.types import is_datetime64_any_dtype as is_datetime


@isStep('cleaning')
class ActDropDateColumn(Actionable):
    name = "Drop date column"
    
    @runner
    def run(self, input, callback=None) -> Output:
        
        def transform(x, y, column):
            x = x.drop(columns=[column])
            return x, y
        
        for column in input.dataset.get_columns_names_by_type(DataType.DATE):
            input.dataset.apply(transform, column=column)
        
        return input.to_output(input.dataset, None, None)
        
    
    def priorize(self, input=None):
        return 0 # Last cleaning action