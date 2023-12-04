from ...actionable import *
from ...data_type import DataType
from ...automed import Output

@isStep('cleaning')
class ActDropDateColumn(Actionable):
    name = "Drop date column"
    
    @runner
    def run(self, input: Input, callback=None) -> Output:    
        
        def transform(x, y, columns: list) -> Output:
            return x.drop(columns, axis=1), y  
        
        columns_to_drop = input.dataset.get_columns_names_by_type(DataType.DATE)

        return input.transform_dataset(transform, columns_to_drop)
        
    
    def priorize(self, input=None):
        return 0 # Last cleaning action
