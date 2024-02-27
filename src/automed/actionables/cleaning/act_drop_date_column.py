from ...actionable import *
from ...data_type import DataType
from ...automed import Output

@isStep('cleaning')
class ActDropDateColumn(Actionable):
    name = "Drop date column"
    
    @runner
    def run(self, input: Input, callback=None) -> Output:
        self.columns_to_drop = input.dataset.get_columns_names_by_type(DataType.DATE)

        return input.transform_dataset(self)
    
    def transform(self, x, y):
        return x.drop(self.columns_to_drop, axis=1), y  
    
    def priorize(self, input=None):
        return 0 # Last cleaning action
