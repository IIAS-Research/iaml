from ...actionable import *
from ...data_type import DataType
from ...automed import Output

@is_step('cleaning')
class ActDropDateColumn(Actionable):
    name = "Drop date column"
    
    @runner
    def run(self, input_data: Input, callback=None) -> Output:
        self.columns_to_drop = input_data.dataset.get_columns_names_by_type(DataType.DATE)

        return input_data.add_transform(self)
    
    def transform(self, x):
        return x.drop(self.columns_to_drop, axis=1) 
    
    def priorize(self, input_data=None):
        return 0 # Last cleaning action
