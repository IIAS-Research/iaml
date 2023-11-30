from ...actionable import *
from ...data_type import DataType
from ...automed import Output


def transform(input: Input, columns: list) -> Output:
    input.dataset.apply(lambda x, y: (x.drop(columns, axis=1), y))

    return input.to_output()


@isStep('cleaning')
class ActDropDateColumn(Actionable):
    name = "Drop date column"
    
    @runner
    def run(self, input: Input, callback=None) -> Output:        
        columns_to_drop = input.dataset.get_columns_names_by_type(DataType.DATE)
        transform(input, columns_to_drop)
        
        return input.to_output(None, None, transform, columns_to_drop)
        
    
    def priorize(self, input=None):
        return 0 # Last cleaning action
