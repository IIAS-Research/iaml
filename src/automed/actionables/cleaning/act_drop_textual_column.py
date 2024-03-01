from ...actionable import *
from ...data_type import DataType
from ...automed import Output


@isStep('cleaning')
class ActDropTextualColumn(Actionable):
    name = 'Drop textual columns'
    description = 'Drop textual columns.'
    
    @runner
    def run(self, input: Input, callback=None) -> Output:
        self.columns_to_drop = input.dataset.get_columns_names_by_type([DataType.TEXT, DataType.SHORT_TEXT])

        input.pipeline.add_explanation(self, [
            f'Dropped column **`{c}`**.' for c in self.columns_to_drop
        ])

        return input.transform_dataset(self)    

    def transform(self, x):
        return x.drop(self.columns_to_drop, axis=1)
    
    def priorize(self, input=None):
        return 0 # Last cleaning action
