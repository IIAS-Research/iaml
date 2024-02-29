from ...actionable import *
from ...automed import Output
from ...data_type import DataType
from sklearn.preprocessing import MinMaxScaler


@isStep('normalize')
class ActMinMaxScaler(Actionable):
    name = "Min Max Scaler"
    def __init__(self):
        self.configurations = [{}]
    
    @runner
    def run(self, input: Input, callback=None) -> Output:
        self.columns = input.dataset.get_columns_names_by_type(DataType.NUMERIC)
        values = input.dataset[self.columns]
        self.scaler = MinMaxScaler()
        self.scaler.fit(values)

        return input.transform_dataset(self)
        
    def transform(self, x):
        x[self.columns] = self.scaler.transform(x[self.columns])
        return x

    
    def priorize(self, input=None):
        return 0.5 # TODO -> Do something better. This function have no sense for now. Only an example.
