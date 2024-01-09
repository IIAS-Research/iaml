from ...actionable import *
from ...automed import Output
from ...data_type import DataType
from sklearn.preprocessing import MinMaxScaler

        
def transform(x, y, scaler, columns):
    x[columns] = scaler.transform(x[columns])
    return x, y


@isStep('normalize')
class ActMinMaxScaler(Actionable):
    name = "Min Max Scaler"
    def __init__(self):
        self.configurations = [{}]
    
    @runner
    def run(self, input: Input, callback=None) -> Output:
        columns = input.dataset.get_columns_names_by_type(DataType.NUMERIC)
        values = input.dataset.X_train[columns]
        scaler = MinMaxScaler()
        scaler.fit(values)

        return input.transform_dataset(transform, scaler, columns)
        
    
    def priorize(self, input=None):
        return 0.5 # TODO -> Do something better. This function have no sense for now. Only an example.
