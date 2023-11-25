from ...actionable import *
from ...automed import Output
from ...data_type import DataType
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import numpy as np

@isStep('normalize')
class ActMinMaxScaler(Actionable):
    name = "Min Max Scaler"
    def __init__(self):
        self.configurations = [{}]
    
    @runner
    def run(self, input, callback=None) -> Output:
        
        def transform(x, y, scaler, column):
            x[column] = scaler.transform(np.array(x[column]).reshape(-1, 1))
            return x, y
        
        for column in input.dataset.get_columns_names_by_type(DataType.NUMERIC):
            values = input.dataset.X_train[column]
            scaler = MinMaxScaler()
            scaler.fit(np.array(values).reshape(-1, 1))
    
            # Transform column
            input.dataset.apply(transform, scaler=scaler, column=column)
        
        return input.to_output(input.dataset, None, None)
    
        
    
    def priorize(self, input=None):
        return 0.5 # TODO -> Do something better. This function have no sense for now. Only an example.