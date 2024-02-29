from ...actionable import *
from ...data_type import DataType
from ...automed import Output
from sklearn.utils.multiclass import type_of_target

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# # For monolabel
def transform(x, y) -> Output:
    if y.dtype == 'object' or y.dtype == 'bool':
       y = y.astype('category').cat.codes
       
    return x, y

@isStep('cleaning')
class ActCategoryStringToNumeric(Actionable):
    name="Encode categorical strings into numeric values."
    def __init__(self):
        self.configurations = [{}]

    #For monolabel
    @runner 
    def run(self, input: Input, callback= None) -> Output:
        if input.dataset.y_train.isnull().any():
            input.dataset.y_train.fillna('', inplace=True)
        return input.transform_dataset(transform)
    
    def priorize(self, input=None):
        return 0.5