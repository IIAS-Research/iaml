from ...actionable import *
from ...automed import Output
from ...data_type import DataType
import copy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

@isStep('cleaning')
class ActTfIdf(Actionable):
    name="TF-IDF"
    def __init__(self):
        self.configurations = [{}]
    
    @runner
    def run(self, input, callback=None) -> Output:
        def transform(x, y, vectorizer, column):
            transformed = vectorizer.transform(x[column].fillna(''))
            
            features_names = list(map(lambda x: "_".join([column, x]), vectorizer.get_feature_names_out()))
            ohe_df = pd.DataFrame(transformed.todense(), columns=features_names)
            
            x = pd.concat([x, ohe_df], axis=1).drop([column], axis=1)
            return x, y
            
        for column in input.dataset.get_columns_names_by_type([DataType.SHORT_TEXT, DataType.TEXT]):
            values = input.dataset.X_train[column].fillna('')
            vectorizer = TfidfVectorizer().fit(values)
            
            input.dataset.apply(transform, vectorizer=vectorizer, column=column)
    
        return input.to_output(input.dataset, None, None)
        
    
    def priorize(self, input=None):
        return 0.4