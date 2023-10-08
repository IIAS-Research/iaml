from ...actionable import *
from ...automed import Output
import copy

from pandas.api.types import is_string_dtype
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

@isStep('cleaning')
class ActTfIdf(Actionable):
    name="TF-IDF"
    def __init__(self):
        self.configurations = [{}]
    
    @runner
    def run(self, input, callback=None) -> Output:
        dataset = copy.deepcopy(input.dataset)
        nb_rows = len(input.dataset.X_train)
        
        def transform(x, y, vectorizer, column):
            transformed = vectorizer.transform(x[column])
            ohe_df = pd.DataFrame(transformed.todense(), columns=vectorizer.get_feature_names_out())
            x = pd.concat([x, ohe_df], axis=1).drop([column], axis=1)
            return x, y
            
        for column, values in input.dataset.X_train.items():
            if is_string_dtype(values.fillna('EMPTY')):
                vectorizer = TfidfVectorizer()
                X = vectorizer.fit(values)
                
                input.dataset.apply(transform, vectorizer=vectorizer, column=column)
        
        return input.to_output(input.dataset, None, None)
        
    
    def priorize(self, input=None):
        return 0.4