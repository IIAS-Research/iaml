from ...actionable import *
from ...automed import Output
from ...data_type import DataType
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

@is_step('cleaning')
class ActTfIdf(Actionable):
    name="TF-IDF"
    def __init__(self):
        self.configurations = [{}]
    
    @runner
    def run(self, input_data: Input, callback=None) -> Output:
        self.columns = []
        for column in input_data.dataset.get_columns_names_by_type([DataType.SHORT_TEXT, DataType.TEXT]):
            values = input_data.dataset.X[column].fillna('')
            vectorizer = TfidfVectorizer().fit(values)
            self.columns.append((column, vectorizer))
        
        return input_data.add_transform(self)
    
    
    def transform(self, x):
        # Without Reset index, the join with vector_df will create NAN (index mismatch)
        x = x.reset_index(drop=True)
            
        for name, vectorizer in self.columns:
            transformed = vectorizer.transform(x[name].fillna(''))

            features_names = list(map(lambda x: "_".join([name, x]), vectorizer.get_feature_names_out()))
            vector_df = pd.DataFrame(transformed.todense(), columns=features_names)

            x = pd.concat([x, vector_df], axis=1).drop([name], axis=1)
        
        return x
        
    
    def priorize(self, input_data=None):
        return 0.4
