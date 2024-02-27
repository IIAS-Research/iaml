from ...actionable import *
from ...automed import Output
from ...data_type import DataType
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

        
def transform(x, y, columns: list[tuple[str, TfidfVectorizer]]) -> Output:
    # Without Reset index, the join with vector_df will create NAN (index mismatch)
    x = x.reset_index(drop=True)
    if isinstance(y, pd.Series):
        y = y.reset_index(drop=True)
        
    for name, vectorizer in columns:
        transformed = vectorizer.transform(x[name].fillna(''))

        features_names = list(map(lambda x: "_".join([name, x]), vectorizer.get_feature_names_out()))
        vector_df = pd.DataFrame(transformed.todense(), columns=features_names)

        x = pd.concat([x, vector_df], axis=1).drop([name], axis=1)
    
    return x, y


@isStep('cleaning')
class ActTfIdf(Actionable):
    name="TF-IDF"
    def __init__(self):
        self.configurations = [{}]
    
    @runner
    def run(self, input: Input, callback=None) -> Output:
        columns = []
        for column in input.dataset.get_columns_names_by_type([DataType.SHORT_TEXT, DataType.TEXT]):
            values = input.dataset.X_train[column].fillna('')
            vectorizer = TfidfVectorizer().fit(values)
            columns.append((column, vectorizer))
        
        return input.transform_dataset(transform, columns)
        
    
    def priorize(self, input=None):
        return 0.4
