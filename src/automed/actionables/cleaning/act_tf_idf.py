from ...actionable import *
from ...automed import Output
from ...data_type import DataType
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def transform(input: Input, columns: list[tuple[str, TfidfVectorizer]]) -> Output:
    for name, vectorizer in columns:
        transformed = vectorizer.transform(input.dataset.pred_data[name].fillna(''))

        features_names = list(map(lambda x: "_".join([name, x]), vectorizer.get_feature_names_out()))
        ohe_df = pd.DataFrame(transformed.todense(), columns=features_names)

        input.dataset.apply(lambda x, y: (pd.concat([x, ohe_df], axis=1).drop([name], axis=1), y))
    
    return input.to_output()


@isStep('cleaning')
class ActTfIdf(Actionable):
    name="TF-IDF"
    def __init__(self):
        self.configurations = [{}]
    
    @runner
    def run(self, input: Input, callback=None) -> Output:
        def _transform(x, y, vectorizer, column):
            transformed = vectorizer.transform(x[column].fillna(''))
            
            features_names = list(map(lambda x: "_".join([column, x]), vectorizer.get_feature_names_out()))
            ohe_df = pd.DataFrame(transformed.todense(), columns=features_names)
            
            x = pd.concat([x, ohe_df], axis=1).drop([column], axis=1)
            return x, y
        
        columns = []
        for column in input.dataset.get_columns_names_by_type([DataType.SHORT_TEXT, DataType.TEXT]):
            values = input.dataset.X_train[column].fillna('')
            vectorizer = TfidfVectorizer().fit(values)
            columns.append((column, vectorizer))
            
            input.dataset.apply(_transform, vectorizer=vectorizer, column=column)
    
        return input.to_output(input.dataset, None, transform, columns)
        
    
    def priorize(self, input=None):
        return 0.4