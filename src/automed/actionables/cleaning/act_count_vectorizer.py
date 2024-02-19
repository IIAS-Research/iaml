from ...actionable import *
from ...automed import Output
from ...data_type import DataType
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def transform(x, y, columns: list[tuple[str, CountVectorizer]]) -> Output:
    for name, vectorizer in columns:
        transformed = vectorizer.transform(x[name].fillna(''))
        
        features_names = list(map(lambda x:  "_".join([name, x]), vectorizer.get_feature_names_out()))
        #print('features_names for :', name, "are: ", features_names)
        ohe_df = pd.DataFrame(transformed.todense(), columns=features_names)
        #print(ohe_df)
        x = pd.concat([x, ohe_df], axis=1).drop([name], axis=1)
    
    return x, y


def preprocess(text):
    text = str(text) if not pd.isnull(text) else ''
    text = text.lower() # Convert to lower case
    text = re.sub('[^\w\s]', '', text) # Remove punctuation
    tokens = word_tokenize(text) #Tokenization
    stemmed_tokens = [stemmer.stem(token) for token in tokens] # Stemming allows words to be reduced to their root, which can help improve classification performance
    return ' '.join(stemmed_tokens) # Return cleaned and stemmed text
    
    
@isStep('cleaning')
class ActCountVectorier(Actionable):
    name = 'CountVectorizer'
    def __init__(self):
        self.configurations = [{}]
        
    @runner
    def run(self, input: Input, callback = None) -> Output:
        columns = []
        #if input.dataset.is_multilabel:
        for column in input.dataset.get_columns_names_by_type([DataType.TEXT]):
            values = input.dataset.X_train[column].apply(preprocess)
            vectorizer = CountVectorizer().fit(values)
            columns.append((column, vectorizer))
        #print(columns)    
        
        return input.transform_dataset(transform, columns)
    
    def priorize(self, input=None):
        return 0.4
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        