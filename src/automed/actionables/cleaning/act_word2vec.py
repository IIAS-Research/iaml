from ...actionable import *
from ...automed import Output
from ...data_type import DataType
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import numpy as np
import string

import nltk
from nltk.tokenize import word_tokenize
import re


def transform(x, y, columns: list[tuple[str, Word2Vec]]) -> Output:
    for name, model in columns:
        transformed = x[name].fillna('').apply(lambda doc: vectorize(preprocess(doc), model))
        features_names = [f"{name}_vec_{i}" for i in range(model.vector_size)]
        vector_df = pd.DataFrame(transformed.tolist(), columns=features_names)   
        x = pd.concat([x, vector_df], axis = 1).drop([name], axis = 1)
    return x, y


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
# Data preparation and cleaning
def preprocess(text):
    # Convert the text to lowercase
    text = text.lower()
    # Remove punctuation from the text
    text = ''.join([word for word in text if word not in string.punctuation])
    # Tokenize the text into words
    tokens = word_tokenize(text)
    # Remove stopwords from the tokenized words
    tokens = [word for word in tokens if word not in stop_words]
    # Apply stemming to the remaining tokens
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)   

# Vectorize the preprocessed text data
def vectorize(sentence, model):
    ''' Convert the preprocessed text data to a vector representation using
        the Word2Vec model by calculating the average of the word vectors
        present in the sentence and returns this average vector. This gives
        a vector representation of the whole sentence.'''
    words = sentence.split()
    vecteurs = [model.wv[word] for word in words if word in model.wv]
    if len(vecteurs) > 0:
        return np.mean(vecteurs, axis = 0)
    else:
        np.zeros(model.vector_size)

@isStep('cleaning')
class ActWord2Vec(Actionable):
    name = 'Word2Vec Vectorization'
    
    def __init__(self) :
        self.configurations = [{}]
      
    @runner
    def run(self, input: Input, callback=None) -> Output:
        columns = []
        for column in input.dataset.get_columns_names_by_type([DataType.TEXT]):
            values = input.dataset.X_train[column].fillna('').apply(preprocess).apply(str.split)
            model = Word2Vec(sentences = values, vector_size=100, window = 5, min_count=1, workers = 4)
            columns.append((column, model))
        return input.transform_dataset(transform, columns)
        
    def priorize(self, input=None):
        return 0.5      
    
# from ...actionable import Actionable
# from ...dataset import Dataset
# from ...candidate import Candidate
# from ...decorators.all import is_step
# from ...automed import Output
# from ...data_type import DataType
# import pandas as pd
# from gensim.models import Word2Vec
# import numpy as np

# @is_step('cleaning')
# class ActWord2Vec(Actionable):
#     name = 'Word2Vec Vectorization'

#     def __init__(self):
#         self.configuration:dict = {}
#         self.columns:list[tuple[str, TfidfVectorizer]] = None


#     stop_words = set(stopwords.words('english'))
#     stemmer = PorterStemmer()
#     # Data preparation and cleaning
#     def preprocess(self, text):
#         # Convert the text to lowercase
#         text = text.lower()
#         # Remove punctuation from the text
#         text = ''.join([word for word in text if word not in string.punctuation])
#         # Tokenize the text into words
#         tokens = word_tokenize(text)
#         # Remove stopwords from the tokenized words
#         tokens = [word for word in tokens if word not in stop_words]
#         # Apply stemming to the remaining tokens
#         tokens = [stemmer.stem(word) for word in tokens]
#         return ' '.join(tokens)   

#     # Vectorize the preprocessed text data
#     def vectorize(self, sentence, model):
#         ''' Convert the preprocessed text data to a vector representation using
#             the Word2Vec model by calculating the average of the word vectors
#             present in the sentence and returns this average vector. This gives
#             a vector representation of the whole sentence.'''
#         words = sentence.split()
#         vecteurs = [model.wv[word] for word in words if word in model.wv]
#         if len(vecteurs) > 0:
#             return np.mean(vecteurs, axis = 0)
#         else:
#             np.zeros(model.vector_size)
    


#     def fit(self, dataset: Dataset) -> Actionable:
#         """
#         Find the columns to be vectorized and adjust the Word2Vec template.

#         Args:
#             dataset (Dataset): Data to be adjusted.

#         Returns:
#             Actionable: Transformed candidate
#         """
#         self.columns = []
#         for column in dataset.get_columns_names_by_type([DataType.TEXT]):
#             values = dataset.X_train[column].fillna('').apply(self.preprocess).apply(str.split)
#             model = Word2Vec(sentences=values, vector_size=100, window=5, min_count=1, workers=4)
#             self.columns.append((column, model))
#         return self

#     def transform(self, x: pd.DataFrame) -> pd.DataFrame:
#         """
#         Apply Word2Vec vectorization to string data.

#         Args:
#             x (pd.DataFrame): Data to be adjusted.
#         

#         Returns:
#             pd.DataFrame: Transformed data.
#         """
#         for name, model in self.columns:
#             transformed = x[name].fillna('').apply(lambda doc: self.vectorize(self.preprocess(doc), model))
#             features_names = [f"{name}_vec_{i}" for i in range(model.vector_size)]
#             vector_df = pd.DataFrame(transformed.tolist(), columns=features_names)
#             x = pd.concat([x, vector_df], axis=1).drop([name], axis=1)
#         return x

#     def priorize(self, candidate: Candidate = None) -> float:
# 
#         return 0.5
