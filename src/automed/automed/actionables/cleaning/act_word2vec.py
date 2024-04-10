"""
[STEP] Vectorize textual columns with Word2Vec
"""
import string
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from concurrent.futures import ProcessPoolExecutor
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step
from ...data_type import DataType
from ...logger import Logger

download('stopwords')
download('punkt')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

@is_step('cleaning')
class ActWord2Vec(Actionable):
    """
    [STEP] Vectorize textual column with Word2Vec
    """
    name = "Word2Vec"
    def __init__(self):
        self.configuration:dict = {}
        self.columns:list[tuple[str, Word2Vec]] = None
            
    def preprocess(self, text:str) -> str:
        """
        Preprocesses the input text for Word2Vec processing tasks.

        Args:
            text (str): The input text to be preprocessed.

        Returns:
            str: The preprocessed text.

        Steps:
        1. Convert the text to lowercase.
        2. Remove punctuation and special characters from the text.
        3. Tokenize the text into words.
        4. Remove stopwords from the tokenized words.
        5. Apply stemming to the remaining tokens.

        """
        text = text.lower()
        
        table = str.maketrans('', '', string.punctuation)
        text = text.translate(table)
        
        tokens = word_tokenize(text)
        tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
        
        return tokens
    
    def vectorize(self, sentence:str, model):
        """
        Convert the preprocessed text data to a vector representation using
        the Word2Vec model by calculating the aveerage of the word vectors
        present in the sentence and returns this average vector. This gives 
        a vector representation of the whole sentence.
        
        Args:
        sentence (str): The preprocessed text data as a string.
        model: The Word2Vec model used for vectorization.

        Returns:
            numpy.ndarray: The average vector representation of the input sentence.
            
        """
        words = sentence.split()
        vectors = [model.wv[word] for word in words if word in model.wv]
        if len(vectors) > 0:
            return np.mean(vectors, axis = 0)
        
        return np.zeros(model.vector_size)
            
    def fit(self, dataset:Dataset) -> Actionable:
        """
        Find columns to vectorize and fit the vectorizer
        
        Args:
            dataset (Dataset): Fit data
        
        Returns:
            Candidate: Transformed candidate
        """
        self.columns = []
        for column in dataset.get_columns_names_by_type([DataType.TEXT]):
            with ProcessPoolExecutor() as executor:
                values = list(executor.map(self.preprocess, dataset.X[column].fillna('')))
                
            vectorizer = Word2Vec(sentences = values,
                                vector_size = 100,
                                window = 5,
                                min_count = 1,
                                workers = 4)
            self.columns.append((column, vectorizer))   
        
        feature_names = {c: list(v.wv.index_to_key) for c, v in self.columns}
        self.explanations = [
            f'Encoded text column **`{c}`** into **{len(v)}** new columns.'
            for c, v in feature_names.items() if len(v) > 0
        ]
        
        return self
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Apply Word2Vec vectorization to string data.
        
        Args:
            X (pd.DataFrame): DataFrame to transform
            
        Returns:
            pd.DataFrame: Transformed dataset
        """
        X = X.reset_index(drop=True)
        for name, vectorizer in self.columns:
            transformed = X[name].fillna('').apply(
                lambda doc: self.vectorize(self.preprocess(doc), vectorizer)
            )
            features_names = [f"{name}_vec_{i}" for i in range(vectorizer.vector_size)]
            vector_df = pd.DataFrame(transformed.tolist(), columns = features_names)
            X = pd.concat([X, vector_df], axis = 1).drop([name], axis = 1)
            
        return X
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.4
    
    def suitable(self, dataset:Dataset) -> bool:
        return bool(dataset.get_columns_names_by_type([DataType.TEXT]))
