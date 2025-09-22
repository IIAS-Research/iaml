"""[STEP] Vectorize textual columns with Word2Vec"""
from typing import Any
import textwrap
import string
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import StemmerI, PorterStemmer
from nltk.tokenize import word_tokenize
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step
from ...data_type import DataType


download('stopwords')
download('punkt')


@is_step('cleaning')
class ActWord2Vec(Actionable):
    """[STEP] Vectorize textual column with Word2Vec"""

    name: str = "Word2Vec"
    _description: str = "Process Word2Vec algorithm over a list of columns"
    _description_long: str = textwrap.dedent('''\
        Word2Vec is a word embedding algorithm auto-supervised algorithm.
        This means we don't need labelled data as the algorithm discove
        the ground truth by himself''')
    refs: list[dict[str, Any]] = [
        {
            'year': 2023,
            'name': 'Efficient Estimation of Word Representations in Vector Space',
            'authors': [
                'Thomas Mikolov',
                'Kai Chen',
                'Greg Corrado',
                'Jeffrey Dean'    
            ],
            'doi': 'https://doi.org/10.48550/arXiv.1301.3781',
            'publisher': 'International Conference on Learning Representations'
        }
    ]

    def __init__(self):
        self.columns: list[tuple[str, Word2Vec]] = None
        self.stop_words: set[str] = set(stopwords.words('english'))
        self.stemmer: StemmerI = PorterStemmer()

    @staticmethod
    def vectorize(words: str, model: Word2Vec):
        """Convert the preprocessed text data to a vector representation using
        the Word2Vec model by calculating the aveerage of the word vectors
        present in the sentence and returns this average vector. This gives 
        a vector representation of the whole sentence.

        :param str words: The preprocessed text data as a string.
        :param Word2Vec model: The Word2Vec model used for vectorization.
        :return: The average vector representation of the input sentence.
        """
        vectors = [ model.wv[word] for word in words if word in model.wv ]
        if len(vectors) > 0:
            return np.mean(vectors, axis=0)

        return np.zeros(model.vector_size)

    def __preprocess(self, text: str) -> list[str]:
        """Preprocesses the input text for Word2Vec processing tasks.

        Steps:
        1. Convert the text to lowercase.
        2. Remove punctuation and special characters from the text.
        3. Tokenize the text into words.
        4. Remove stopwords from the tokenized words.
        5. Apply stemming to the remaining tokens.

        :param str text: The input text to be preprocessed.
        :return: The preprocessed text.
        """
        text = text.lower()

        table = str.maketrans('', '', string.punctuation)
        text = text.translate(table)

        tokens = word_tokenize(text)
        tokens = [ self.stemmer.stem(word) for word in tokens if word not in self.stop_words ]

        return tokens

    def fit(self, dataset: Dataset) -> Actionable:
        """Find columns to vectorize and fit the vectorizer.

        :param Dataset dataset: Data.
        :return: Transformed candidate.
        """
        self.columns = []
        for column in dataset.get_columns_names_by_type([DataType.TEXT]):
            values = dataset.X[column].fillna('').apply(self.__preprocess)
            vectorizer = Word2Vec(sentences = values,
                                vector_size = 100,
                                window = 5,
                                min_count = 1,
                                workers = 4)
            self.columns.append((column, vectorizer))

        feature_names = { c: list(v.wv.index_to_key) for c, v in self.columns }
        self.explanations = [
            f'Encoded text column **`{c}`** into **{len(v)}** new columns.'
            for c, v in feature_names.items() if len(v) > 0
        ]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply Word2Vec vectorization to string data.

        :param pd.DataFrame X: DataFrame to transform.
        :return: Transformed dataset.
        """
        X = X.reset_index(drop=True)
        for name, vectorizer in self.columns:
            transformed = X[name].fillna('').apply(
                lambda doc: self.vectorize(self.__preprocess(doc), vectorizer) # pylint: disable=cell-var-from-loop
            )
            features_names = [f"{name}_vec_{i}" for i in range(vectorizer.vector_size)]
            vector_df = pd.DataFrame(transformed.tolist(), columns = features_names)
            X = pd.concat([X, vector_df], axis = 1).drop([name], axis = 1)

        return X

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.4

    def suitable(self, dataset: Dataset) -> bool:
        return bool(dataset.get_columns_names_by_type([DataType.TEXT]))
