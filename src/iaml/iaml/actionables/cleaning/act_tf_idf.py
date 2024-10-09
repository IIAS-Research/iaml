"""
[STEP] Vectorize textual columns with TF-IDF
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step
from ...data_type import DataType
import textwrap


@is_step('cleaning')
class ActTfIdf(Actionable):
    """
    [STEP] Vectorize textual columns with TF-IDF
    """
    name= 'TF-IDF'
    description = textwrap.dedent('''\
        Process "Term Frequency / Inversed Document Frequency"
        over a list of textual columns''')
    description_long = textwrap.dedent('''\
        This algorithm is used to evaluate the importance of a word inside
        it\'s corpus. A word with a lot of repetitions will
        have more importance than a word appearing once.''')
    refs = [
        {
            'year': 1972,
            'name': 'A STATISTICAL INTERPRETATION OF TERM SPECIFICITY AND ITS APPLICATION \
                IN RETRIEVAL',
            'authors': [
                'Karen Sparck Jones'    
            ],
            'doi': 'https://doi.org/10.1108/eb026526',
            'publisher': 'Journal of Documentation Vol.21, No.1, page 11--21'
        }
    ]
    def __init__(self):
        self.configuration:dict = {}
        self.columns:list[tuple[str, TfidfVectorizer]] = None
    
    def fit(self, dataset:Dataset) -> Actionable:
        """
        Find columns to vectorize and fit vectorizer

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        
        self.columns = []
        for column in dataset.get_columns_names_by_type([DataType.SHORT_TEXT]):
            values = dataset.X[column].fillna('')
            vectorizer = TfidfVectorizer().fit(values)
            self.columns.append((column, vectorizer))

        feature_names = { c: v.get_feature_names_out() for c, v in self.columns }
        self.explanations = [
            f'Encoded text column **`{c}`** into **{len(v)}** new columns.'
            for c, v in feature_names.items() if len(v) > 0
        ]
        
        return self
    
    def suitable(self, dataset:Dataset) -> bool:
        return bool(dataset.get_columns_names_by_type(DataType.SHORT_TEXT))    
    
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Apply TD-IDF on textual column

        Args:
            x (pd.DataFrame): DataFrame to transform

        Returns:
            pd.DataFrame: Transformed dataset
        """
        # Without Reset index, the join with vector_df will create NAN (index mismatch)
        X = X.reset_index(drop=True)
            
        for name, vectorizer in self.columns:
            transformed = vectorizer.transform(X[name].fillna(''))

            new_names = list(map(lambda x: "_".join([name, x]), vectorizer.get_feature_names_out())) # pylint: disable=cell-var-from-loop
            vector_df = pd.DataFrame(transformed.todense(), columns=new_names)

            X = pd.concat([X, vector_df], axis=1).drop([name], axis=1)
        
        return X
        
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.4
