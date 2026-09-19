"""[STEP] Vectorize textual columns with TF-IDF"""
import textwrap
from typing import Any
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step
from ...data_type import DataType


@is_step('cleaning')
class ActTfIdf(Actionable):
    """[STEP] Vectorize textual columns with TF-IDF"""

    name: str = 'TF-IDF'
    _description: str = textwrap.dedent('''\
        Process "Term Frequency / Inversed Document Frequency"
        over a list of textual columns''')
    _description_long: str = textwrap.dedent('''\
        This algorithm is used to evaluate the importance of a word inside
        it\'s corpus. A word with a lot of repetitions will
        have more importance than a word appearing once.''')
    _usage: str = 'Use when short text needs weighted term features; prefer over ActCountVectorizer for damping frequent terms. Applicable to short text columns. Avoid when you need raw counts or want to drop the column (ActCountVectorizer, ActDropCategoricalColumn).'
    refs: list[dict[str, Any]] = [
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
        self.columns: list[tuple[str, TfidfVectorizer]] = None

    def fit(self, dataset: Dataset) -> Actionable:
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

    def suitable(self, dataset: Dataset) -> bool:
        return bool(dataset.get_columns_names_by_type(DataType.SHORT_TEXT))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply TF-IDF on textual columns.

        :param pd.DataFrame x: DataFrame to transform.
        :return: Transformed dataset.
        """
        # Without Reset index, the join with vector_df will create NAN (index mismatch)
        X = X.reset_index(drop=True)

        for name, vectorizer in self.columns:
            transformed = vectorizer.transform(X[name].fillna(''))

            new_names = list(map(lambda x: "_".join([name, x]), vectorizer.get_feature_names_out())) # pylint: disable=cell-var-from-loop
            vector_df = pd.DataFrame(transformed.todense(), columns=new_names)

            X = pd.concat([X, vector_df], axis=1).drop([name], axis=1)

        return X

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.4
