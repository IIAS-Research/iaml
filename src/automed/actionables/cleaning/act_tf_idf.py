"""
[STEP] Vectorize textual columns with TF-IDF
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from ...actionable import Actionable
from ...output import Output, Input
from ...step import is_step, runner
from ...data_type import DataType

@is_step('cleaning')
class ActTfIdf(Actionable):
    """
    [STEP] Vectorize textual columns with TF-IDF
    """
    name="TF-IDF"
    def __init__(self):
        self.configurations:list[dict] = [{}]
        self.columns:list[tuple[str, TfidfVectorizer]] = None
    
    @runner
    def run(self, input_data: Input, callback:callable=None) -> Output: # pylint: disable=unused-argument
        """
        Find columns to vectorize and fit vectorizer

        Args:
            input_data (Input): Fit data
            callback (callable, optional): Call after each step. Defaults to None.

        Returns:
            Output: Transformed input
        """
        self.columns = []
        for column in input_data.dataset.get_columns_names_by_type(
            [DataType.SHORT_TEXT, DataType.TEXT]
            ):
            values = input_data.dataset.X[column].fillna('')
            vectorizer = TfidfVectorizer().fit(values)
            self.columns.append((column, vectorizer))
        
        return input_data.add_transform(self)
    
    
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
        
    
    def priorize(self, input_data:Input=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.4
