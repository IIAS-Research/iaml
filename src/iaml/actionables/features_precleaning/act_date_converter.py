"""
[STEP] Convert Short text to date if possible
"""
import textwrap
import pandas as pd
from ...actionable import Actionable
from ...dataset import Dataset
from ...data_type import DataType
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('features_precleaning')
class ActDateConverter(Actionable):
    """
    [STEP] Convert Short text to date if possible
    """
    name = 'Text to Date Converter'
    _description = 'Convert Text to Date if possible'
    _description_long = textwrap.dedent('''\
        Try to convert all text of a column to date.
        If more than {authorized_error_ratios}% of the rows return errors, then the
        column is not converted. As converting to date is time consuming, we will
        perform the test on {sample_size}.
        ''')
    
    def __init__(self):
        self.configuration:dict = {
            'authorized_error_ratios': {
                'default': 0.05,
                'description': 'Over this ratios, the column will not be converted into date.'
            },
            'sample_size': {
                'default': 200,
                'description': textwrap.dedent('''\
                    Convert date is time consuming. To save time, date
                    detection will be done on a random sample.
                    Set to -1 to detect on the whole dataset.''')
                }
            }
        self.columns:list[str] = None


    def fit(self, dataset:Dataset) -> Actionable:
        """
        Find columns to convert

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.columns = []
        if self.get_config('sample_size') > 0:
            sample = dataset.X.sample(n=min(self.get_config('sample_size'), len(dataset.X)))
        else:
            sample = dataset.X
        
        for column in dataset.get_columns_names_by_type(DataType.SHORT_TEXT):
            threshold_count:float = sample[column].count() * \
                (1 - self.get_config('authorized_error_ratios'))
            new_columns:pd.DataFrame = sample[column].apply(self.__string_value_to_date)
            
            if new_columns.count() >= threshold_count:
                self.columns.append(column)
            
        self.explanations = [
            f'Convert text column **`{c}`** into datetime column.'
            for c in self.columns
        ]

        return self
    
    def suitable(self, dataset:Dataset) -> bool:
        return bool(dataset.get_columns_names_by_type(DataType.SHORT_TEXT))    
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Apply Text to date convert

        Args:
            x (pd.DataFrame): DataFrame to transform

        Returns:
            pd.DataFrame: Transformed dataset
        """
        for column in self.columns:
            X[column] = pd.to_datetime(X[column], errors='coerce')

        return X
        
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5
            
    def __string_value_to_date(self, value:str, \
                            date_formats = None) -> pd.Timestamp:
        """
        Convert a string value (with date format) to a date
        

        Args:
            value (str): String in a date format

        Returns:
            pd.datetime: converted date
        """
        if date_formats is None:
            date_formats = [None]
            # date_formats = ['%Y-%M-%d', '%d-%M-%Y', '%Y/%M/%d', '%d/%M/%Y', None]
        elif isinstance(date_formats, list):
            date_formats = list(date_formats)
        
        for date_format in date_formats:
            try:
                return pd.to_datetime(value, format=date_format)
            except ValueError:
                pass # Let's try the next date format

        return pd.NaT
