"""
Encapsulate X, y data to be used by Steps
Add features like data type detection and splitting 
"""
import copy
from typing import Iterator, TYPE_CHECKING
from sklearn.utils.multiclass import type_of_target
import numpy as np
import pandas as pd

from .data_type import DataType

if TYPE_CHECKING:
    from .auto_pipeline import AutoPipeline

class Dataset:
    """
    Encapsulate X, y data to be used by Steps
    Add features like data type detection and splitting
    """
    
    def __init__(self, X:pd.DataFrame, y:list=None):
        self.__X:pd.DataFrame = X
        self.__y:np.array = np.array(y)
        
        self.columns_types:list[DataType] = self.__detect_columns_types()
        self.type_of_target:str = type_of_target(self.__y)
    
    @property
    def features(self) -> list[str]:
        """
        List columns names of X data

        Returns:
            list[str]: columns names
        """
        return self.X.columns.to_list()
    
    @property
    def X(self) -> pd.DataFrame:
        """
        X data getter

        Returns:
            pd.DataFrame: X data
        """
        return self.__X

    @property
    def y(self) -> pd.DataFrame:
        """
        y data getter

        Returns:
            pd.DataFrame: y data
        """
        return self.__y

    def copy(self, deep:bool=True) -> 'Dataset':
        """
        Copy Dataset into a new instance

        Args:
            deep (bool, optional): Also copy sub-objects. Defaults to True.

        Returns:
            Dataset: Copied Dataset
        """
        if deep:
            return copy.deepcopy(self)
        return copy.copy(self)

    def transform(self, method:callable) -> None:
        """
        Apply transform method to X data

        Args:
            method (callable): Callable to apply. Will be call with X as parameter 
        """
        self.__X = method(self.X)
        self.columns_types = self.__detect_columns_types()
        
    def split(self, splitter: callable) -> Iterator[tuple['Dataset', 'Dataset']]: 
        """
        Use splitter to split dataset into a list of tuple (train set, test set) 

        Yields:
            tuple['Dataset', 'Dataset']: Train set and Test set 
        """
        # Split the dataset as many times as the splitter requires it
        for i_train, i_test in splitter(self.X, self.__y):
            X_train = self.X.iloc[i_train].copy()
            X_test = self.X.iloc[i_test].copy()
            y_train = self.__y[i_train].copy()
            y_test = self.__y[i_test].copy()

            ds_train = Dataset(X_train, y_train)
            ds_test = Dataset(X_test, y_test)

            yield (ds_train, ds_test)
        
    def get_columns_names_by_type(self, types:list[DataType]) -> list[str]:
        """
        Get names of all the columns with DataType in types

        Args:
            types (list[DataType]): List of Datatype to search

        Returns:
            list[str]: columns names
        """
        if not isinstance(types, list):
            types = [types]

        return [
            column
            for column, type in self.columns_types.items()
            if type in types
        ]

    def __detect_data_type(self, column_name:str) -> DataType:
        """
        Detect data type of a column

        Args:
            column_name (str): Name of the column to analyse

        Returns:
            DataType: Type of the columns
        """
        column_value = self.X[column_name]
        detected:DataType = None
        if column_value.dtype == object:
            if self.__string_column_to_date(column_name):
                detected = DataType.DATE
            else:
                if (len(column_value.unique()) / len(column_value) < 0.05 \
                    or len(column_value.unique()) < 7):
                    detected = DataType.CATEGORICAL
                elif column_value.astype(str).apply(len).max() <= 85:
                    detected = DataType.SHORT_TEXT
                else:
                    detected = DataType.TEXT
        elif np.issubdtype(column_value.dtype, np.number):
            detected = DataType.NUMERIC
        elif np.issubdtype(column_value.dtype, np.datetime64):
            detected = DataType.DATE
            
        return detected
    
    def __detect_columns_types(self) -> dict:
        """
        Detect column type of all features in X

        Returns:
            dict: column name as key, data type as value
        """
        types = {}
        for column in self.features:
            types[column] = self.__detect_data_type(column)
            
        return types
    
    def __string_column_to_date(self, column_name: str) -> bool:
        """
        Convert a column of string value (with data format) to a column of dates
        
        Args:
            column_name (str): Column to convert

        Returns:
            bool: Successful convert ?
        """
        threshold_count:float = self.X[column_name].count() * 0.95
        new_columns:pd.DataFrame = self.X[column_name].apply(self.__string_value_to_date)
            
        if new_columns.count() >= threshold_count:
            self.X[column_name] = new_columns.replace(pd.NaT, None)
            return True

        return False
            
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
            date_formats = ['%Y-%M-%d', '%d-%M-%Y', '%Y/%M/%d', '%d/%M/%Y', None]
        elif isinstance(date_formats, list):
            date_formats = list(date_formats)
        
        for date_format in date_formats:
            try:
                return pd.to_datetime(value, format=date_format)
            except ValueError:
                pass # Let's try the next date format

        return pd.NaT
