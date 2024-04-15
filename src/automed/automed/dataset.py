"""
Encapsulate X, y data to be used by Steps
Add features like data type detection and splitting 
"""
import copy
from typing import Iterator, TYPE_CHECKING
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import numpy as np
import pandas as pd

from .data_type import DataType
from .type_of_target import type_of_target


if TYPE_CHECKING:
    from .auto_pipeline import AutoPipeline

class Dataset:
    """
    Encapsulate X, y data to be used by Steps
    Add features like data type detection and splitting
    """
    
    def __init__(self, X:pd.DataFrame, y:list=None, groups:pd.DataFrame=None, columns_types:dict=None):
        self.__X:pd.DataFrame = X
        self.__y:np.array = np.array(y)
        
        
        if type(groups) in [pd.Series, list, np.array]:
            self.groups = pd.DataFrame(groups)
        else:
            self.groups = groups
        self.columns_types:dict = columns_types if columns_types else {}
        self.__detect_columns_types()
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
    
    def decline(self, X, y, groups=None) -> 'Dataset':
        if not groups:
            groups = self.groups
        return Dataset(X, y, groups=groups, columns_types=self.columns_types)
    
    def sample(self, n) -> 'Dataset':
        """
        Return a dataset with a sample of data

        Args:
            n (int): Number of line in the sample dataset

        Returns:
            Dataset: dataset with a sample of data
        """
        if isinstance(n, float):
            n = int(self.X.shape[0]*n)
            
        if n >= self.X.shape[0]:
            return self.decline(X, y)
        
        if self.type_of_target == 'continuous':
            _, test_idx = next(
                ShuffleSplit(n_splits=1, test_size=n, random_state=42
                ).split(self.X, self.y))
        else:
            try: 
                _, test_idx = next(
                    StratifiedShuffleSplit(n_splits=1, test_size=n, random_state=42
                    ).split(self.X, self.y))
            except ValueError:
                _, test_idx = next(
                    ShuffleSplit(n_splits=1, test_size=n, random_state=42
                    ).split(self.X, self.y))
                
        return self.decline(self.X.iloc[test_idx], self.y[test_idx])
    
    def transform(self, method:callable) -> None:
        """
        Apply transform method to X or y data based on the method signature

        Args:
            method (callable): Callable to apply. Will be call with X or y as parameter

        Returns:
            Dataset: Transformed dataset
        """
        self.__X = method(self.__X)
        self.__detect_columns_types()
    
    @property
    def has_groups(self) -> bool:
        """
        Groups exist ?
        """
        return self.groups is not None and not self.groups.empty
    
    def resample(self, resampler:callable) -> 'Dataset':
        """
        Apply a resampler on X, y and groups data. 

        Args:
            resampler (callable): Resampler method
        """
        
        if self.has_groups:
            # Merge groups with X
            X = self.X.reset_index(drop=True).join(self.groups.reset_index(drop=True))
    
            # Resampler
            X, y = resampler(X, self.y)
            
            # Split groups and X
            return self.decline(X.drop(columns=self.groups.columns),
                            y,
                            groups=X[self.groups.columns])
        return self.decline(*resampler(self.X, self.y))
        
    def split(self, splitter: callable, *args, **kwargs) -> Iterator[tuple['Dataset', 'Dataset']]: 
        """
        Use splitter to split dataset into a list of tuple (train set, test set) 

        Yields:
            tuple['Dataset', 'Dataset']: Train set and Test set 
        """
    
        # Split the dataset as many times as the splitter requires it
        for i_train, i_test in splitter(self.X, self.y, *args, **kwargs):
            X_train = self.X.iloc[i_train].copy()
            X_test = self.X.iloc[i_test].copy()
            y_train = self.__y[i_train].copy()
            y_test = self.__y[i_test].copy()

            ds_train = self.decline(X_train, y_train)
            ds_test = self.decline(X_test, y_test)

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
            for column, (dtype, type) in self.columns_types.items()
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
            
        return column_value.dtype, detected
    
    @property
    def needed_estimator(self) -> str:
        """
        Kind of estimator needed for this dataset
        """
        return 'regressor' if self.type_of_target == 'continuous' else 'classifier'
    
    def __detect_columns_types(self) -> None:
        """
        Detect column type of all features in X

        Returns:
            dict: column name as key, data type as value
        """
        new_types = {}
        for column in self.features:
            if column not in self.columns_types \
                or self.columns_types[column][0] != self.X[column].dtype:
                new_types[column] = self.__detect_data_type(column)
            else:
                new_types[column] = self.columns_types[column]
                
        self.columns_types = new_types
