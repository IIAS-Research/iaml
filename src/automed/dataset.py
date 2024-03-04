import copy
import numpy as np
import pandas as pd
from typing import Iterator

from sklearn.utils.multiclass import type_of_target

from .data_type import DataType


class Dataset:
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, resample: list=[], splitted:bool=False):
        self.__X = X
        self.__y = y
        
        self.__resample_stack = resample
        
        if splitted: 
            self.__apply_resample()
        
        self.__splitted = splitted

        self.columns_types = self.__detect_columns_types()
        self.type_of_target = type_of_target(self.__y)
    
    
    @property
    def splitted(self) -> bool:
        return self.__splitted
    
    @property
    def features(self) -> list:
        return self.X.columns.to_list()

    @property
    def labels(self) -> list:
        return self.__y.columns.to_list()
    
    @property
    def X(self) -> pd.DataFrame:
        return self.__X

    @property
    def y(self) -> pd.DataFrame:
        if self.__splitted:
            if self.is_multilabel:
                return self.__y
            else:
                return self.__y.squeeze(axis=0).values.ravel()
        else:
            raise Exception("Dataset mush be splitted before access to y value")
    
    @property
    def is_multilabel(self):
        return len(self.__y.columns) > 1

    def copy(self, deep=True) -> 'Dataset':
        if deep:
            return copy.deepcopy(self)
        return copy.copy(self)

    def transform(self, method) -> None:
        if self.__splitted:
            raise Exception("Cannot edit a splitted Dataset")
        
        self.__X = method(self.X)
        self.columns_types = self.__detect_columns_types()
        
    def resample(self, method) -> None:
        if self.__splitted:
            raise Exception("Cannot edit a splitted Dataset")
        
        self.__resample_stack.append(method)
        
    def __apply_resample(self) -> None:
        for method in self.__resample_stack:
            self.__X, self.__y = method(self.X, self.__y)
        self.columns_types = self.__detect_columns_types()
        
    def split(self, splitter: callable) -> Iterator[tuple['Dataset', 'Dataset']]: 
        if self.__splitted:
            raise Exception("Dataset alreadly splitted !")
        
        # Split the dataset as many times as the splitter requires it
        for i_train, i_test in splitter(self.X, self.__y):
            X_train = self.X.iloc[i_train].copy()
            X_test = self.X.iloc[i_test].copy()
            y_train = self.__y.iloc[i_train].copy()
            y_test = self.__y.iloc[i_test].copy()

            ds_train = Dataset(X_train, y_train, resample=self.__resample_stack, splitted=True)
            ds_test = Dataset(X_test, y_test, splitted=True)

            yield (ds_train, ds_test)
        
        # Yield the whole dataset as training data
        yield Dataset(self.X.copy(), self.__y.copy(), resample=self.__resample_stack, splitted=True), None
        
    def get_columns_names_by_type(self, types) -> list:
        if type(types) != list:
            types = [types]

        return [
            column
            for column, type in self.columns_types.items()
            if type in types
        ]

    # Detect data types
    def detect_data_type(self, column_name):
        column_value = self.X[column_name]
        if column_value.dtype == object:
            if self.__string_column_to_date(column_name):
                return DataType.DATE
            else:
                if (len(column_value.unique()) / len(column_value) < 0.05 or len(column_value.unique()) < 7):
                    return DataType.CATEGORICAL
                elif column_value.astype(str).apply(len).max() <= 85:
                    return DataType.SHORT_TEXT
                else:
                    return DataType.TEXT
        elif np.issubdtype(column_value.dtype, np.number):
            return DataType.NUMERIC
        elif np.issubdtype(column_value.dtype, np.datetime64):
            return DataType.DATE
    
    # Return a dict of with column name as key and value as type of data
    def __detect_columns_types(self) -> dict:
        types = {}
        for column in self.features:
            types[column] = self.detect_data_type(column)
            
        return types
    
    def __string_column_to_date(self, column_name: str):
        threshold_count = self.X[column_name].count() * 0.95
        new_columns = self.X[column_name].apply(self.__string_value_to_date)
            
        if new_columns.count() >= threshold_count:
            self.X[column_name] = new_columns.replace(pd.NaT, None)

            return True

        return False
            
    def __string_value_to_date(self, value, date_formats = ['%Y-%M-%d', '%d-%M-%Y', '%Y/%M/%d', '%d/%M/%Y', None]):
        if type(date_formats) != list:
            date_formats = list(date_formats)
        
        for date_format in date_formats:
            try:
                return pd.to_datetime(value, format=date_format)
            except ValueError:
                pass

        return pd.NaT
    
    def compute_metric(self, model, metric):
        y_pred = model.predict(self.X, model_only = self.__splitted)
        return metric.compute(self.__y, y_pred)
