import copy
import numpy as np
import pandas as pd

from sklearn.utils.multiclass import type_of_target

from .data_type import DataType


class Dataset:
    def __init__(self, X: pd.DataFrame, Y: pd.DataFrame):
        self.__features = pd.concat((X, Y), axis=1)
        self.__labels = Y.columns.to_list()
        self.__disabled_columns = []

        self.columns_types = self.__detect_columns_types()
        self.type_of_target = type_of_target(self.Y) # an array with type of target (multiclass, binary, etc)
    
    @property
    def features(self) -> pd.DataFrame:
        return self.__features.drop(self.__disabled_columns, axis=1)

    @property
    def labels(self) -> list:
        return self.__labels
    
    @property
    def X(self) -> pd.DataFrame:
        return self.features.drop(self.__labels, axis=1)
    
    @X.setter
    def X(self, X: pd.DataFrame) -> None:
        for column in X.columns:
            if column in self.__disabled_columns:
                raise TypeError(f"Column '{column}' is supposed to be disabled, but is defined in new X.")
            
        self.__features[self.X.columns] = X

    @property
    def Y(self) -> pd.DataFrame:
        return self.features[self.__labels]

    @Y.setter
    def Y(self, Y: pd.DataFrame) -> None:
        for column in Y.columns:
            if column in self.__disabled_columns:
                raise TypeError(f"Column '{column}' is supposed to be disabled, but is defined in new Y.")

        self.__features[self.__labels] = Y
    
    @property
    def is_multilabel(self):
        return len(self.__labels) > 1

    def copy(self, deep=True):
        if deep:
            return copy.deepcopy(self)

        return copy.copy(self)

    def apply(self, method, *args, **kw):
        self.__features = pd.concat(method(self.X, self.Y, *args, **kw), axis=1)
        self.columns_types = self.__detect_columns_types()
        # TODO find and document changes
    
    def get_columns_names_by_type(self, types):
        if type(types) != list:
            types = [types]

        return [
            column
            for column, type in self.columns_types.items()
            if type in types and column not in self.__disabled_columns and column not in self.labels
        ]
                
    def active_columns(self):
        return [ c for c in self.__features.columns if c not in self.__disabled_columns ]

    # Detect data types
    def detect_data_type(self, column_name):
        column_value = self.__features[column_name]
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
    def __detect_columns_types(self):
        types = {}
        for column in self.__features.columns:
            types[column] = self.detect_data_type(column)
            
        return types

    # Disable a column with delete it
    def disable_column(self, column):
        if column not in self.__disabled_columns:
            self.__disabled_columns.append(column)
        else:
            raise Exception(f"Column '{column}' is already disabled!")
    
    # Disable several columns
    def disable_columns(self, columns):
        for column in columns:
            self.disable_column(column)
            
    # Enable column from disabled dataset
    def enable_column(self, column):
        if column in self.__disabled_columns:
            self.__disabled_columns.pop(column)
        else:
            raise Exception(f"Column '{column}' is already enabled!")
            
    # Enable several columns
    def enable_columns(self, columns):
        for column in columns:
            self.enable_column(column)
    
    def __string_column_to_date(self, column_name: str):
        threshold_count = self.__features[column_name].count() * 0.95
        new_columns = self.__features[column_name].apply(self.__string_value_to_date)
            
        if new_columns.count() >= threshold_count:
            self.__features[column_name] = new_columns.replace(pd.NaT, None)

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

    def __getitem__(self, key) -> pd.DataFrame | pd.Series:
        return self.__features.drop(self.__disabled_columns, axis=1)[key]


class TrainingDataset():

    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.__X_train = X_train
        self.__X_test = X_test
        self.__Y_train = Y_train
        self.__Y_test = Y_test

    @property
    def X_train(self):
        return self.__X_train

    @property
    def X_test(self):
        return self.__X_test
    
    @property
    def Y_train(self):
        if self.is_multilabel:
            return self.__Y_train
        else:
            return self.__Y_train.squeeze(axis=0).values.ravel()
    
    @property
    def Y_test(self):
        if self.is_multilabel:
            return self.__Y_test
        else:
            return self.__Y_test.squeeze(axis=0).values.ravel()
    
    @property
    def is_multilabel(self):
        return len(self.__Y_train.columns) > 1
        
    def compute_metric(self, model, metric):
        Y_pred = model.predict(self.__X_test, model_only=True)
        Y_true = self.__Y_test.copy()

        return metric.compute(Y_true, Y_pred, multilabel=self.is_multilabel)
