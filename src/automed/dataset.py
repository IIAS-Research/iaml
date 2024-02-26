import pandas as pd
import numpy as np
from .data_type import DataType
from sklearn.utils.multiclass import type_of_target

    
class Dataset:
    def __init__(self, train_data, test_data=None, label_name=None):
        self.__data = {
            'train': {
                'features': pd.DataFrame(),
                'disabled': pd.DataFrame(),
                'labels': pd.DataFrame()
                },
            'test': {
                'features': pd.DataFrame(),
                'disabled': pd.DataFrame(),
                'labels': pd.DataFrame()
            }
        }
        
        self.__data['train']['features'] = train_data
        
        if type(test_data) != type(None):
            self.__data['test']['features'] = test_data
        else:
            self.__data['test']['features'] = pd.DataFrame(columns=train_data.columns)
            
        self.columns_types = self.__detect_columns_types() 
        
        self.type_of_target = None # Will be an array with type of target (multiclass, binary, etc)
        
        if label_name:
            self.set_label(label_name)
            
    @classmethod
    def from_splited_data(cls, X_train, y_train, X_test, y_test):
        dataset = cls(train_data=X_train)
        dataset.y_train = y_train
        dataset.X_test = X_test
        dataset.y_test = y_test
        
        return dataset
    
    def split(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    

    def copy(self, deep=True):
        return Dataset.from_splited_data(
            self.X_train.copy(deep=deep),
            self.y_train.copy(deep=deep),
            self.X_test.copy(deep=deep),
            self.y_test.copy(deep=deep)
        )
    
    # Compute metrics
    def compute_metric(self, model, metric):
        # To avoid warninggs ( UserWarning: X has feature names, but GaussianNB was fitted without feature names)
        y_pred = model.predict(self.X_test, model_only = True)
        if not isinstance(y_pred, np.ndarray):
            y_pred = y_pred.toarray()
        else:
            return metric.compute(self.y_test, y_pred)

    def apply(self, method, only_train=False, *args, **kw):
        self.X_train, self.y_train = method(self.X_train, self.y_train, *args, **kw)
        if not only_train:
            self.X_test, self.y_test = method(self.X_test, self.y_test, *args, **kw)
        # TODO find and document changes
        # TODO With change compute again columns types 
        
    def __find_differencies(self, old_dataset):
        return ['No diff']
        
    def _merge_df(self, main_df, add_df):
        return main_df.join(add_df)
    
    def get_columns_names_by_type(self, types):
        if type(types) != list:
            types = [types]
            
        return list(dict(filter(
            lambda pair: pair[1] in types,
            self.usable_columns_types.items())).keys())
        
    @property
    def usable_columns_types(self):
        return dict(filter(
            lambda pair: pair[0] in self.X_train.columns,
            self.columns_types.items()))
        
    def set_label(self, label_name):
        for data_env in ['train', 'test']:
            if set(label_name).issubset(set(self.__data[data_env]['features'].columns) | set(self.__data[data_env]['labels'].columns)):
                self.reset_label(env=[data_env])
                for column in label_name:
                    self.__data[data_env]['labels'][column] = self.__data[data_env]['features'].pop(column)
            else:
                raise Exception("Columns must exist")
        
        self.compute_type_of_target()
        
    # TODO 
    # DEBUG
    # Permet de forcer le mono label -> fix temporaire a supprimer quand toute la class dataset forcera le monolabel
    # Utiliser temporairement pour finaliser les metrics
    # TODO supprimer aussi toutes les lignes avec # TODO DEBUG TMP FIX MONOLABEL
    def debug_force_monolabel(self):
        col = self.labels_columns[0]
        self.__data['train']['labels'] = pd.DataFrame(self.__data['train']['labels'][col])
        self.__data['test']['labels'] = pd.DataFrame(self.__data['test']['labels'][col])
        self.compute_type_of_target()
        
    def compute_type_of_target(self) -> None: 
        self.type_of_target = type_of_target(self.y_train)
                
    def active_columns(self):
        return self.__data['train']['features'].columns
    
    @property
    def is_multilabel(self):
        return len(self.labels_columns) > 1
    
    @property
    def labels_columns(self):
        return self.__data['train']['labels'].columns
    @property      
    def train_data(self):
        return self.__data['train']['features'].copy(deep=True)
    
    @property
    def X_train(self):
        return self.train_data
    
    @X_train.setter
    def X_train(self, value):
        self.__data['train']['features'] = value
    
    @property      
    def __test_data(self):
        return self.__data['test']['features']
    
    
    @property
    def __X_test(self):
        return self.__test_data
    
    @__X_test.setter
    def __X_test(self, value):
        self.__data['test']['features'] = value
    
    @property      
    def train_labels(self):
        return self.__data['train']['labels'].copy(deep=True)
    
    @property      
    def y_train(self):
        if self.is_multilabel:
            return self.train_labels
        else:
            return self.train_labels[self.train_labels.columns[0]]
    
    @y_train.setter
    def y_train(self, value):
        self.__data['train']['labels'] = pd.DataFrame(value)
    
    @property     
    def __test_labels(self):
        return self.__data['test']['labels']
    
    @property      
    def __y_test(self):
        if self.is_multilabel:
            return self.__test_labels
        else:
            return self.__test_labels[self.__test_labels.columns[0]]
    
    # TODO DEBUG purpose -> to remove
    @property
    def check_X_test(self):
        return self.__test_data
    
    # TODO DEBUG purpose -> to remove
    @property
    def check_y_test(self):
        return self.y_test
    
    @__y_test.setter
    def __y_test(self, value):
        self.__data['test']['labels'] = pd.DataFrame(value)
        
    # Detect data types
    def detect_data_type(self, column_name):
        column_value = self.X_train[column_name]
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
        for column in self.X_train.columns:
            types[column] = self.detect_data_type(column)
            
        return types  

    # Disable a column with delete it
    def disable_column(self, column): 
        for data_env in ['train', 'test']:
            if column in self.__data[data_env]['features'].columns:
                self.__data[data_env]['disabled'][column] = self.__data[data_env]['features'].pop(column)
            else:
                raise Exception(f"Column '{column}' doesn't exist ! ")
    
    # Disable several columns
    def disable_columns(self, columns):
        for column in columns:
            self.disable_column(column)
            
    # Enable column from disabled dataset
    def enable_column(self, column):
        for data_env in ['train', 'test']:
            if column in self.__data[data_env]['disabled']:
                self.__data[data_env]['features'][column] = self.__data[data_env]['disabled'].pop(column)
            else:
                raise Exception(f"Column '{column}' doesn't exist ! ")
            
    # Enable several columns
    def enable_columns(self, columns):
        for column in columns:
            self.enable_column(column)
                   
    # Reset selected label           
    def reset_label(self, env=['train', 'test']):
        for data_env in env:
            if any(self.__data[data_env]['labels']):
                for column in self.__data[data_env]['labels']:
                    column_name = self.__data[data_env]['labels'].name # Get column name
                    self.__data[data_env]['features'][column_name] = self.__data[data_env]['labels'].pop(column) # Add label to dataset
            self.__data[data_env]['labels'] = pd.DataFrame() # Set label empty
    
    def __string_column_to_date(self, column_name, env=['train', 'test']):
        threshold_count = sum([self.__data[data_env]['features'][column_name].count() for data_env in env]) * 0.95
        
        new_columns = {
            'train': None,
            'test': None
            }
        
        for data_env in env:
            new_columns[data_env] = self.__data[data_env]['features'][column_name].apply(self.__string_value_to_date)
            
        if sum([new_columns[data_env].count() for data_env in env]) >= threshold_count:
            for data_env in env:
                self.__data[data_env]['features'][column_name] = new_columns[data_env].replace(pd.NaT, None)
            return True
        else:
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