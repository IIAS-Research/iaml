import pandas as pd

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
        self.label_column = None
        
        self.__data['train']['features'] = train_data
        
        if type(test_data) != type(None):
            self.__data['test']['features'] = test_data
        else:
            self.__data['test']['features'] = pd.DataFrame(columns=train_data.columns)
            
        if label_name:
            self.set_label(label_name)
            
    @classmethod
    def from_splited_data(cls, X_train, y_train, X_test, y_test):
        dataset = cls(train_data=X_train)
        dataset.y_train = y_train
        dataset.__X_test = X_test
        dataset.__y_test = y_test
        
        return dataset
    
    def split(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.__X_test = X_test
        self.__y_test = y_test
    
    def reset_label(self, env=['train', 'test']):
        for dset in env:
            if not self.__data[dset]['labels'].empty:
                self.__data[dset]['features'] = self._merge_df(self.__data[dset]['features'], self.__data[dset]['labels'])
                self.__data[dset]['labels'] = pd.DataFrame()
                    
        
    
    def copy(self, deep=True):
        return Dataset.from_splited_data(
            self.X_train.copy(deep=deep),
            self.y_train.copy(deep=deep),
            self.__X_test.copy(deep=deep),
            self.__y_test.copy(deep=deep)
        )
        
    def compute(self, model, metric):
        y_pred = model.predict(self.__X_test)
        return metric(self.__y_test, y_pred)

    def apply(self, method, only_train=False, *args, **kw):
        self.X_train, self.y_train = method(self.X_train, self.y_train, *args, **kw)
        if not only_train:
            self.__X_test, self.y_test__X_test = method(self.__X_test, self.__y_test, *args, **kw)
        # TODO find and document changes
        
    def __find_differencies(self, old_dataset):
        return ['No diff']
        
    def _merge_df(self, main_df, add_df):
        return main_df.join(add_df)
    
    
    def set_label(self, label_name):
        for dset in ['train', 'test']:
            if label_name not in self.__data[dset]['features'].columns:
                raise Exception("Column must exist")
            else:
                self.reset_label(env=[dset])
                    
                self.__data[dset]['labels'] = self.__data[dset]['features'][label_name]
                self.__data[dset]['features'].drop(columns=[label_name], inplace=True)
            
        
    def disable_column(self, column):
        for dset in ['train', 'test']:
            if column in self.__data[dset]['features'].columns:
                self.__data[dset]['disabled'] = self._merge_df(self.__data[dset]['disabled'], self.__data[dset]['features'][column])
                self.__data[dset]['features'].drop(columns=[column], inplace=True)
            else:
                raise Exception(f"Column '{column}' does not exist")
        
    def disable_columns(self, columns):
        for column in columns:
            self.disable_column(column)
            
    def enable_column(self, column):
        for dset in ['train', 'test']:
            if column in self.__data[dset]['disabled'].columns:
                self.__data[dset]['features'] = self._merge_df(self.__data[dset]['features'], self.__data[dset]['disabled'][column])
                self.__data[dset]['disabled'].drop(columns=[column], inplace=True)
            else:
                raise Exception(f"Column '{column}' does not exist")
        
    def enable_columns(self, columns):
        for column in columns:
            self.enable_column(column)
            
    def active_columns(self):
        return self.__data['train']['features'].columns
    
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
        return self.train_labels
    
    @y_train.setter
    def y_train(self, value):
        self.__data['train']['labels'] = value
    
    @property     
    def __test_labels(self):
        return self.__data['test']['labels']
    
    @property      
    def __y_test(self):
        return self.__test_labels
    
    # TODO DEBUG purpose -> to remove
    @property
    def check_X_test(self):
        return self.__test_data
    
    @__y_test.setter
    def __y_test(self, value):
        self.__data['test']['labels'] = value
        