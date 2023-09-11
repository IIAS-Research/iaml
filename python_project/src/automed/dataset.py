import pandas as pd

class Dataset:
    data = {
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
    label_column = None
    
    def __init__(self, train_data, test_data=None, label_name=None):
        self.data['train']['features'] = train_data
        
        if type(test_data) != type(None):
            self.data['test']['features'] = test_data
        else:
            self.data['test']['features'] = pd.DataFrame(columns=train_data.columns)
            
        if label_name:
            self.set_label(label_name)
        
        
    def _merge_df(self, main_df, add_df):
        return main_df.join(add_df)
    
    
    def set_label(self, label_name):
        for dset in ['train', 'test']:
            if label_name not in self.data[dset]['features'].columns:
                raise Exception("Column must exist")
            else:
                if not self.data[dset]['labels'].empty:
                    self.data[dset]['features'] = self._merge_df(self.data[dset]['features'], self.data[dset]['labels'])
                    self.data[dset]['labels'] = None
                    
                self.data[dset]['labels'] = self.data[dset]['features'][label_name]
                self.data[dset]['features'].drop(columns=[label_name], inplace=True)
            
        
    def disable_column(self, column):
        for dset in ['train', 'test']:
            if column in self.data[dset]['features'].columns:
                self.data[dset]['disabled'] = self._merge_df(self.data[dset]['disabled'], self.data[dset]['features'][column])
                self.data[dset]['features'].drop(columns=[column], inplace=True)
            else:
                raise Exception(f"Column '{column}' does not exist")
        
    def disable_columns(self, columns):
        for column in columns:
            self.disable_column(column)
            
    def enable_column(self, column):
        for dset in ['train', 'test']:
            if column in self.data[dset]['disabled'].columns:
                self.data[dset]['features'] = self._merge_df(self.data[dset]['features'], self.data[dset]['disabled'][column])
                self.data[dset]['disabled'].drop(columns=[column], inplace=True)
            else:
                raise Exception(f"Column '{column}' does not exist")
        
    def enable_columns(self, columns):
        for column in columns:
            self.enable_column(column)
            
    def active_columns(self):
        return self.data['train']['features'].columns
    
    @property      
    def train_data(self):
        return self.data['train']['features']
    
    @property
    def X_train(self):
        return self.data['train']['features']
    
    @X_train.setter
    def X_train(self, value):
        self.data['train']['features'] = value
    
    @property      
    def test_data(self):
        return self.data['test']['features']
    
    
    @property
    def X_test(self):
        return self.data['test']['features']
    
    @X_test.setter
    def X_test(self, value):
        self.data['test']['features'] = value
    
    @property      
    def train_labels(self):
        return self.data['train']['labels']
    
    @property      
    def y_train(self):
        return self.data['train']['labels']
    
    @y_train.setter
    def y_train(self, value):
        self.data['train']['labels'] = value
    
    @property     
    def test_labels(self):
        return self.data['test']['labels']
    
    @property      
    def y_test(self):
        return self.data['test']['labels']
    
    @y_test.setter
    def y_test(self, value):
        self.data['test']['labels'] = value
        