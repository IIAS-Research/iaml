import pandas as pd
import numpy as np

class Dataset:
    
    def __init__(self, train_data, test_data=None, label_name=None):
        self.__data = {
            'train': {
                'features': pd.DataFrame(),
                'disabled': pd.DataFrame(),
                'labels': pd.Series()
                },
            'test': {
                'features': pd.DataFrame(),
                'disabled': pd.DataFrame(),
                'labels': pd.Series()
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
                self.res_label(env=[dset])
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
        
    
    # TODO Traduire en anglais (commentaire, nom de variable/fonction, etc)
    # Détection des types de données
    def detecter_types_de_donnees(self, colonne):
        # Cette condition vérifie 2 choses: 
        # Si le rapport entre le nombre de valeurs uniques dans la colonnnne et le nombre total de valeurs dans cette colonne est inférieur à 5%.
        # Le nombre total  de valeurs uniques dans cette colon,ne est inférieur à 7.
        if len(colonne.unique()) / len(colonne) < 0.05 or len(colonne.unique()) < 7:
            return 'catégoriel' # TODO Remplacer par un enum ou quelque chose comme ça
        elif np.issubdtype(colonne.dtype, np.number):
            return 'Numérique'
        elif np.issubdtype(colonne.dtype, np.datetime64): # TODO Vérifier aussi dans les string si ce n'est pas une date
            return 'Date'
        elif colonne.astype(str).apply(len).max() <= 85:
            return 'Chaine de caractères'
        else:
            return 'Texte libre'
    
    def types_de_données(self, data):
        # Dictionnaire vide pour stoker le nom de chaque colonne ainsi que son type de données
        types = {}
        for column in data.columns:
            resultat = self.detecter_types_de_donnees(data[column])
            types[column] = resultat
        return types  


    # La fonction disable_column affiche des valeurs NaN dans le __data['train' ou 'test']['disabled'], la fonction dc affiche la colonne desactivée ainsi que toutes ses valeurs
    # Désactiver une colonne
    def dc(self, column): 
        for dset in ['train', 'test']:
            if column in self.__data[dset]['features'].columns:
                dis_col = self.__data[dset]['features'].pop(column)
                self.__data[dset]['disabled'][column] = dis_col
            else:
                raise Exception(f"La colonne '{column}' n'existe pas ! ")
    
    # Désactiver plusieurs colonnes      
    def dcs(self, columns):
        for column in columns:
            self.dc(column)
            
    # Activer une colonne 
    def ec(self, column):
        for dset in ['train', 'test']:
            if column in self.__data[dset]['disabled']:
                act_col = self.__data[dset]['disabled'].pop(column)
                self.__data[dset]['features'][column] = act_col
            else:
                raise Exception(f"La colonne '{column}' n'existe pas ! ")
            
    # Activer plusieurs colonnes  
    def ecs(self, columns):
        for column in columns:
            self.ec(column)
            
            
    # Reset_label             
    def res_label(self, env=['train', 'test']):
        for dset in env:
            if self.__data[dset]['labels'].name:
                # Récpeation du nom de la colonne
                column = self.__data[dset]['labels'].name
                # Insertion de la colonne label dans le ['train' ou 'test']['features']
                self.__data[dset]['features'][column] = self.__data[dset]['labels']
                # Réinitialiser le ['train' ou 'test']['labels']
                self.__data[dset]['labels'] = pd.Series()
    
    def show_labels(self):
        print("LABELS")
        print("TRAIN",  self.__data['train']['features'].columns,  self.__data['train']['labels'])
        print("TEST",  self.__data['test']['features'].columns,  self.__data['test']['labels'])
                    
                