import pandas as pd
import pickle

from .output import Input, Output
from sklearn.pipeline import Pipeline


class Model:
    def __init__(self, stack: list[callable] = [], ml_model = None) -> None:
        self.__stack = stack.copy()
        self.ml_model = ml_model
    

    @property
    def stack(self):
        return self.__stack.copy()
    
    def __str__(self) -> str:
        if self.have_model:
            return self.ml_model.__str__()
        else:
            return "Unknown model"
    

    def add_to_stack(self, instance) -> None:
        self.__stack.append(instance)
                
    def set_model(self, instance) -> 'Model':
        new_model = self.copy()
        new_model.ml_model = instance
        
        return new_model

    def copy(self) -> 'Model':
        return Model(self.__stack, self.ml_model)
    

    def pickle(self) -> bytes:
        return pickle.dumps(self)
    
    @property
    def have_model(self):
        if self.ml_model and callable(self.ml_model.predict):
            return True
        return False
    
    def predict(self, X:pd.DataFrame, model_only:bool = False):
        if not(self.have_model):
            raise Exception("Model need to be set before predict")
        
        if not(model_only):
            X = self.run(X)
        
        return self.ml_model.predict(X)

    def run(self, dataset: pd.DataFrame) -> Output:
        last_output = dataset
        for instance in self.__stack:
            last_output, _ = instance.transform(last_output, [])
            
            
        return last_output
