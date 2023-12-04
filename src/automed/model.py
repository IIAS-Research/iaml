import pandas as pd
import pickle

from .output import Input, Output


def sklearn_predict(input: Input, model):
    return model.predict(input.dataset.pred_data)


def sklearn_transform(input: Input, model):
    return model.transform(input.dataset.pred_data)


class Model:
    def __init__(self, stack: list[callable] = [], ml_model = None, predict_function=None) -> None:
        self.__stack = stack.copy()
        self.ml_model = ml_model
        self.predict_function = predict_function
    

    @property
    def stack(self):
        return self.__stack.copy()
    
    def __str__(self) -> str:
        if self.have_model:
            return self.ml_model.__str__()
        else:
            return "Unknown model"
    

    def add_to_stack(self, model_or_function, tag, *args, **kw) -> None:
        if tag == 'predict':
            if hasattr(model_or_function, 'predict') and callable(model_or_function.predict):
                # model_or_function is probably a sklearn model
                self.ml_model = model_or_function
        elif tag == 'transform':
            if hasattr(model_or_function, 'transform') and callable(model_or_function.predict):
                self.__stack.append((sklearn_transform, (model_or_function,), {}))
            else:
                # model_or_function is a function
                self.__stack.append((model_or_function, args, kw))
                
    def set_model(self, model, function, *args, **kw) -> 'Model':
        new_model = self.copy()
        new_model.ml_model = model
        new_model.predict_function = function
        
        return new_model

    def copy(self) -> 'Model':
        return Model(self.__stack, self.ml_model, self.predict_function)
    

    def pickle(self) -> bytes:
        return pickle.dumps(self)
    
    @property
    def have_model(self):
        if self.ml_model and self.predict_function and callable(self.predict_function):
            return True
        return False
    
    def predict(self, X:pd.DataFrame, model_only:bool = False):
        if not(self.have_model):
            raise Exception("Model need to be set before predict")
        
        if not(model_only):
            X = self.run(X)
        
        return self.predict_function(self.ml_model, X)

    def run(self, dataset: pd.DataFrame) -> Output:
        last_output = dataset
        for (function, args, kw) in self.__stack:
            last_output = function(last_output, [], *args, **kw)
        
        return last_output
