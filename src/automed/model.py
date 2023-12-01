import pandas as pd
import pickle

from .output import Input, Output


def sklearn_predict(input: Input, model):
    return model.predict(input.dataset.pred_data)


def sklearn_transform(input: Input, model):
    return model.transform(input.dataset.pred_data)


class Model:
    def __init__(self, stack: list[callable] = [], sklearn_model = None) -> None:
        self.__stack = stack.copy()
        self.sklearn_model = sklearn_model
    

    @property
    def stack(self):
        return self.__stack.copy()
    

    def add_to_stack(self, model_or_function, tag, *args, **kw) -> None:
        if tag == 'predict':
            if hasattr(model_or_function, 'predict') and callable(model_or_function.predict):
                # model_or_function is probably a sklearn model
                self.sklearn_model = model_or_function
        elif tag == 'transform':
            if hasattr(model_or_function, 'transform') and callable(model_or_function.predict):
                self.__stack.append((sklearn_transform, (model_or_function,), {}))
            else:
                # model_or_function is a function
                self.__stack.append((model_or_function, args, kw))
    

    def copy(self) -> 'Model':
        return Model(self.__stack, self.sklearn_model)
    

    def pickle(self) -> bytes:
        return pickle.dumps(self)
    

    def run(self, dataset: pd.DataFrame) -> Output:
        last_output = dataset
        for (function, args, kw) in self.__stack:
            last_output = function(last_output, [], *args, **kw)
        
        return last_output
