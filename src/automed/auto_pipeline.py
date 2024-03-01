import pandas as pd
import pickle

from sklearn.pipeline import Pipeline

from .explanation import Explanation


class AutoPipeline(Pipeline):
    def __init__(self, steps: list[tuple[str, object]] = [], explanations: list[Explanation] = []) -> None:
        self.explanations = explanations
        self.steps = steps.copy()
        
    def fit(self, x, y):
        raise Exception("Fit is not usable with AutoMed Pipeline. You have to use AutoMed.run()")
    
    def fit_predict(self, x, y):
        self.fit(x, y)
        
    def fit_transform(self, x, y):
        self.fit(x, y)
        
    @property
    def model(self):
        return self.steps[-1][1] if self.have_model else None
    
    # def __str__(self) -> str:
    #     if self.have_model:
    #         return self.ml_model.__str__()
    #     else:
    #         return "Unknown model"
    

    def add_to_stack(self, instance) -> None:
        self.steps.append((instance.__str__(), instance))
                
    def set_model(self, instance) -> 'Model':
        if self.have_model:
            self.steps.pop(-1)
        self.add_to_stack(instance)
        
        return self

    def copy(self) -> 'AutoPipeline':
        return AutoPipeline(self.steps, self.explanations)
    

    def pickle(self) -> bytes:
        return pickle.dumps(self)
    
    @property
    def have_model(self):
        if self.steps and hasattr(self.steps[-1][1], 'predict'):
            return True
        return False
    
    def predict(self, X:pd.DataFrame, model_only:bool = False):
        if not(self.have_model):
            raise Exception("Model need to be set before predict")
        
        if not(model_only):
            return super().predict(X)
        
        return self.model.predict(X)
    
    def add_explanation(self, step, processings: list[str]):
        self.explanations.append(Explanation(step, processings))
