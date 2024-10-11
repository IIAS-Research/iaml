"""
[PLOT] Parent of all others Plot, implement the default behavior
"""
import base64
from functools import wraps
import matplotlib.pyplot as plt
import pandas as pd

class Plot:
    """
    [PLOT] Parent of all others Plot, implement the default behavior
    """
    
    title = "Here is the plot title"
    description = "Here is an explanation of how this plot work"
    
    def __init__(self):
        self.__visualizer = None  # pylint: disable=unused-private-member
        self._binary_image = None
    
    @property
    def image(self):
        """
            Get binary representation of the plot image
        """
        if self._binary_image is not None:
            return self._binary_image.getvalue()
        
        raise AttributeError("Plot must be computed before")
    
    @property
    def b64_image(self):
        """
            Get b64 representation of the plot image
        """
        return base64.b64encode(self.image).decode()
        
    def _compute(self, estimator:'IAMLPipeline', X:pd.DataFrame, y:pd.DataFrame, **kwargs):
        """
        Compute plot given X, y. 
        Must be overwrote by children classes
        """
        raise NotImplementedError('Subclass must implement abstract method')
    
    @classmethod
    def suitable(cls, type_of_target:str) -> bool:  # pylint: disable=unused-argument
        """
        Does this plot is usable for a given type_of_target ?
        """
        return False

    def to_json(self, data_format='binary') -> dict:
        """
        Convert plot into json with name, description and b64 image
        """
        match data_format:
            case 'binary':
                data = self.image
            case 'b64':
                data = self.b64_image
            case _:
                raise AttributeError('Invalide Data Format')
        
        return {'title': self.title,
            'description': self.description,
            'image': data}
        
    def to_markdown(self) -> str:
        base64_md = f"![{self.title}](data:image/png;base64,{self.b64_image})"
        return "\n\n".join([f"# {self.title}", self.description, base64_md])
    

class MetricPlot(Plot):
    """
    To be used by performance Explainer
    """
    def __init__(self, estimator:'IAMLPipeline', 
                X:pd.DataFrame, y:pd.DataFrame, 
                X_train:pd.DataFrame, y_train:pd.DataFrame, 
                **kwargs):
        self._compute(estimator, X, y, X_train=X_train, y_train=y_train, **kwargs)
    

def capture(func):
    """
    A decorator to capture a matplotlib plot into a BytesIO object and return it
    as binary data instead of showing it.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        plt.close()
        return ret
    
    return wrapper
