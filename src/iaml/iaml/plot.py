"""
[PLOT] Parent of all others Plot, implement the default behavior
"""
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
        
    def compute(self, estimator:'IAMLPipeline', X:pd.DataFrame, y:pd.DataFrame, **kwargs):
        """
        Compute plot given X, y. 
        Must be overwrote by children classes
        """
        raise NotImplementedError('Subclass must implement abstract method')
    
    def suitable(self, type_of_target:str) -> bool:  # pylint: disable=unused-argument
        """
        Does this plot is usable for a given type_of_target ?
        """
        return False

    def to_json(self) -> dict:
        return {'title': self.title,
            'description': self.description,
            'image': self.image}
    

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
