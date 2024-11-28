"""
[PLOT] Parent of all others Plot, implement the default behavior
"""
import base64
from functools import wraps
from typing import Any, Dict, TYPE_CHECKING
import matplotlib.pyplot as plt
import pandas as pd
if TYPE_CHECKING:
    from .iaml_pipeline import IAMLPipeline
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
    def b64_image(self) -> str:
        """
        Get b64 representation of the plot image
        
        Returns
        -------
        str
            b64 string image
        """
        return base64.b64encode(self.image).decode()
        
    def _compute(self, estimator: 'IAMLPipeline', X: pd.DataFrame, y: pd.DataFrame, **kwargs):
        """
        Compute plot given X, y. 
        Must be overwrote by children classes
        
        Parameters
        ----------
        estimator : IAMLPipeline
            The pipeline we compute the plot on
        X : pd.DataFrame
            The dataset we wanna compute plot on
        y : pd.DataFrame
            The dataset target we wanna compute plot on
        
        Raises
        ------
        NotImplementedError
            Subclass should be called instead of abstract class
        """
        raise NotImplementedError('Subclass must implement abstract method')
    
    @classmethod
    def suitable(cls, type_of_target: str) -> bool:  # pylint: disable=unused-argument
        """
        Does this plot is usable for a given type_of_target ?
        
        Parameters
        ----------
        type_of_target : str
            The type of target we wanna predict
            
        Returns
        -------
        bool
            Suitable ?
        """
        return False

    def to_json(self, data_format: str='binary') -> Dict[str, Any]:
        """
        Convert plot into json with name, description and b64 image
        
        Parameters
        ----------
        data_format : str
            Type of data we want. Can be 'binary' or 'b64'
        
        Raises
        ------
        AttributeError
            Invalid data format
        
        Returns
        -------
        Dict[str, Any]
            A dictonnary containing plot informations and data
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
        """
        Return plot as markdown format
        
        Returns
        -------
        str
            Markdown formatted plot
        """
        base64_md = f"![{self.title}](data:image/png;base64,{self.b64_image})"
        return "\n\n".join([f"# {self.title}", self.description, base64_md])
    

class MetricPlot(Plot):
    """
    To be used by performance Explainer
    """
    def __init__(self, estimator:'IAMLPipeline', 
                X: pd.DataFrame, y: pd.DataFrame, 
                X_train: pd.DataFrame, y_train: pd.DataFrame, 
                **kwargs) -> Any:
        """
        Initialize a metricp lot
        
        Parameters
        ----------
        estimator : IAMLPipeline
            The pipeline we'll compute the plot on
        X : pd.DataFrame
            The dataframe we wanna compute our plot on
        y : pd.DataFrame
            The dataframe target we wanna compute our plot on
        X_train : pd.DataFrame  
            The dataframe used for training
        y_train : pd.DataFrame
            The dataframe target used for training
        
        Returns
        -------
        Any
            The computation result
        """
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
