"""
[PLOT] Parent of all others Plot, implement the default behavior
"""
import base64
from functools import wraps
from typing import Any, TYPE_CHECKING
import matplotlib.pyplot as plt
import pandas as pd
if TYPE_CHECKING:
    from .iaml_pipeline import IAMLPipeline
    import io


class Plot:
    """[PLOT] Parent of all others Plot, implement the default behavior"""
    
    title: str = "Here is the plot title"
    description: str = "Here is an explanation of how this plot work"
    
    def __init__(self):
        self.__visualizer: MetricPlot = None  # pylint: disable=unused-private-member
        """The class object holding data to plot"""

        self._binary_image: io.BytesIO = None
        """The generated plot image"""
    
    @property
    def image(self) -> bytes:
        """Get binary representation of the plot image

        :raise AttributeError: Plot must be computed before.
        :return: The image bytes.
        """
        if self._binary_image is not None:
            return self._binary_image.getvalue()
        
        raise AttributeError("Plot must be computed before")
    
    @property
    def b64_image(self) -> str:
        """Get b64 representation of the plot image
        
        :return: b64 string image.
        """
        return base64.b64encode(self.image).decode()
 
    def _compute(
        self,
        estimator: 'IAMLPipeline',
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs) -> None:
        """Compute plot given X, y. 
        Must be overloaded by children classes
        
        :param IAMLPipeline estimator: The pipeline we compute the plot on.
        :param pd.DataFrame X: The dataset we wanna compute plot on.
        :param pd.Series y: The dataset target we wanna compute plot on.
        :param optional \\**kwargs: Additional parameters for plotting.
        """
        raise NotImplementedError('Subclass must implement abstract method')

    @classmethod
    def suitable(cls, type_of_target: str) -> bool:  # pylint: disable=unused-argument
        """Does this plot is usable for a given type_of_target ?
        
        :param str type_of_target: The type of target we wanna predict.
        :return: Suitable ?
        """
        return False

    def to_json(self, data_format: str='binary') -> dict[str, Any]:
        """Convert plot into json with name, description and b64 image
        
        :param str data_format: Type of data we want. Can be 'binary' or 'b64'.
        :raise AttributeError: Invalid data format.
        :return: A dictonnary containing plot informations and data.
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
        """Return plot as markdown format
        
        :return: Markdown formatted plot.
        """
        base64_md = f"![{self.title}](data:image/png;base64,{self.b64_image})"
        return "\n\n".join([f"# {self.title}", self.description, base64_md])


class MetricPlot(Plot):
    """To be used by performance Explainer"""

    def __init__(
        self,
        estimator:'IAMLPipeline',
        X: pd.DataFrame,
        y: pd.Series, 
        X_train: pd.DataFrame = None,
        y_train: pd.Series = None, 
        **kwargs) -> None:
        """Initialize a metric plot
        
        :param IAMLPipeline estimator: The pipeline we'll compute the plot on.
        :param pd.DataFrame X: The dataframe we wanna compute our plot on.
        :param pd.Series y: The dataframe target we wanna compute our plot on.
        :param pd.DataFrame, optional X_train: The dataframe used for training.
        :param pd.Series, optional y_train: The dataframe target used for training.
        :param optional \\**kwargs: Additional arguments for plot computing.
        """
        self._compute(estimator, X, y, X_train=X_train, y_train=y_train, **kwargs)

    def _compute(
        self,
        estimator: 'IAMLPipeline',
        X: pd.DataFrame,
        y: pd.Series,
        X_train: pd.DataFrame = None,
        y_train: pd.Series = None,
        **kwargs) -> 'MetricPlot':
        """Compute plot given X, y. 
        Must be overridden by children classes
        
        :param IAMLPipeline estimator: The pipeline we compute the plot on.
        :param pd.DataFrame X: The dataset we wanna compute plot on.
        :param pd.Series y: The dataset target we wanna compute plot on.
        :param pd.DataFrame, optional X_train: The dataframe used for training.
        :param pd.Series, optional y_train: The dataframe target used for training.
        :param optional \\**kwargs: Additional parameters for plotting.
        :return: A MetricPlot object.
        """
        raise NotImplementedError('Subclass must implement abstract method')


def capture(func) -> Any:
    """A decorator to capture a matplotlib plot into a BytesIO object and return it
    as binary data instead of showing it.
    
    :return: binary plot data.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        plt.close()
        return ret

    return wrapper
