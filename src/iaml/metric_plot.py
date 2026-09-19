"""[PLOT] Parent for all metrics plots."""
from __future__ import annotations
from typing import TYPE_CHECKING, Type

import io
import pandas as pd

from yellowbrick.base import Visualizer

from .plot import capture, Plot

if TYPE_CHECKING:
    from .iaml_pipeline import IAMLPipeline


class MetricPlot(Plot):
    """To be used by performance Explainer"""

    def compute(
        self,
        estimator: IAMLPipeline,
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


def _estimator_is_fitted(estimator) -> bool:
    checker = getattr(estimator, "__sklearn_is_fitted__", None)
    if callable(checker):
        try:
            fitted = bool(checker())
        except Exception:  # noqa: BLE001
            return False
        if fitted and hasattr(estimator, "predictor"):
            predictor = getattr(estimator, "predictor", None)
            if predictor and isinstance(predictor, tuple):
                model = getattr(predictor[1], "model", None)
                if model is None:
                    return False
        return fitted
    return False


def yellowbrick_plot(yellowbrick_visualizer: Type[Visualizer]):
    """Decorator for creating metric plots based on Yellowbrick visualizers.

    :param Type[Visualizer] yellowbrick_visualizer: Base Yellowbrick visualizer.
    """
    def decorator(cls):
        @capture
        def compute(self, estimator, X, y, X_train = None, y_train = None, **kwargs): # pylint: disable=missing-function-docstring, unused-argument
            self._binary_image = io.BytesIO() # pylint: disable=protected-access
            already_fitted = _estimator_is_fitted(estimator)
            visualizer = yellowbrick_visualizer(estimator, is_fitted=already_fitted)

            if X_train is not None and y_train is not None and not already_fitted:
                visualizer.fit(X_train, y_train)

            visualizer.score(X, y)
            visualizer.poof(self._binary_image) # pylint: disable=protected-access

            return self

        cls.compute = compute

        return cls

    return decorator
