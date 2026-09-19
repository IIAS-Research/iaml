"""A mandatory sklearn input transformer, fitted inside each candidate pipeline."""
from __future__ import annotations

from typing import Any

from joblib import hash as joblib_hash
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from .dataset import Dataset
from .step import Step


class SklearnPreprocessor(Step):
    """Adapt an unsupervised sklearn transformer to IAML's fit/transform protocol.

    The caller's transformer is a template only. A fresh sklearn clone is fitted
    on each supplied Dataset.X, without target values or patient-group columns.
    ``transformer_`` is the fitted clone, retained for prediction/provenance and
    serialization. The template and its hyperparameters participate in cache
    keys; the learned vocabulary does not alter the pipeline configuration.

    A DataFrame output is required so column names and row alignment remain
    explicit. This step is deliberately not registered under a search tag: it
    can only enter the search through IAML's explicit initial_preprocessor.
    """

    name = "Initial sklearn preprocessing"
    can_be_disabled = False

    def __init__(self, transformer: Any):
        super().__init__()
        if not callable(getattr(transformer, "fit", None)) or not callable(getattr(transformer, "transform", None)):
            raise TypeError("initial_preprocessor must implement sklearn fit and transform")
        self.transformer = clone(transformer)
        self.configuration = {
            "transformer_class": {"default": f"{type(transformer).__module__}.{type(transformer).__qualname__}"},
            "transformer_parameters_hash": {"default": joblib_hash(self.transformer.get_params(deep=True))},
        }
        self.default_configuration()
        self.is_interchangeable = False
        self.optimizable = False

    def fit(self, dataset: Dataset) -> "SklearnPreprocessor":
        # Never retain categories learned by the generation sample or an earlier
        # fold. sklearn.clone also removes fitted state supplied by the caller.
        self.__dict__.pop("transformer_", None)
        fitted = clone(self.transformer)
        fitted.fit(dataset.X.copy(deep=True))
        self.transformer_ = fitted
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "transformer_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("IAML preprocessing requires a DataFrame input")
        result = self.transformer_.transform(X.copy(deep=True))
        if not isinstance(result, pd.DataFrame):
            raise TypeError("initial_preprocessor must return a numeric DataFrame")
        if len(result) != len(X) or not result.index.equals(X.index):
            raise ValueError("initial_preprocessor must preserve row count, order and index")
        if not result.columns.is_unique or not len(result.columns):
            raise ValueError("initial_preprocessor must return unique, nonempty feature columns")
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in result.dtypes):
            raise TypeError("initial_preprocessor output columns must all be numeric")
        # IAML regards bool columns as categorical. Keep the explicit numeric
        # representation, including BooleanDtype missing values, unambiguous.
        boolean_columns = result.select_dtypes(include=["bool", "boolean"]).columns
        if len(boolean_columns):
            result = result.copy()
            result[boolean_columns] = result[boolean_columns].astype(np.float32)
        return result
