"""Prediction-aware storage shared by the survival forest steps."""

import numpy as np
from sklearn.utils._param_validation import StrOptions


def survival_forest_constraints(estimator):
    """Let recipes validate the adapter's automatic mode before backend fitting."""
    return {
        **estimator._parameter_constraints,
        "low_memory": ["boolean", StrOptions({"auto"})],
    }


class SurvivalForestMemoryMixin:
    """Keep curves unless a known prediction contract only requires risk scores."""

    def configure_prediction_requirements(self, required_predictions):
        """Set the prediction methods that must remain available after fitting.

        ``None`` leaves the output contract unspecified and preserves curves in
        automatic mode. An explicit risk-only contract enables scalar storage.
        """
        if isinstance(required_predictions, str):
            required_predictions = (required_predictions,)
        self._required_predictions = (
            None if required_predictions is None else frozenset(required_predictions)
        )
        return self

    def _survival_forest_parameters(self):
        """Resolve the storage policy without changing the configured value."""
        parameters = self.passthrough_parameters()
        low_memory = parameters["low_memory"]
        required_predictions = getattr(self, "_required_predictions", None)
        risk_only = bool(required_predictions) and required_predictions <= {"predict"}

        if isinstance(low_memory, str) and low_memory == "auto":
            parameters["low_memory"] = risk_only
        elif not isinstance(low_memory, (bool, np.bool_)):
            raise ValueError("low_memory must be 'auto', True, or False")
        elif low_memory and required_predictions and not risk_only:
            methods = ", ".join(sorted(required_predictions - {"predict"}))
            raise ValueError(
                f"low_memory=True disables required prediction methods: {methods}. "
                "Use low_memory='auto' or False to preserve survival curves."
            )
        else:
            parameters["low_memory"] = bool(low_memory)
        return parameters

    def _fit_survival_forest(self, estimator_class, dataset):
        """Fit once with storage matching the requested prediction methods."""
        parameters = self._survival_forest_parameters()
        X, y = dataset.to_survival()
        self.model = estimator_class(**parameters)
        self.model.fit(X, y)
        return self

    def _require_survival_curves(self, method):
        if self.model is not None and self.model.low_memory:
            raise ValueError(
                f"{method} is unavailable because this forest was fitted with risk-only storage. "
                "Configure low_memory=False and refit to preserve survival curves."
            )

    def predict_survival_function(self, X):
        """Predict survival curves when the fitting contract preserved them."""
        self._require_survival_curves("predict_survival_function")
        return super().predict_survival_function(X)

    def predict_cumulative_hazard_function(self, X):
        """Predict cumulative hazard curves when the fitting contract preserved them."""
        self._require_survival_curves("predict_cumulative_hazard_function")
        return super().predict_cumulative_hazard_function(X)
