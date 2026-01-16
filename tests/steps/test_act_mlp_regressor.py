"""Tests for ActMLPRegressor step."""
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

from .step_test_case import StepTestCase

from iaml.actionables.predictors.regressor.act_mlp_regressor import (
    ActMLPRegressor,
)


class TestActMLPRegressor(StepTestCase):
    def _make_features(self, n_samples: int = 30) -> pd.DataFrame:
        values = list(range(n_samples))
        return pd.DataFrame(
            {
                "f1": values,
                "f2": [value % 2 for value in values],
                "f3": [value / 10.0 for value in values],
            }
        )

    def _fit_without_warnings(self, step: ActMLPRegressor, dataset) -> ActMLPRegressor:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            return self.fit_step(step, dataset)

    def test_fit_sets_model_and_predicts(self) -> None:
        df = self._make_features()
        y = [value * 0.5 for value in range(len(df))]
        step = ActMLPRegressor()
        step.configure("node_per_layer", 16)
        dataset = self.make_dataset(df, y)

        result = self._fit_without_warnings(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, MLPRegressor)

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))

    def test_configuration_builds_hidden_layers(self) -> None:
        df = self._make_features()
        y = [value / 10.0 for value in range(len(df))]
        step = ActMLPRegressor()
        step.configure(
            {
                "hidden_layer_count": 3,
                "node_per_layer": 16,
                "activation": "tanh",
                "alpha": 0.05,
                "learning_rate_init": 0.01,
            }
        )
        dataset = self.make_dataset(df, y)

        self._fit_without_warnings(step, dataset)

        self.assertEqual(list(step.model.hidden_layer_sizes), [16, 16, 16])
        self.assertEqual(step.model.activation, "tanh")
        self.assertAlmostEqual(step.model.alpha, 0.05)
        self.assertAlmostEqual(step.model.learning_rate_init, 0.01)

    def test_suitable_target_types(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActMLPRegressor()

        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4])
        categorical = self.make_dataset(df, y=["low", "high", "low", "high"])

        self.assertTrue(step.suitable(continuous))
        self.assertFalse(step.suitable(categorical))
