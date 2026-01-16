"""Tests for ActSGDRegressor step."""
import math
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDRegressor

from .step_test_case import StepTestCase

from iaml.actionables.predictors.regressor.act_sgd_regressor import (
    ActSGDRegressor,
)


class TestActSGDRegressor(StepTestCase):
    def _make_features(self, n_samples: int = 20) -> pd.DataFrame:
        values = list(range(n_samples))
        return pd.DataFrame(
            {
                "f1": values,
                "f2": [value % 3 for value in values],
                "f3": [value / 10.0 for value in values],
            }
        )

    def _fit_without_warnings(self, step: ActSGDRegressor, dataset) -> ActSGDRegressor:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            return self.fit_step(step, dataset)

    def test_fit_sets_model_and_predicts(self) -> None:
        df = self._make_features()
        y = [value * 0.5 for value in range(len(df))]
        step = ActSGDRegressor()
        dataset = self.make_dataset(df, y)

        result = self._fit_without_warnings(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, SGDRegressor)

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))
        self.assertEqual(len(step.model.coef_), df.shape[1])
        self.assertTrue(all(math.isfinite(float(pred)) for pred in predictions))

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [value / 10.0 for value in range(len(df))]
        step = ActSGDRegressor()
        step.configure(
            {
                "alpha": 0.001,
                "tol": 0.001,
                "loss": "huber",
                "penalty": "elasticnet",
                "l1_ratio": 0.4,
                "average": True,
            }
        )
        dataset = self.make_dataset(df, y)

        self._fit_without_warnings(step, dataset)

        self.assertAlmostEqual(step.model.alpha, 0.001)
        self.assertAlmostEqual(step.model.tol, 0.001)
        self.assertEqual(step.model.loss, "huber")
        self.assertEqual(step.model.penalty, "elasticnet")
        self.assertAlmostEqual(step.model.l1_ratio, 0.4)
        self.assertTrue(step.model.average)

    def test_suitable_target_types(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActSGDRegressor()

        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4])
        categorical = self.make_dataset(df, y=["low", "high", "low", "high"])

        self.assertTrue(step.suitable(continuous))
        self.assertFalse(step.suitable(categorical))
