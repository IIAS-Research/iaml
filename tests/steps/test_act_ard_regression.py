"""Tests for ActARDRegression step."""
import warnings

import pandas as pd
from sklearn.linear_model import ARDRegression
from sklearn.exceptions import ConvergenceWarning

from .step_test_case import StepTestCase

from iaml.actionables.predictors.regressor.act_ard_regression import (
    ActARDRegression,
)


class TestActARDRegression(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [0, 1, 2, 3, 4, 5, 6, 7],
                "f2": [1, 0, 1, 0, 1, 0, 1, 0],
                "f3": [0.1, 0.3, 0.2, 0.5, 0.4, 0.7, 0.6, 0.8],
            }
        )

    def _fit_without_warnings(self, step: ActARDRegression, dataset) -> ActARDRegression:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            return self.fit_step(step, dataset)

    def test_fit_sets_model_and_predicts(self) -> None:
        df = self._make_features()
        y = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
        step = ActARDRegression()
        dataset = self.make_dataset(df, y)

        result = self._fit_without_warnings(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, ARDRegression)

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8]
        step = ActARDRegression()
        step.configure(
            {
                "alpha_1": 1e-5,
                "alpha_2": 2e-5,
                "lambda_1": 3e-6,
                "lambda_2": 4e-6,
                "threshold_lambda": 5000.0,
                "tol": 0.01,
            }
        )
        dataset = self.make_dataset(df, y)

        self._fit_without_warnings(step, dataset)

        self.assertAlmostEqual(step.model.alpha_1, 1e-5)
        self.assertAlmostEqual(step.model.alpha_2, 2e-5)
        self.assertAlmostEqual(step.model.lambda_1, 3e-6)
        self.assertAlmostEqual(step.model.lambda_2, 4e-6)
        self.assertAlmostEqual(step.model.threshold_lambda, 5000.0)
        self.assertAlmostEqual(step.model.tol, 0.01)

    def test_suitable_target_types(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActARDRegression()

        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4])
        categorical = self.make_dataset(df, y=["low", "high", "low", "high"])

        self.assertTrue(step.suitable(continuous))
        self.assertFalse(step.suitable(categorical))
