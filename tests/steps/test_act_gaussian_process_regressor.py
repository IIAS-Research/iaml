"""Tests for ActGaussianProcessRegressor step."""
import warnings

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import ConvergenceWarning

from .step_test_case import StepTestCase
from iaml.actionables.predictors.regressor.act_gaussian_process_regressor import (
    ActGaussianProcessRegressor,
)


class TestActGaussianProcessRegressor(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "f2": [1.0, 0.0, 1.5, 0.5, 1.0, 0.0],
                "f3": [0.2, 0.1, 0.4, 0.3, 0.6, 0.5],
            }
        )

    def _fit_without_warnings(
        self, step: ActGaussianProcessRegressor, dataset
    ) -> ActGaussianProcessRegressor:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            return self.fit_step(step, dataset)

    def test_fit_sets_model_and_predicts(self) -> None:
        df = self._make_features()
        y = [0.1, 0.2, 0.4, 0.7, 1.1, 1.6]
        step = ActGaussianProcessRegressor()
        dataset = self.make_dataset(df, y)

        result = self._fit_without_warnings(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, GaussianProcessRegressor)

        predictions = step.predict(df)
        predictions_array = np.asarray(predictions)
        self.assertEqual(predictions_array.shape, (len(df),))
        self.assertTrue(np.isfinite(predictions_array).all())

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [1.5, 1.4, 1.2, 0.9, 0.7, 0.6]
        step = ActGaussianProcessRegressor()
        step.configure("alpha", 1e-4)
        dataset = self.make_dataset(df, y)

        self._fit_without_warnings(step, dataset)

        self.assertAlmostEqual(step.model.alpha, 1e-4)

    def test_suitable_target_types(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActGaussianProcessRegressor()

        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4])
        categorical = self.make_dataset(df, y=["low", "high", "low", "high"])

        self.assertTrue(step.suitable(continuous))
        self.assertFalse(step.suitable(categorical))
