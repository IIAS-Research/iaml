"""Tests for ActLinearRegression step."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .step_test_case import StepTestCase
from iaml.actionables.predictors.regressor.act_linear_regression import (
    ActLinearRegression,
)


class TestActLinearRegression(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [0.0, 1.0, 2.0, 3.0],
                "f2": [1.5, 0.5, 2.0, 3.5],
            }
        )

    def test_fit_sets_model_and_predicts(self) -> None:
        df = self._make_features()
        y = [2.0 + 1.5 * f1 + 0.5 * f2 for f1, f2 in zip(df["f1"], df["f2"])]
        step = ActLinearRegression()
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, LinearRegression)

        predictions = step.model.predict(df)
        self.assertEqual(predictions.shape, (len(df),))
        self.assertTrue(np.allclose(predictions, y))

    def test_suitable_target_types(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActLinearRegression()

        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4])
        categorical = self.make_dataset(df, y=["low", "high", "low", "high"])

        self.assertTrue(step.suitable(continuous))
        self.assertFalse(step.suitable(categorical))

    def test_priorize_returns_neutral_score(self) -> None:
        step = ActLinearRegression()

        self.assertEqual(step.priorize(), 0.5)
