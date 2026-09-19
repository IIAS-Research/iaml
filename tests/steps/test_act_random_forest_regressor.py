"""Tests for ActRandomForestRegressor step."""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .step_test_case import StepTestCase
from iaml.actionables.predictors.regressor.act_randomforest_regressor import (
    ActRandomForestRegressor,
)


class TestActRandomForestRegressor(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [0, 1, 2, 3, 4, 5, 6, 7],
                "f2": [1, 0, 1, 0, 1, 0, 1, 0],
                "f3": [0.2, 0.4, 0.1, 0.6, 0.5, 0.8, 0.7, 0.9],
            }
        )

    def test_fit_sets_model_and_predicts(self) -> None:
        df = self._make_features()
        y = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]
        step = ActRandomForestRegressor()
        step.configure("n_estimators", 5)
        step.configure("random_state", 7)
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, RandomForestRegressor)

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2]
        step = ActRandomForestRegressor()
        step.configure(
            {
                "n_estimators": 7,
                "min_samples_leaf": 2,
                "min_samples_split": 3,
                "max_features": 0.5,
                "bootstrap": True,
                "criterion": "absolute_error",
                "random_state": 13,
            }
        )
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        self.assertEqual(step.model.n_estimators, 7)
        self.assertEqual(step.model.min_samples_leaf, 2)
        self.assertEqual(step.model.min_samples_split, 3)
        self.assertAlmostEqual(step.model.max_features, 0.5)
        self.assertTrue(step.model.bootstrap)
        self.assertEqual(step.model.criterion, "absolute_error")
        self.assertEqual(step.model.random_state, 13)

    def test_suitable_target_types(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActRandomForestRegressor()

        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4])
        categorical = self.make_dataset(df, y=["low", "high", "low", "high"])

        self.assertTrue(step.suitable(continuous))
        self.assertFalse(step.suitable(categorical))
