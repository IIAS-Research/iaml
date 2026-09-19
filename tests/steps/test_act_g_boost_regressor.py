"""Tests for ActGBoostRegressor step."""
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .step_test_case import StepTestCase
from iaml.actionables.predictors.regressor.act_gboost_regressor import (
    ActGBoostRegressor,
)


class TestActGBoostRegressor(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [0, 1, 2, 3, 4, 5, 6, 7],
                "f2": [1, 0, 1, 0, 1, 0, 1, 0],
                "f3": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            }
        )

    def test_fit_sets_model_and_predicts(self) -> None:
        df = self._make_features()
        y = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
        step = ActGBoostRegressor()
        step.configure({"n_estimators": 5, "max_depth": 2, "random_state": 7})
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, GradientBoostingRegressor)

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2]
        step = ActGBoostRegressor()
        step.configure(
            {
                "learning_rate": 0.25,
                "loss": "huber",
                "criterion": "squared_error",
                "max_depth": 3,
                "min_samples_leaf": 2,
                "min_samples_split": 4,
                "max_features": 0.5,
                "n_estimators": 7,
                "random_state": 11,
            }
        )
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        self.assertAlmostEqual(step.model.learning_rate, 0.25)
        self.assertEqual(step.model.loss, "huber")
        self.assertEqual(step.model.criterion, "squared_error")
        self.assertEqual(step.model.max_depth, 3)
        self.assertEqual(step.model.min_samples_leaf, 2)
        self.assertEqual(step.model.min_samples_split, 4)
        self.assertAlmostEqual(step.model.max_features, 0.5)
        self.assertEqual(step.model.n_estimators, 7)
        self.assertEqual(step.model.random_state, 11)

    def test_suitable_target_types(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActGBoostRegressor()

        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4])
        categorical = self.make_dataset(df, y=["low", "high", "low", "high"])

        self.assertTrue(step.suitable(continuous))
        self.assertFalse(step.suitable(categorical))
