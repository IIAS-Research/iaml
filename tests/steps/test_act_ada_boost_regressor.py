"""Tests for ActAdaBoostRegressor step."""
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor

from .step_test_case import StepTestCase

from iaml.actionables.predictors.regressor.act_ada_boost_regressor import (
    ActAdaBoostRegressor,
)


class TestActAdaBoostRegressor(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [0, 1, 2, 3, 4, 5, 6, 7],
                "f2": [1, 0, 1, 0, 1, 0, 1, 0],
                "f3": [0.1, 0.3, 0.2, 0.5, 0.4, 0.7, 0.6, 0.8],
            }
        )

    def test_fit_sets_model_and_predicts(self) -> None:
        df = self._make_features()
        y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        step = ActAdaBoostRegressor()
        step.configure({"n_estimators": 5, "random_state": 7})
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, AdaBoostRegressor)

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        step = ActAdaBoostRegressor()
        step.configure(
            {
                "learning_rate": 0.5,
                "loss": "square",
                "n_estimators": 3,
                "random_state": 9,
            }
        )
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        self.assertEqual(step.model.learning_rate, 0.5)
        self.assertEqual(step.model.loss, "square")
        self.assertEqual(step.model.n_estimators, 3)
        self.assertEqual(step.model.random_state, 9)

    def test_suitable_target_types(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActAdaBoostRegressor()

        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4])
        binary = self.make_dataset(df, y=["yes", "no", "yes", "no"])

        self.assertTrue(step.suitable(continuous))
        self.assertFalse(step.suitable(binary))
