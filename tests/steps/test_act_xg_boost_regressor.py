"""Tests for ActXGBoostRegressor step."""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

from .step_test_case import StepTestCase
from iaml.actionables.predictors.regressor.act_xgboost_regressor import (
    ActXGBoostRegressor,
)
from iaml.actionables.predictors.regressor.act_gboost_regressor import ActGBoostRegressor


class TestActXGBoostRegressor(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [0, 1, 2, 3, 4, 5, 6, 7],
                "f2": [1, 0, 1, 0, 1, 0, 1, 0],
                "f3": [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
            }
        )

    def test_fit_sets_model_and_predicts(self) -> None:
        df = self._make_features()
        y = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
        step = ActXGBoostRegressor()
        step.configure({"n_estimators": 5, "max_depth": 2, "random_state": 7})
        dataset = self.make_dataset(df, y)

        result = step.fit(dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, XGBRegressor)
        self.assertEqual(step.model.get_booster().num_boosted_rounds(), 5)

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))
        self.assertTrue(np.isfinite(predictions).all())

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2]
        step = ActXGBoostRegressor()
        parameters = {
            "learning_rate": 0.25,
            "max_depth": 3,
            "min_child_weight": 2.0,
            "subsample": 0.75,
            "colsample_bytree": 0.5,
            "n_estimators": 7,
            "random_state": 11,
        }
        step.configure(parameters)
        dataset = self.make_dataset(df, y)

        step.fit(dataset)

        actual = step.model.get_params()
        for name, value in parameters.items():
            with self.subTest(parameter=name):
                self.assertEqual(actual[name], value)
        self.assertEqual(actual["objective"], "reg:squarederror")
        self.assertEqual(actual["tree_method"], "hist")
        self.assertEqual(actual["n_jobs"], 1)
        self.assertEqual(step.model.get_booster().num_boosted_rounds(), 7)

    def test_default_configuration_uses_xgboost(self) -> None:
        df = self._make_features()
        dataset = self.make_dataset(df, [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
        step = ActXGBoostRegressor()

        step.fit(dataset)

        self.assertIsInstance(step.model, XGBRegressor)
        self.assertEqual(step.model.get_booster().num_boosted_rounds(), 100)
        self.assertTrue(np.isfinite(step.predict(df)).all())

    def test_gboost_remains_a_distinct_backend(self) -> None:
        df = self._make_features()
        dataset = self.make_dataset(df, [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
        xgboost = ActXGBoostRegressor()
        gradient_boosting = ActGBoostRegressor()
        for step in (xgboost, gradient_boosting):
            step.configure({"n_estimators": 5, "max_depth": 2})
            step.fit(dataset)

        self.assertIsInstance(xgboost.model, XGBRegressor)
        self.assertIsInstance(gradient_boosting.model, GradientBoostingRegressor)
        self.assertFalse(np.allclose(xgboost.predict(df), gradient_boosting.predict(df)))

    def test_suitable_target_types(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActXGBoostRegressor()

        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4])
        categorical = self.make_dataset(df, y=["low", "high", "low", "high"])

        self.assertTrue(step.suitable(continuous))
        self.assertFalse(step.suitable(categorical))
