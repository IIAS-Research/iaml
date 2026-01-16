"""Tests for ActCatBoostRegressor step."""
from unittest.mock import patch

import pandas as pd
from catboost import CatBoostError, CatBoostRegressor

from .step_test_case import StepTestCase
from iaml.actionables.predictors.regressor.act_catboost_regressor import (
    ActCatBoostRegressor,
)


class TestActCatBoostRegressor(StepTestCase):
    def _configure_fast(self, step: ActCatBoostRegressor) -> None:
        step.configure(
            {
                "iterations": 10,
                "learning_rate": 0.1,
                "depth": 2,
                "l2_leaf_reg": 1,
                "border_count": 32,
                "leaf_estimation_iterations": 1,
            }
        )

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
        y = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]
        step = ActCatBoostRegressor()
        self._configure_fast(step)
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, CatBoostRegressor)

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))

    def test_fit_wraps_catboost_error(self) -> None:
        df = self._make_features()
        y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        step = ActCatBoostRegressor()
        self._configure_fast(step)
        dataset = self.make_dataset(df, y)

        with patch(
            "iaml.actionables.predictors.regressor.act_catboost_regressor.CatBoostRegressor.fit",
            side_effect=CatBoostError("boom"),
        ):
            with self.assertRaises(ValueError) as ctx:
                self.fit_step(step, dataset)

        self.assertIn("CatBoostRegressor training failed", str(ctx.exception))

    def test_suitable_target_types(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActCatBoostRegressor()

        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4])
        categorical = self.make_dataset(df, y=["low", "high", "low", "high"])

        self.assertTrue(step.suitable(continuous))
        self.assertFalse(step.suitable(categorical))
