"""Tests for ActHistGradientBoostingRegressor step."""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .step_test_case import StepTestCase
from iaml.actionables.predictors.regressor.act_hist_gradient_boosting_regressor import (
    ActHistGradientBoostingRegressor,
)


class TestActHistGradientBoostingRegressor(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        values = list(range(20))
        return pd.DataFrame(
            {
                "f1": values,
                "f2": [value % 2 for value in values],
                "f3": [value / 10 for value in values],
            }
        )

    def test_fit_sets_model_and_predicts(self) -> None:
        df = self._make_features()
        y = [value * 0.5 for value in range(len(df))]
        step = ActHistGradientBoostingRegressor()
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, HistGradientBoostingRegressor)

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [2.0 - value * 0.1 for value in range(len(df))]
        step = ActHistGradientBoostingRegressor()
        step.configure(
            {
                "l2_regularization": 0.5,
                "quantile": 0.6,
                "learning_rate": 0.2,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 2,
                "loss": "quantile",
                "n_iter_no_change": 3,
                "tol": 1e-3,
                "max_depth": 9,
            }
        )
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        self.assertAlmostEqual(step.model.l2_regularization, 0.5)
        self.assertAlmostEqual(step.model.quantile, 0.6)
        self.assertAlmostEqual(step.model.learning_rate, 0.2)
        self.assertEqual(step.model.max_leaf_nodes, 7)
        self.assertEqual(step.model.min_samples_leaf, 2)
        self.assertEqual(step.model.loss, "quantile")
        self.assertEqual(step.model.n_iter_no_change, 3)
        self.assertAlmostEqual(step.model.tol, 1e-3)
        self.assertEqual(step.model.max_depth, 9)

    def test_suitable_target_types(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActHistGradientBoostingRegressor()

        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4])
        categorical = self.make_dataset(df, y=["low", "high", "low", "high"])

        self.assertTrue(step.suitable(continuous))
        self.assertFalse(step.suitable(categorical))

    def test_quantile_search_bounds_support_real_fit(self) -> None:
        df = self._make_features()
        dataset = self.make_dataset(df, [value * 0.1 for value in range(len(df))])
        bounds = ActHistGradientBoostingRegressor().configuration['quantile']['range']
        for quantile in bounds:
            with self.subTest(quantile=quantile):
                step = ActHistGradientBoostingRegressor()
                step.configure({'loss': 'quantile', 'quantile': quantile})
                step.fit(dataset)
                self.assertEqual(len(step.predict(df)), len(df))
