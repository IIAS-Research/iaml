"""Tests for ActComponentwiseGradientBoostingSurvivalAnalysis step."""
import numpy as np
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.predictors.survival.act_survival_component_wise_gboost import (
    ActComponentwiseGradientBoostingSurvivalAnalysis,
)


class TestActComponentwiseGradientBoostingSurvivalAnalysis(StepTestCase):
    def _make_survival_dataset(self) -> tuple[pd.DataFrame, list[tuple[bool, float]]]:
        X = pd.DataFrame(
            {
                "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 28, 33],
                "marker": [0.8, 1.1, 0.4, 1.5, 0.7, 1.2, 0.5, 1.6, 0.9, 1.3, 0.6, 1.0],
            }
        )
        events = [True, False, True, True, False, True, False, True, True, False, True, False]
        times = [6.0, 9.5, 5.5, 12.0, 8.2, 10.5, 7.1, 13.4, 9.0, 11.8, 6.7, 8.9]
        y = list(zip(events, times))
        return X, y

    def test_fit_sets_model_and_predicts(self) -> None:
        X, y = self._make_survival_dataset()
        dataset = self.make_dataset(X, y)
        step = ActComponentwiseGradientBoostingSurvivalAnalysis()
        step.configure("n_estimators", 5)
        step.configure("learning_rate", 0.2)
        self.assertEqual(step.get_config("n_estimators"), 5)
        self.assertEqual(step.get_config("learning_rate"), 0.2)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertEqual(
            step.model.__class__.__name__,
            "ComponentwiseGradientBoostingSurvivalAnalysis",
        )
        predictions = step.predict(X)
        predictions_array = np.asarray(predictions, dtype=float)
        self.assertEqual(predictions_array.shape[0], len(X))
        if predictions_array.ndim > 1:
            self.assertEqual(predictions_array.shape[1], 1)
        self.assertTrue(np.isfinite(predictions_array).all())

    def test_fit_rejects_non_survival_targets(self) -> None:
        X, _ = self._make_survival_dataset()
        dataset = self.make_dataset(X, [0, 1] * (len(X) // 2))
        step = ActComponentwiseGradientBoostingSurvivalAnalysis()

        with self.assertRaises((TypeError, ValueError)):
            step.fit(dataset)

    def test_suitable_detects_survival_target(self) -> None:
        X, y = self._make_survival_dataset()
        survival_dataset = self.make_dataset(X, y)
        non_survival_dataset = self.make_dataset(X, [0, 1] * (len(X) // 2))
        step = ActComponentwiseGradientBoostingSurvivalAnalysis()

        self.assertTrue(step.suitable(survival_dataset))
        self.assertFalse(step.suitable(non_survival_dataset))
