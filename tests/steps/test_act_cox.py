"""Tests for ActCox step."""
import numpy as np
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.predictors.survival.act_cox import ActCox


class TestActCox(StepTestCase):
    def _make_survival_dataset(self):
        X = pd.DataFrame(
            {
                "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 28, 33, 38, 43, 48, 53, 58, 63, 68, 73],
                "marker": [0.8, 1.1, 0.4, 1.5, 0.7, 1.2, 0.5, 1.6, 0.9, 1.3, 0.6, 1.0, 0.3, 1.4, 0.8, 1.1, 0.6, 1.5, 0.7, 1.2],
            }
        )
        events = [True, False, True, True, False, True, False, True, True, False, True, False, True, False, True, True, False, True, False, True]
        times = [6.0, 9.5, 5.5, 12.0, 8.2, 10.5, 7.1, 13.4, 9.0, 11.8, 6.7, 8.9, 5.9, 12.5, 7.8, 10.9, 8.4, 13.0, 9.6, 11.2]
        y = list(zip(events, times))
        return X, y

    def test_fit_trains_model_and_predicts(self) -> None:
        X, y = self._make_survival_dataset()
        dataset = self.make_dataset(X, y)
        step = ActCox()

        self.fit_step(step, dataset)

        self.assertIsNotNone(step.model)
        self.assertEqual(step.model.__class__.__name__, "CoxPHSurvivalAnalysis")
        preds = step.predict(X)
        self.assertEqual(len(preds), len(X))
        self.assertTrue(np.isfinite(np.asarray(preds)).all())

    def test_fit_passes_ties_configuration(self) -> None:
        X, y = self._make_survival_dataset()
        dataset = self.make_dataset(X, y)
        step = ActCox()
        step.configure("ties", "efron")

        self.fit_step(step, dataset)

        params = step.model.get_params()
        self.assertEqual(params.get("ties"), "efron")

    def test_suitable_detects_survival_target(self) -> None:
        X, y = self._make_survival_dataset()
        survival_dataset = self.make_dataset(X, y)
        non_survival_dataset = self.make_dataset(X, [0, 1] * (len(X) // 2))
        step = ActCox()

        self.assertTrue(step.suitable(survival_dataset))
        self.assertFalse(step.suitable(non_survival_dataset))
