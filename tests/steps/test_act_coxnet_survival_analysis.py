"""Tests for ActCoxnetSurvivalAnalysis step."""
import unittest

import numpy as np
import pandas as pd

try:
    import sksurv.linear_model  # noqa: F401
    SKSURV_AVAILABLE = True
except Exception:
    SKSURV_AVAILABLE = False

IMPORT_ERROR = None
ActCoxnetSurvivalAnalysis = None
StepTestCase = unittest.TestCase

if SKSURV_AVAILABLE:
    try:
        from .step_test_case import StepTestCase
        from iaml.actionables.predictors.survival.act_coxnet_survival_analysis import (
            ActCoxnetSurvivalAnalysis,
        )
    except Exception as exc:
        IMPORT_ERROR = exc


@unittest.skipIf(IMPORT_ERROR is not None, f"iaml import failed: {IMPORT_ERROR}")
@unittest.skipUnless(SKSURV_AVAILABLE, "sksurv not available")
class TestActCoxnetSurvivalAnalysis(StepTestCase):
    def _make_survival_dataset(self):
        X = pd.DataFrame(
            {
                "age": [25, 30, 35, 40, 45, 50, 55, 60, 29, 34, 39, 44],
                "bmi": [22.1, 24.0, 26.5, 23.2, 25.1, 27.3, 24.8, 28.0, 21.9, 23.7, 25.9, 24.5],
                "group": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            }
        )
        events = [True, False, True, True, False, True, False, True, True, False, True, False]
        times = [5.0, 6.5, 4.8, 7.2, 6.0, 8.1, 5.5, 9.0, 5.8, 6.9, 7.5, 6.3]
        y = list(zip(events, times))
        return X, y

    def test_fit_predicts_and_uses_numeric_columns(self) -> None:
        X, y = self._make_survival_dataset()
        dataset = self.make_dataset(X, y)
        step = ActCoxnetSurvivalAnalysis()
        step.configure("n_alphas", 10)
        step.configure("max_iter", 200)
        step.configure("l1_ratio", 0.3)

        self.fit_step(step, dataset)

        self.assertIsNotNone(step.model)
        self.assertEqual(step.model.__class__.__name__, "CoxnetSurvivalAnalysis")
        self.assertCountEqual(step.columns, ["age", "bmi"])

        params = step.model.get_params()
        self.assertEqual(params.get("n_alphas"), 10)
        self.assertEqual(params.get("max_iter"), 200)
        self.assertEqual(params.get("l1_ratio"), 0.3)

        preds = step.predict(X)
        self.assertEqual(len(preds), len(X))
        self.assertTrue(np.isfinite(np.asarray(preds)).all())

    def test_suitable_requires_survival_and_numeric(self) -> None:
        X, y = self._make_survival_dataset()
        dataset_survival = self.make_dataset(X, y)
        dataset_non_survival = self.make_dataset(X, [0, 1] * (len(X) // 2))
        X_non_numeric = pd.DataFrame({"group": ["A", "B"] * (len(X) // 2)})
        dataset_non_numeric = self.make_dataset(X_non_numeric, y)
        step = ActCoxnetSurvivalAnalysis()

        self.assertTrue(step.suitable(dataset_survival))
        self.assertFalse(step.suitable(dataset_non_survival))
        self.assertFalse(step.suitable(dataset_non_numeric))
