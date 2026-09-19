"""Tests for ActGradientBoostingSurvivalAnalysis step."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

_IAML_ROOT = SRC_PATH / "iaml"


def _ensure_package(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    if not hasattr(module, "__path__"):
        module.__path__ = [str(path)]


# Avoid importing iaml.__init__ which pulls in optional steps at import time.
_ensure_package("iaml", _IAML_ROOT)
_ensure_package("iaml.actionables", _IAML_ROOT / "actionables")
_ensure_package("iaml.actionables.predictors", _IAML_ROOT / "actionables" / "predictors")
_ensure_package(
    "iaml.actionables.predictors.survival",
    _IAML_ROOT / "actionables" / "predictors" / "survival",
)

missing_deps = []
if importlib.util.find_spec("sksurv") is None:
    missing_deps.append("sksurv")
if importlib.util.find_spec("shap") is None:
    missing_deps.append("shap")

SKIP_REASON = ""
if missing_deps:
    SKIP_REASON = "Missing optional dependencies: " + ", ".join(missing_deps)

if not SKIP_REASON:
    from .step_test_case import StepTestCase
    from iaml.actionables.predictors.survival.act_gradient_boosting_survival_analysis import (
        ActGradientBoostingSurvivalAnalysis,
    )
else:
    StepTestCase = unittest.TestCase


@unittest.skipIf(SKIP_REASON, SKIP_REASON)
class TestActGradientBoostingSurvivalAnalysis(StepTestCase):
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
        step = ActGradientBoostingSurvivalAnalysis()
        step.configure("n_estimators", 5)
        step.configure("learning_rate", 0.2)
        self.assertEqual(step.get_config("n_estimators"), 5)
        self.assertEqual(step.get_config("learning_rate"), 0.2)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertEqual(step.model.__class__.__name__, "GradientBoostingSurvivalAnalysis")
        predictions = step.predict(X)
        predictions_array = np.asarray(predictions, dtype=float)
        self.assertEqual(predictions_array.shape[0], len(X))
        self.assertTrue(np.isfinite(predictions_array).all())

    def test_fit_rejects_non_survival_targets(self) -> None:
        X, _ = self._make_survival_dataset()
        dataset = self.make_dataset(X, [0, 1] * (len(X) // 2))
        step = ActGradientBoostingSurvivalAnalysis()

        with self.assertRaises((TypeError, ValueError)):
            step.fit(dataset)

    def test_suitable_detects_survival_target(self) -> None:
        X, y = self._make_survival_dataset()
        survival_dataset = self.make_dataset(X, y)
        non_survival_dataset = self.make_dataset(X, [0, 1] * (len(X) // 2))
        step = ActGradientBoostingSurvivalAnalysis()

        self.assertTrue(step.suitable(survival_dataset))
        self.assertFalse(step.suitable(non_survival_dataset))
