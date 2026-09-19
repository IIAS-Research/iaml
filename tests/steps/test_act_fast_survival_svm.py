"""Tests for ActFastSurvivalSVM step."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IAML_PATH = PROJECT_ROOT / "src" / "iaml"


def _ensure_iaml_package() -> None:
    if "iaml" in sys.modules:
        return
    package = types.ModuleType("iaml")
    package.__path__ = [str(IAML_PATH)]
    package.__package__ = "iaml"
    sys.modules["iaml"] = package


# Avoid importing iaml/__init__.py during test discovery.
_ensure_iaml_package()

from .step_test_case import StepTestCase

try:
    import sksurv.svm  # noqa: F401
    SKSURV_AVAILABLE = True
except Exception:
    SKSURV_AVAILABLE = False

if SKSURV_AVAILABLE:
    def _load_act_fast_survival_svm():
        module_name = "iaml.actionables.predictors.survival.act_fast_survival_svm"
        module_path = IAML_PATH / "actionables" / "predictors" / "survival" / "act_fast_survival_svm.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError("Unable to load ActFastSurvivalSVM module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.ActFastSurvivalSVM

    ActFastSurvivalSVM = _load_act_fast_survival_svm()
else:
    ActFastSurvivalSVM = None


@unittest.skipUnless(SKSURV_AVAILABLE, "sksurv not available")
class TestActFastSurvivalSVM(StepTestCase):
    def _make_survival_dataset(self) -> tuple[pd.DataFrame, list[tuple[bool, float]]]:
        X = pd.DataFrame(
            {
                "age": [25, 30, 35, 40, 45, 50, 55, 60, 29, 34, 39, 44],
                "marker": [0.8, 1.1, 0.4, 1.5, 0.7, 1.2, 0.5, 1.6, 0.9, 1.3, 0.6, 1.0],
                "group": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            }
        )
        events = [True, False, True, True, False, True, False, True, True, False, True, False]
        times = [6.0, 9.5, 5.5, 12.0, 8.2, 10.5, 7.1, 13.4, 9.0, 11.8, 6.7, 8.9]
        y = list(zip(events, times))
        return X, y

    def test_fit_predicts_and_uses_numeric_columns(self) -> None:
        X, y = self._make_survival_dataset()
        dataset = self.make_dataset(X, y)
        step = ActFastSurvivalSVM()
        step.configure("alpha", 0.5)
        step.configure("max_iter", 100)
        step.configure("rank_ratio", 0.5)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertEqual(step.model.__class__.__name__, "FastSurvivalSVM")
        self.assertCountEqual(step.columns, ["age", "marker"])

        params = step.model.get_params()
        self.assertEqual(params.get("alpha"), 0.5)
        self.assertEqual(params.get("max_iter"), 100)

        predictions = step.predict(X)
        predictions_array = np.asarray(predictions, dtype=float)
        self.assertEqual(predictions_array.shape, (len(X),))
        self.assertTrue(np.isfinite(predictions_array).all())

    def test_suitable_requires_survival_and_numeric(self) -> None:
        X, y = self._make_survival_dataset()
        dataset_survival = self.make_dataset(X, y)
        dataset_non_survival = self.make_dataset(X, [0, 1] * (len(X) // 2))
        X_non_numeric = pd.DataFrame({"group": ["A", "B"] * (len(X) // 2)})
        dataset_non_numeric = self.make_dataset(X_non_numeric, y)
        step = ActFastSurvivalSVM()

        self.assertTrue(step.suitable(dataset_survival))
        self.assertFalse(step.suitable(dataset_non_survival))
        self.assertFalse(step.suitable(dataset_non_numeric))
