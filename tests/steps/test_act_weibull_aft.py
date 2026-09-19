"""Tests for ActWeibullAFT step."""
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

try:
    from sksurv.linear_model import WeibullAFT as _SKWeibullAFT  # noqa: F401
    SKSURV_AVAILABLE = True
except Exception:
    try:
        from sksurv.parametric import WeibullAFT as _SKWeibullAFT  # noqa: F401
        SKSURV_AVAILABLE = True
    except Exception:
        SKSURV_AVAILABLE = False

try:
    from lifelines import WeibullAFTFitter as _LLWeibullAFTFitter  # noqa: F401
    LIFELINES_AVAILABLE = True
except Exception:
    LIFELINES_AVAILABLE = False

BACKEND_AVAILABLE = SKSURV_AVAILABLE or LIFELINES_AVAILABLE


def _ensure_package(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    elif not hasattr(module, "__path__"):
        module.__path__ = [str(path)]


def _bootstrap_iaml() -> None:
    project_root = Path(__file__).resolve().parents[2]
    iaml_root = project_root / "src" / "iaml"
    _ensure_package("iaml", iaml_root)
    _ensure_package("iaml.actionables", iaml_root / "actionables")
    _ensure_package("iaml.actionables.predictors", iaml_root / "actionables" / "predictors")
    _ensure_package(
        "iaml.actionables.predictors.survival",
        iaml_root / "actionables" / "predictors" / "survival",
    )


_bootstrap_iaml()

IMPORT_ERROR = None
ActWeibullAFT = None
StepTestCase = unittest.TestCase

try:
    from .step_test_case import StepTestCase
    from iaml.actionables.predictors.survival.act_weibull_aft import ActWeibullAFT
except Exception as exc:
    IMPORT_ERROR = exc

SKIP_REASON = "ActWeibullAFT import failed"
if IMPORT_ERROR is not None:
    SKIP_REASON = f"ActWeibullAFT import failed: {IMPORT_ERROR}"


def _make_survival_dataset() -> tuple[pd.DataFrame, list[tuple[bool, float]]]:
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


@unittest.skipIf(ActWeibullAFT is None, SKIP_REASON)
class TestActWeibullAFTFallback(StepTestCase):
    def test_fit_predict_score_with_stub_backend(self) -> None:
        class DummyLifelinesModel:
            def __init__(self, **kwargs):
                self.fit_X = None
                self.fit_duration_col = None
                self.fit_event_col = None
                self.score_X = None
                self.predict_X = None
                self.params = kwargs

            def fit(self, X, duration_col=None, event_col=None):
                self.fit_X = X.copy()
                self.fit_duration_col = duration_col
                self.fit_event_col = event_col
                return self

            def predict_median(self, X):
                self.predict_X = X.copy()
                return np.arange(len(X), dtype=float)

            def score(self, X, *args, **kwargs):
                self.score_X = X.copy()
                return 0.25

        X, y = _make_survival_dataset()
        dataset = self.make_dataset(X, y)
        step = ActWeibullAFT()

        with mock.patch.object(
            ActWeibullAFT, "_resolve_backend", return_value=("lifelines", DummyLifelinesModel)
        ):
            result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertEqual(step.backend, "lifelines")
        self.assertIsInstance(step.model, DummyLifelinesModel)
        self.assertIsNotNone(step._lifelines_time_col)
        self.assertIsNotNone(step._lifelines_event_col)
        self.assertNotIn("group", step.model.fit_X.columns)
        self.assertIn(step._lifelines_time_col, step.model.fit_X.columns)
        self.assertIn(step._lifelines_event_col, step.model.fit_X.columns)

        preds = step.predict(X)
        preds_array = np.asarray(preds, dtype=float)
        self.assertEqual(preds_array.shape, (len(X),))
        self.assertTrue(np.isfinite(preds_array).all())
        self.assertIsNotNone(step.model.predict_X)
        self.assertNotIn("group", step.model.predict_X.columns)

        score = step.score(X, y)
        self.assertEqual(score, 0.25)
        self.assertIn(step._lifelines_time_col, step.model.score_X.columns)
        self.assertIn(step._lifelines_event_col, step.model.score_X.columns)

    def test_fit_raises_import_error_without_backend(self) -> None:
        X, y = _make_survival_dataset()
        dataset = self.make_dataset(X, y)
        step = ActWeibullAFT()

        with mock.patch.object(ActWeibullAFT, "_resolve_backend", return_value=(None, None)):
            with self.assertRaises(ImportError):
                self.fit_step(step, dataset)


@unittest.skipIf(ActWeibullAFT is None, SKIP_REASON)
@unittest.skipUnless(BACKEND_AVAILABLE, "sksurv or lifelines not available")
class TestActWeibullAFT(StepTestCase):
    def test_fit_predicts_and_sets_backend(self) -> None:
        X, y = _make_survival_dataset()
        dataset = self.make_dataset(X, y)
        step = ActWeibullAFT()

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIn(step.backend, ("sksurv", "lifelines"))
        self.assertIn(step.model.__class__.__name__, ("WeibullAFT", "WeibullAFTFitter"))
        self.assertCountEqual(step.columns, ["age", "marker"])

        preds = step.predict(X)
        preds_array = np.asarray(preds, dtype=float)
        self.assertEqual(preds_array.shape, (len(X),))
        self.assertTrue(np.isfinite(preds_array).all())

        if step.backend == "lifelines":
            self.assertIsNotNone(step._lifelines_time_col)
            self.assertIsNotNone(step._lifelines_event_col)

    def test_suitable_requires_survival_numeric_and_backend(self) -> None:
        X, y = _make_survival_dataset()
        dataset_survival = self.make_dataset(X, y)
        dataset_non_survival = self.make_dataset(X, [0, 1] * (len(X) // 2))
        X_non_numeric = pd.DataFrame({"group": ["A", "B"] * (len(X) // 2)})
        dataset_non_numeric = self.make_dataset(X_non_numeric, y)
        step = ActWeibullAFT()

        self.assertTrue(step.suitable(dataset_survival))
        self.assertFalse(step.suitable(dataset_non_survival))
        self.assertFalse(step.suitable(dataset_non_numeric))
