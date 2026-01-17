"""Tests for ActRANSACRegressor step."""
from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
IAML_PATH = SRC_PATH / "iaml"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _ensure_package(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    if not hasattr(module, "__path__"):
        module.__path__ = [str(path)]
    module.__package__ = name


SKLEARN_AVAILABLE = False
RANSACRegressor = None
ActRANSACRegressor = None
StepTestCase = unittest.TestCase
_IMPORT_ERROR = None

try:
    from sklearn.linear_model import RANSACRegressor
    SKLEARN_AVAILABLE = True
except Exception as exc:  # pragma: no cover - optional dependency guard
    _IMPORT_ERROR = exc

if SKLEARN_AVAILABLE:
    try:
        # Avoid triggering package __init__ imports that pull unrelated modules.
        _ensure_package("iaml", IAML_PATH)
        _ensure_package("iaml.actionables", IAML_PATH / "actionables")
        _ensure_package(
            "iaml.actionables.predictors", IAML_PATH / "actionables" / "predictors"
        )
        _ensure_package(
            "iaml.actionables.predictors.regressor",
            IAML_PATH / "actionables" / "predictors" / "regressor",
        )

        from .step_test_case import StepTestCase
        from iaml.actionables.predictors.regressor.act_ransac_regressor import (
            ActRANSACRegressor,
        )
    except Exception as exc:  # pragma: no cover - optional dependency guard
        _IMPORT_ERROR = exc
        ActRANSACRegressor = None
        StepTestCase = unittest.TestCase


_SKIP_REASON = None
if not SKLEARN_AVAILABLE:
    _SKIP_REASON = "scikit-learn is required for ActRANSACRegressor tests"
elif ActRANSACRegressor is None:
    _SKIP_REASON = f"ActRANSACRegressor unavailable ({_IMPORT_ERROR!r})"


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON)
class TestActRANSACRegressor(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "num1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "num2": [1.0, 0.5, 1.5, 2.0, 2.5, 3.0],
            }
        )

    def test_fit_sets_model_and_scores(self) -> None:
        df = self._make_features()
        y = [2.0 + 1.5 * f1 - 0.3 * f2 for f1, f2 in zip(df["num1"], df["num2"])]
        step = ActRANSACRegressor()
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, RANSACRegressor)

        predictions = np.asarray(step.predict(df))
        self.assertEqual(predictions.shape, (len(df),))
        self.assertGreater(step.score(df, y), 0.98)

    def test_uses_numeric_columns_only(self) -> None:
        df = self._make_features()
        df["cat"] = ["a", "b", "a", "b", "a", "b"]
        y = [1.0 + 0.5 * f1 + 2.0 * f2 for f1, f2 in zip(df["num1"], df["num2"])]
        step = ActRANSACRegressor()
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        self.assertEqual(step.columns, ["num1", "num2"])

        df_alt = df.copy()
        df_alt["cat"] = ["b", "a", "b", "a", "b", "a"]
        preds_original = step.predict(df)
        preds_alt = step.predict(df_alt)
        self.assertTrue(np.allclose(preds_original, preds_alt))

    def test_suitable_requires_continuous_and_numeric(self) -> None:
        numeric_df = pd.DataFrame({"num1": [0, 1, 2], "num2": [1, 0, 1]})
        step = ActRANSACRegressor()

        continuous = self.make_dataset(numeric_df, y=[0.1, 0.2, 0.3])
        categorical = self.make_dataset(numeric_df, y=["low", "high", "low"])
        text_df = pd.DataFrame({"cat": ["a", "b", "c"]})
        no_numeric = self.make_dataset(text_df, y=[0.1, 0.2, 0.3])

        self.assertTrue(step.suitable(continuous))
        self.assertFalse(step.suitable(categorical))
        self.assertFalse(step.suitable(no_numeric))

    def test_priorize_returns_neutral_score(self) -> None:
        step = ActRANSACRegressor()

        self.assertEqual(step.priorize(), 0.5)
