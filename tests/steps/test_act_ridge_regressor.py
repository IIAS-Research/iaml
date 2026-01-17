"""Tests for ActRidgeRegressor step."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest

try:
    import pandas as pd
    from sklearn.linear_model import Ridge
    _HAS_DEPS = True
except Exception:
    pd = None
    Ridge = None
    _HAS_DEPS = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IAML_PATH = PROJECT_ROOT / "src" / "iaml"


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


def _load_act_ridge_regressor():
    _ensure_package("iaml", IAML_PATH)
    _ensure_package("iaml.actionables", IAML_PATH / "actionables")
    _ensure_package("iaml.actionables.predictors", IAML_PATH / "actionables" / "predictors")
    _ensure_package(
        "iaml.actionables.predictors.regressor",
        IAML_PATH / "actionables" / "predictors" / "regressor",
    )

    module_name = "iaml.actionables.predictors.regressor.act_ridge_regressor"
    module = sys.modules.get(module_name)
    if module is None:
        module_path = (
            IAML_PATH / "actionables" / "predictors" / "regressor" / "act_ridge_regressor.py"
        )
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load {module_name} from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.ActRidgeRegressor


if _HAS_DEPS:
    _ensure_package("iaml", IAML_PATH)
    from .step_test_case import StepTestCase
    ActRidgeRegressor = _load_act_ridge_regressor()
else:
    class StepTestCase(unittest.TestCase):
        pass

    ActRidgeRegressor = None


@unittest.skipUnless(_HAS_DEPS, "Optional dependency missing: pandas or scikit-learn")
class TestActRidgeRegressor(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [0.0, 1.0, 2.0, 3.0],
                "f2": [1.5, 0.5, 2.5, 3.5],
                "cat": ["a", "b", "a", "b"],
            }
        )

    def test_fit_predicts_with_numeric_selection(self) -> None:
        df = self._make_features()
        y = [1.2 * f1 - 0.7 * f2 for f1, f2 in zip(df["f1"], df["f2"])]
        step = ActRidgeRegressor()
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, Ridge)
        self.assertListEqual(step.columns, ["f1", "f2"])

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))
        self.assertTrue(all(math.isfinite(float(pred)) for pred in predictions))

        df_extra = df.copy()
        df_extra["extra"] = ["x", "y", "x", "y"]
        predictions_extra = step.predict(df_extra)
        self.assertEqual(len(predictions_extra), len(df))
        self.assertTrue(
            all(
                math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
                for a, b in zip(predictions, predictions_extra)
            )
        )

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [0.5, 0.7, 0.9, 1.1]
        step = ActRidgeRegressor()
        step.configure("alpha", 0.5)
        step.configure("fit_intercept", False)
        step.configure("solver", "auto")
        step.configure("tol", 1e-3)
        step.configure("max_iter", 200)
        step.configure("random_state", 7)
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        params = step.model.get_params()
        self.assertAlmostEqual(params["alpha"], 0.5)
        self.assertFalse(params["fit_intercept"])
        self.assertEqual(params["solver"], "auto")
        self.assertAlmostEqual(params["tol"], 1e-3)
        self.assertEqual(params["max_iter"], 200)
        self.assertEqual(params["random_state"], 7)

    def test_suitable_requires_continuous_target_and_numeric_features(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActRidgeRegressor()

        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4])
        categorical = self.make_dataset(df, y=["low", "high", "low", "high"])

        self.assertTrue(step.suitable(continuous))
        self.assertFalse(step.suitable(categorical))

        df_no_numeric = pd.DataFrame({"cat": ["a", "b", "a", "b"]})
        no_numeric = self.make_dataset(df_no_numeric, y=[0.1, 0.2, 0.3, 0.4])
        self.assertFalse(step.suitable(no_numeric))
