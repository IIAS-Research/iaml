"""Tests for ActRidgeClassifier step."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest

try:
    import pandas as pd
    from sklearn.linear_model import RidgeClassifier
    _HAS_DEPS = True
except Exception:
    pd = None
    RidgeClassifier = None
    _HAS_DEPS = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IAML_PATH = PROJECT_ROOT / "src" / "iaml"


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


def _load_act_ridge_classifier():
    _ensure_package("iaml", IAML_PATH)
    _ensure_package("iaml.actionables", IAML_PATH / "actionables")
    _ensure_package("iaml.actionables.predictors", IAML_PATH / "actionables" / "predictors")
    _ensure_package(
        "iaml.actionables.predictors.classifier",
        IAML_PATH / "actionables" / "predictors" / "classifier",
    )

    module_name = "iaml.actionables.predictors.classifier.act_ridge_classifier"
    module = sys.modules.get(module_name)
    if module is None:
        module_path = (
            IAML_PATH / "actionables" / "predictors" / "classifier" / "act_ridge_classifier.py"
        )
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load {module_name} from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.ActRidgeClassifier


if _HAS_DEPS:
    _ensure_package("iaml", IAML_PATH)
    from .step_test_case import StepTestCase
    ActRidgeClassifier = _load_act_ridge_classifier()
else:
    class StepTestCase(unittest.TestCase):
        pass

    ActRidgeClassifier = None


@unittest.skipUnless(_HAS_DEPS, "Optional dependency missing: pandas or scikit-learn")
class TestActRidgeClassifier(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "f2": [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                "cat": ["a", "b", "a", "b", "a", "b"],
            }
        )

    def test_fit_predict_and_score_with_mixed_columns(self) -> None:
        df = self._make_features()
        y = [0, 1, 0, 1, 1, 0]
        step = ActRidgeClassifier()
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, RidgeClassifier)
        self.assertListEqual(step.columns, ["f1", "f2"])

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))
        self.assertSetEqual(set(step.classes_), {0, 1})
        self.assertTrue(set(predictions).issubset({0, 1}))

        df_extra = df.copy()
        df_extra["extra"] = ["x", "y", "x", "y", "x", "y"]
        score = step.score(df_extra, y)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [0, 1, 0, 1, 1, 0]
        step = ActRidgeClassifier()
        step.configure("alpha", 0.5)
        step.configure("fit_intercept", False)
        step.configure("solver", "auto")
        step.configure("tol", 1e-3)
        step.configure("max_iter", 200)
        step.configure("class_weight", "balanced")
        step.configure("random_state", 7)
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        params = step.model.get_params()
        self.assertAlmostEqual(params["alpha"], 0.5)
        self.assertFalse(params["fit_intercept"])
        self.assertEqual(params["solver"], "auto")
        self.assertAlmostEqual(params["tol"], 1e-3)
        self.assertEqual(params["max_iter"], 200)
        self.assertEqual(params["class_weight"], "balanced")
        self.assertEqual(params["random_state"], 7)

    def test_suitable_target_types_and_requires_numeric(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActRidgeClassifier()

        binary = self.make_dataset(df, y=["yes", "no", "yes", "no"])
        multiclass = self.make_dataset(df, y=["a", "b", "c", "a"])
        multilabel = self.make_dataset(
            df, y=[[True, False], [False, True], [True, True], [False, False]]
        )
        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4])

        self.assertTrue(step.suitable(binary))
        self.assertTrue(step.suitable(multiclass))
        self.assertTrue(step.suitable(multilabel))
        self.assertFalse(step.suitable(continuous))

        df_no_numeric = pd.DataFrame({"cat": ["a", "b", "a", "b"]})
        no_numeric = self.make_dataset(df_no_numeric, y=["yes", "no", "yes", "no"])
        self.assertFalse(step.suitable(no_numeric))
