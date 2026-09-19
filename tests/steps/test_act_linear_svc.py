"""Tests for ActLinearSVC step."""
from __future__ import annotations

import sys
import types
from pathlib import Path
import unittest


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
    _ensure_package(
        "iaml.actionables.predictors",
        iaml_root / "actionables" / "predictors",
    )
    _ensure_package(
        "iaml.actionables.predictors.classifier",
        iaml_root / "actionables" / "predictors" / "classifier",
    )


_bootstrap_iaml()

_IMPORT_ERROR = None
try:
    import pandas as pd
    from sklearn.svm import LinearSVC

    from .step_test_case import StepTestCase

    from iaml.actionables.predictors.classifier.act_linear_svc import ActLinearSVC
except ImportError as exc:
    _IMPORT_ERROR = exc
    pd = None
    LinearSVC = None
    StepTestCase = unittest.TestCase
    ActLinearSVC = None


class TestActLinearSVC(StepTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if _IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"optional dependency missing: {_IMPORT_ERROR}")

    def _make_features(self, include_text: bool = False) -> pd.DataFrame:
        data = {
            "f1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "f2": [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            "f3": [2.5, 1.5, 3.0, 0.5, 4.0, 2.0],
        }
        if include_text:
            data["cat"] = ["a", "b", "a", "b", "a", "b"]
        return pd.DataFrame(data)

    def test_fit_predict_and_score_with_numeric_columns(self) -> None:
        df = self._make_features(include_text=True)
        y = [0, 1, 0, 1, 1, 0]
        step = ActLinearSVC()
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, LinearSVC)
        self.assertListEqual(step.columns, ["f1", "f2", "f3"])

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))
        self.assertSetEqual(set(step.classes_), {0, 1})
        self.assertTrue(set(predictions).issubset({0, 1}))

        numeric_df = df[["f1", "f2", "f3"]]
        predictions_numeric = step.predict(numeric_df)
        self.assertListEqual(list(predictions), list(predictions_numeric))

        df_extra = df.copy()
        df_extra["extra"] = ["x", "y", "x", "y", "x", "y"]
        score = step.score(df_extra, y)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [0, 1, 0, 1, 1, 0]
        step = ActLinearSVC()
        step.configure(
            {
                "C": 0.5,
                "penalty": "l1",
                "loss": "squared_hinge",
                "dual": False,
                "tol": 1e-3,
                "max_iter": 200,
                "fit_intercept": False,
                "class_weight": "balanced",
                "random_state": 7,
            }
        )
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        params = step.model.get_params()
        self.assertAlmostEqual(params["C"], 0.5)
        self.assertEqual(params["penalty"], "l1")
        self.assertEqual(params["loss"], "squared_hinge")
        self.assertFalse(params["dual"])
        self.assertAlmostEqual(params["tol"], 1e-3)
        self.assertEqual(params["max_iter"], 200)
        self.assertFalse(params["fit_intercept"])
        self.assertEqual(params["class_weight"], "balanced")
        self.assertEqual(params["random_state"], 7)

    def test_suitable_target_types_and_requires_numeric(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActLinearSVC()

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

        text_only = self.make_dataset(
            pd.DataFrame({"cat": ["a", "b", "a", "b"]}),
            y=["yes", "no", "yes", "no"],
        )
        self.assertFalse(step.suitable(text_only))
