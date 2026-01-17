"""Tests for ActBaggingClassifier step."""
from __future__ import annotations

from pathlib import Path
import sys
import types
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
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import BaggingClassifier

    from .step_test_case import StepTestCase

    from iaml.actionables.predictors.classifier.act_bagging_classifier import (
        ActBaggingClassifier,
    )
except ImportError as exc:
    _IMPORT_ERROR = exc
    np = None
    pd = None
    BaggingClassifier = None
    StepTestCase = unittest.TestCase
    ActBaggingClassifier = None


class TestActBaggingClassifier(StepTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if _IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"scikit-learn is required: {_IMPORT_ERROR}")

    def _make_features(self, include_text: bool = False) -> pd.DataFrame:
        data = {
            "f1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "f2": [1.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "f3": [2.5, 1.5, 3.0, 0.5, 4.0, 2.0, 3.5, 1.0],
        }
        if include_text:
            data["text"] = ["a", "b", "a", "b", "a", "b", "a", "b"]
        return pd.DataFrame(data)

    def test_fit_predict_proba_and_score_prefers_numeric(self) -> None:
        df = self._make_features(include_text=True)
        y = [0, 1, 0, 1, 0, 1, 0, 1]
        step = ActBaggingClassifier()
        step.configure("random_state", 7)
        step.configure("n_estimators", 5)
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, BaggingClassifier)
        self.assertListEqual(step.columns, ["f1", "f2", "f3"])

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))
        self.assertSetEqual(set(step.classes_), {0, 1})
        self.assertTrue(set(predictions).issubset({0, 1}))

        predictions_numeric = step.predict(df[["f1", "f2", "f3"]])
        self.assertListEqual(list(predictions), list(predictions_numeric))

        proba = step.predict_proba(df)
        self.assertEqual(proba.shape, (len(df), 2))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))

        score = step.score(df, y)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [0, 1, 0, 1, 0, 1, 0, 1]
        step = ActBaggingClassifier()
        step.configure(
            {
                "n_estimators": 7,
                "max_samples": 0.8,
                "max_features": 0.6,
                "bootstrap": False,
                "bootstrap_features": True,
                "oob_score": False,
                "random_state": 13,
            }
        )
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        params = step.model.get_params()
        self.assertEqual(params["n_estimators"], 7)
        self.assertAlmostEqual(params["max_samples"], 0.8)
        self.assertAlmostEqual(params["max_features"], 0.6)
        self.assertFalse(params["bootstrap"])
        self.assertTrue(params["bootstrap_features"])
        self.assertFalse(params["oob_score"])
        self.assertEqual(params["random_state"], 13)

    def test_suitable_target_types_and_requires_numeric(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActBaggingClassifier()

        binary = self.make_dataset(df, y=["yes", "no", "yes", "no"])
        multiclass = self.make_dataset(df, y=["a", "b", "c", "a"])
        multilabel = self.make_dataset(
            df, y=[[True, False], [False, True], [True, True], [False, False]]
        )
        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4])
        text_only = self.make_dataset(
            pd.DataFrame({"text": ["a", "b", "a", "b"]}),
            y=["yes", "no", "yes", "no"],
        )

        self.assertTrue(step.suitable(binary))
        self.assertTrue(step.suitable(multiclass))
        self.assertTrue(step.suitable(multilabel))
        self.assertFalse(step.suitable(continuous))
        self.assertFalse(step.suitable(text_only))
