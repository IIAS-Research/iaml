"""Tests for ActPassiveAggressiveClassifier step."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
import unittest

import numpy as np
import pandas as pd

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None

if SKLEARN_AVAILABLE:
    from sklearn.linear_model import PassiveAggressiveClassifier

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    SRC_PATH = PROJECT_ROOT / "src"
    if str(SRC_PATH) not in sys.path:
        sys.path.append(str(SRC_PATH))

    def _ensure_package(name: str, path: Path) -> None:
        if name in sys.modules:
            return
        package = ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package

    _ensure_package("iaml", SRC_PATH / "iaml")
    _ensure_package("iaml.actionables", SRC_PATH / "iaml" / "actionables")
    _ensure_package(
        "iaml.actionables.predictors",
        SRC_PATH / "iaml" / "actionables" / "predictors",
    )
    _ensure_package(
        "iaml.actionables.predictors.classifier",
        SRC_PATH / "iaml" / "actionables" / "predictors" / "classifier",
    )

    from .step_test_case import StepTestCase
    from iaml.actionables.predictors.classifier.act_passive_aggressive_classifier import (
        ActPassiveAggressiveClassifier,
    )
else:
    PassiveAggressiveClassifier = None
    StepTestCase = unittest.TestCase
    ActPassiveAggressiveClassifier = None


@unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required")
class TestActPassiveAggressiveClassifier(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "f2": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                "f3": [2.5, 1.5, 3.0, 0.5, 4.0, 2.0],
            }
        )

    def test_fit_predict_binary(self) -> None:
        df = self._make_features()
        y = [0, 1, 0, 1, 1, 0]
        step = ActPassiveAggressiveClassifier()
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, PassiveAggressiveClassifier)

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))
        self.assertSetEqual(set(step.classes_), {0, 1})
        self.assertTrue(set(predictions).issubset({0, 1}))

    def test_predict_proba_behavior(self) -> None:
        df = self._make_features()
        y = [0, 1, 0, 1, 1, 0]
        step = ActPassiveAggressiveClassifier()
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        if hasattr(step.model, "predict_proba"):
            proba = step.predict_proba(df)
            self.assertEqual(proba.shape, (len(df), 2))
            self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))
        else:
            with self.assertRaises(AttributeError):
                step.predict_proba(df)

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [0, 1, 0, 1, 1, 0]
        step = ActPassiveAggressiveClassifier()
        step.configure(
            {
                "C": 0.5,
                "loss": "squared_hinge",
                "max_iter": 200,
                "tol": 1e-3,
                "fit_intercept": False,
                "shuffle": False,
                "class_weight": "balanced",
                "average": True,
                "random_state": 7,
            }
        )
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        params = step.model.get_params()
        self.assertAlmostEqual(params["C"], 0.5)
        self.assertEqual(params["loss"], "squared_hinge")
        self.assertEqual(params["max_iter"], 200)
        self.assertAlmostEqual(params["tol"], 1e-3)
        self.assertFalse(params["fit_intercept"])
        self.assertFalse(params["shuffle"])
        self.assertEqual(params["class_weight"], "balanced")
        self.assertTrue(params["average"])
        self.assertEqual(params["random_state"], 7)

    def test_suitable_target_types(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActPassiveAggressiveClassifier()

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
