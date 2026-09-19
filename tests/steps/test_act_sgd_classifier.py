"""Tests for ActSGDClassifier step."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
import unittest

import numpy as np
import pandas as pd

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None

if SKLEARN_AVAILABLE:
    from sklearn.linear_model import SGDClassifier

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
    from iaml.actionables.predictors.classifier.act_sgd_classifier import (
        ActSGDClassifier,
    )
else:
    SGDClassifier = None
    StepTestCase = unittest.TestCase
    ActSGDClassifier = None


@unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required")
class TestActSGDClassifier(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "f2": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                "f3": [2.5, 1.5, 3.0, 0.5, 4.0, 2.0],
            }
        )

    def test_fit_predict_and_proba_log_loss(self) -> None:
        df = self._make_features()
        y = [0, 1, 0, 1, 1, 0]
        step = ActSGDClassifier()
        step.configure("random_state", 0)
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, SGDClassifier)

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))
        self.assertSetEqual(set(step.classes_), {0, 1})
        self.assertTrue(set(predictions).issubset({0, 1}))

        proba = step.predict_proba(df)
        self.assertEqual(proba.shape, (len(df), 2))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))

    def test_predict_proba_with_hinge_loss(self) -> None:
        df = self._make_features()
        y = [0, 1, 0, 1, 1, 0]
        step = ActSGDClassifier()
        step.configure("loss", "hinge")
        step.configure("random_state", 0)
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        with self.assertRaises(AttributeError):
            step.predict_proba(df)

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [0, 1, 0, 1, 1, 0]
        step = ActSGDClassifier()
        step.configure(
            {
                "random_state": 7,
                "alpha": 0.01,
                "learning_rate": "constant",
                "eta0": 0.05,
                "penalty": "l1",
                "max_iter": 200,
                "tol": 1e-3,
                "class_weight": "balanced",
                "average": True,
            }
        )
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        params = step.model.get_params()
        self.assertEqual(params["random_state"], 7)
        self.assertAlmostEqual(params["alpha"], 0.01)
        self.assertEqual(params["learning_rate"], "constant")
        self.assertAlmostEqual(params["eta0"], 0.05)
        self.assertEqual(params["penalty"], "l1")
        self.assertEqual(params["max_iter"], 200)
        self.assertAlmostEqual(params["tol"], 1e-3)
        self.assertEqual(params["class_weight"], "balanced")
        self.assertTrue(params["average"])

    def test_suitable_target_types(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActSGDClassifier()

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
