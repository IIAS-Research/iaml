"""Tests for ActHistGradientBoostingClassifier step."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
import unittest

import numpy as np
import pandas as pd

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None

if SKLEARN_AVAILABLE:
    from sklearn.ensemble import HistGradientBoostingClassifier

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

    # Avoid executing package __init__ files that pull unrelated steps.
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
    from iaml.actionables.predictors.classifier.act_hist_gradient_boosting_classifier import (
        ActHistGradientBoostingClassifier,
    )
else:
    HistGradientBoostingClassifier = None
    StepTestCase = unittest.TestCase
    ActHistGradientBoostingClassifier = None


@unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required")
class TestActHistGradientBoostingClassifier(StepTestCase):
    def _make_features(self, include_text: bool = False) -> pd.DataFrame:
        values = list(range(30))
        data = {
            "f1": values,
            "f2": [value % 3 for value in values],
            "f3": [value / 10 for value in values],
        }
        if include_text:
            data["text"] = ["even" if value % 2 == 0 else "odd" for value in values]
        return pd.DataFrame(data)

    def test_fit_predict_and_proba(self) -> None:
        df = self._make_features()
        y = [value % 2 for value in range(len(df))]
        step = ActHistGradientBoostingClassifier()
        step.configure("max_iter", 20)
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, HistGradientBoostingClassifier)

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))
        self.assertSetEqual(set(step.classes_), {0, 1})
        self.assertTrue(set(predictions).issubset({0, 1}))

        proba = step.predict_proba(df)
        self.assertEqual(proba.shape, (len(df), 2))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))

    def test_prefers_numeric_columns(self) -> None:
        df = self._make_features(include_text=True)
        y = [value % 2 for value in range(len(df))]
        step = ActHistGradientBoostingClassifier()
        step.configure("random_state", 1)
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        self.assertListEqual(step.columns, ["f1", "f2", "f3"])

        predictions_full = step.predict(df)
        predictions_numeric = step.predict(df[["f1", "f2", "f3"]])
        self.assertListEqual(list(predictions_full), list(predictions_numeric))

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [value % 2 for value in range(len(df))]
        step = ActHistGradientBoostingClassifier()
        step.configure(
            {
                "learning_rate": 0.2,
                "max_iter": 25,
                "max_leaf_nodes": 7,
                "max_depth": 3,
                "min_samples_leaf": 2,
                "l2_regularization": 0.5,
                "max_bins": 32,
                "validation_fraction": 0.2,
                "n_iter_no_change": 5,
                "tol": 1e-3,
                "class_weight": "balanced",
                "random_state": 7,
            }
        )
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        params = step.model.get_params()
        self.assertAlmostEqual(params["learning_rate"], 0.2)
        self.assertEqual(params["max_iter"], 25)
        self.assertEqual(params["max_leaf_nodes"], 7)
        self.assertEqual(params["max_depth"], 3)
        self.assertEqual(params["min_samples_leaf"], 2)
        self.assertAlmostEqual(params["l2_regularization"], 0.5)
        self.assertEqual(params["max_bins"], 32)
        self.assertAlmostEqual(params["validation_fraction"], 0.2)
        self.assertEqual(params["n_iter_no_change"], 5)
        self.assertAlmostEqual(params["tol"], 1e-3)
        self.assertEqual(params["class_weight"], "balanced")
        self.assertEqual(params["random_state"], 7)

    def test_suitable_target_and_numeric_features(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1.0, 0.0, 1.0, 0.0]})
        step = ActHistGradientBoostingClassifier()

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
