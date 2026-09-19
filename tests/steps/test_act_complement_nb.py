"""Tests for ActComplementNB step."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
import unittest

import numpy as np
import pandas as pd

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None

if SKLEARN_AVAILABLE:
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
    from iaml.actionables.predictors.classifier.act_complement_nb import (
        ActComplementNB,
    )
else:
    StepTestCase = unittest.TestCase
    ActComplementNB = None


@unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required")
class TestActComplementNB(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [1, 0, 3, 0, 2, 1],
                "f2": [0, 1, 0, 2, 1, 0],
                "f3": [2, 1, 1, 0, 3, 2],
            }
        )

    def test_fit_predict_and_proba(self) -> None:
        df = self._make_features()
        y = [0, 1, 0, 1, 1, 0]
        step = ActComplementNB()
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)

        predictions = np.asarray(step.predict(df)).ravel()
        self.assertEqual(len(predictions), len(df))
        self.assertSetEqual(set(step.classes_), set(y))
        self.assertTrue(set(predictions).issubset(set(step.classes_)))

        proba = np.asarray(step.predict_proba(df))
        self.assertEqual(proba.shape, (len(df), len(step.classes_)))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))

    def test_prefers_numeric_columns_and_score(self) -> None:
        df = pd.DataFrame(
            {
                "f1": [0, 1, 2, 3, 4, 5],
                "f2": [1, 0, 1, 0, 1, 0],
                "text": ["a", "b", "a", "b", "a", "b"],
            }
        )
        y = [0, 1, 0, 1, 1, 0]
        step = ActComplementNB()
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        self.assertListEqual(step.columns, ["f1", "f2"])

        predictions_full = step.predict(df)
        predictions_numeric = step.predict(df[["f1", "f2"]])
        self.assertListEqual(list(predictions_full), list(predictions_numeric))

        score_full = step.score(df, y)
        score_numeric = step.model.score(df[["f1", "f2"]], y)
        self.assertAlmostEqual(score_full, score_numeric)

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [0, 1, 0, 1, 1, 0]
        step = ActComplementNB()
        step.configure("alpha", 0.25)
        step.configure("fit_prior", False)
        step.configure("norm", True)
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        params = step.model.get_params()
        self.assertTrue(np.allclose(params["alpha"], 0.25))
        self.assertFalse(params["fit_prior"])
        self.assertTrue(params["norm"])

    def test_suitable_target_types_and_non_negative(self) -> None:
        df = pd.DataFrame({"f1": list(range(12)), "f2": [1, 0] * 6})
        step = ActComplementNB()

        binary = self.make_dataset(df, y=[0, 1] * 6)
        multiclass = self.make_dataset(df, y=["a", "b", "c"] * 4)
        continuous = self.make_dataset(df, y=np.linspace(0.1, 1.2, len(df)))
        negative = self.make_dataset(
            pd.DataFrame({"f1": [-1] + list(range(1, 12)), "f2": [1, 0] * 6}),
            y=[0, 1] * 6,
        )

        self.assertTrue(step.suitable(binary))
        self.assertTrue(step.suitable(multiclass))
        self.assertFalse(step.suitable(continuous))
        self.assertFalse(step.suitable(negative))
        with self.assertRaises(ValueError):
            step.fit(negative)
