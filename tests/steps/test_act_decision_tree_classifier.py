"""Tests for ActDecisionTreeClassifier step."""
import importlib.machinery
from pathlib import Path
import sys
import types
import unittest

try:
    import sklearn  # noqa: F401
except Exception as exc:
    raise unittest.SkipTest(
        "scikit-learn is required for ActDecisionTreeClassifier tests"
    ) from exc

# Avoid importing iaml/__init__ and actionables/__init__ during test discovery.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_IAML_ROOT = _PROJECT_ROOT / "src" / "iaml"


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    package.__package__ = name
    package.__spec__ = importlib.machinery.ModuleSpec(
        name=name,
        loader=None,
        is_package=True,
    )
    package.__spec__.submodule_search_locations = [str(path)]
    sys.modules[name] = package


_ensure_package("iaml", _IAML_ROOT)
_ensure_package("iaml.actionables", _IAML_ROOT / "actionables")
_ensure_package("iaml.actionables.predictors", _IAML_ROOT / "actionables" / "predictors")
_ensure_package(
    "iaml.actionables.predictors.classifier",
    _IAML_ROOT / "actionables" / "predictors" / "classifier",
)

import numpy as np
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.predictors.classifier.act_decision_tree_classifier import (
    ActDecisionTreeClassifier,
)


class TestActDecisionTreeClassifier(StepTestCase):
    def _make_features(self, include_text: bool = False) -> pd.DataFrame:
        data = {
            "num1": [0, 1, 2, 3, 4, 5],
            "num2": [1.0, 0.5, 0.0, 0.5, 1.0, 0.0],
        }
        if include_text:
            data["text"] = ["a", "b", "a", "b", "a", "b"]
        return pd.DataFrame(data)

    def test_fit_predict_and_proba(self) -> None:
        df = self._make_features()
        y = [0, 1, 0, 1, 0, 1]
        step = ActDecisionTreeClassifier()
        step.configure("random_state", 7)
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))
        self.assertSetEqual(set(step.classes_), {0, 1})
        self.assertTrue(set(predictions).issubset({0, 1}))

        proba = step.predict_proba(df)
        self.assertEqual(proba.shape, (len(df), 2))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))

    def test_prefers_numeric_columns(self) -> None:
        df = self._make_features(include_text=True)
        y = [0, 1, 0, 1, 0, 1]
        step = ActDecisionTreeClassifier()
        step.configure("random_state", 1)
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        self.assertListEqual(step.columns, ["num1", "num2"])

        predictions_full = step.predict(df)
        predictions_numeric = step.predict(df[["num1", "num2"]])
        self.assertListEqual(list(predictions_full), list(predictions_numeric))

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [0, 1, 0, 1, 0, 1]
        step = ActDecisionTreeClassifier()
        step.configure(
            {
                "max_depth": 3,
                "min_samples_leaf": 2,
                "min_samples_split": 3,
                "max_features": 0.5,
                "criterion": "entropy",
                "splitter": "random",
                "class_weight": "balanced",
                "random_state": 13,
            }
        )
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        params = step.model.get_params()
        self.assertEqual(params["max_depth"], 3)
        self.assertEqual(params["min_samples_leaf"], 2)
        self.assertEqual(params["min_samples_split"], 3)
        self.assertAlmostEqual(params["max_features"], 0.5)
        self.assertEqual(params["criterion"], "entropy")
        self.assertEqual(params["splitter"], "random")
        self.assertEqual(params["class_weight"], "balanced")
        self.assertEqual(params["random_state"], 13)

    def test_suitable_target_and_numeric_features(self) -> None:
        df = self._make_features()
        step = ActDecisionTreeClassifier()

        binary = self.make_dataset(df, y=["yes", "no", "yes", "no", "yes", "no"])
        multiclass = self.make_dataset(df, y=["a", "b", "c", "a", "b", "c"])
        multilabel = self.make_dataset(
            df,
            y=[
                [True, False],
                [False, True],
                [True, True],
                [False, False],
                [True, False],
                [False, True],
            ],
        )
        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        text_only = self.make_dataset(
            pd.DataFrame({"text": ["a", "b", "a", "b"]}),
            y=["yes", "no", "yes", "no"],
        )

        self.assertTrue(step.suitable(binary))
        self.assertTrue(step.suitable(multiclass))
        self.assertTrue(step.suitable(multilabel))
        self.assertFalse(step.suitable(continuous))
        self.assertFalse(step.suitable(text_only))
