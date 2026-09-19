"""Tests for ActDecisionTreeRegressor step."""
import importlib.machinery
from pathlib import Path
import sys
import types
import unittest

try:
    import sklearn  # noqa: F401
except Exception as exc:
    raise unittest.SkipTest(
        "scikit-learn is required for ActDecisionTreeRegressor tests"
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
    "iaml.actionables.predictors.regressor",
    _IAML_ROOT / "actionables" / "predictors" / "regressor",
)

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from .step_test_case import StepTestCase
from iaml.actionables.predictors.regressor.act_decision_tree_regressor import (
    ActDecisionTreeRegressor,
)


class TestActDecisionTreeRegressor(StepTestCase):
    def _make_features(self, include_text: bool = False) -> pd.DataFrame:
        data = {
            "num1": [0, 1, 2, 3, 4, 5],
            "num2": [1.0, 0.5, 0.0, 0.5, 1.0, 0.0],
        }
        if include_text:
            data["text"] = ["a", "b", "a", "b", "a", "b"]
        return pd.DataFrame(data)

    def test_fit_sets_model_and_predicts(self) -> None:
        df = self._make_features()
        y = [value * 1.5 + 0.2 for value in df["num1"]]
        step = ActDecisionTreeRegressor()
        step.configure("random_state", 7)
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, DecisionTreeRegressor)

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))

    def test_prefers_numeric_columns(self) -> None:
        df = self._make_features(include_text=True)
        y = [2.0, 1.5, 1.0, 2.5, 3.0, 2.0]
        step = ActDecisionTreeRegressor()
        step.configure("random_state", 3)
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        self.assertListEqual(step.columns, ["num1", "num2"])

        predictions_full = step.predict(df)
        predictions_numeric = step.predict(df[["num1", "num2"]])
        self.assertTrue(np.allclose(predictions_full, predictions_numeric))

        score_full = step.score(df, y)
        score_numeric = step.score(df[["num1", "num2"]], y)
        self.assertAlmostEqual(score_full, score_numeric)

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [0.5, 0.7, 1.1, 1.3, 1.7, 1.9]
        step = ActDecisionTreeRegressor()
        step.configure(
            {
                "max_depth": 3,
                "min_samples_leaf": 2,
                "min_samples_split": 4,
                "max_features": 0.5,
                "criterion": "friedman_mse",
                "splitter": "random",
                "random_state": 11,
            }
        )
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        params = step.model.get_params()
        self.assertEqual(params["max_depth"], 3)
        self.assertEqual(params["min_samples_leaf"], 2)
        self.assertEqual(params["min_samples_split"], 4)
        self.assertAlmostEqual(params["max_features"], 0.5)
        self.assertEqual(params["criterion"], "friedman_mse")
        self.assertEqual(params["splitter"], "random")
        self.assertEqual(params["random_state"], 11)

    def test_suitable_target_and_numeric_features(self) -> None:
        df = self._make_features()
        step = ActDecisionTreeRegressor()

        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        categorical = self.make_dataset(
            df,
            y=["low", "high", "low", "high", "low", "high"],
        )
        text_only = self.make_dataset(
            pd.DataFrame({"text": ["a", "b", "c", "d"]}),
            y=[0.1, 0.2, 0.3, 0.4],
        )

        self.assertTrue(step.suitable(continuous))
        self.assertFalse(step.suitable(categorical))
        self.assertFalse(step.suitable(text_only))
