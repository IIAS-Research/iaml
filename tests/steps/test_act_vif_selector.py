"""Tests for ActVIFSelector."""
from __future__ import annotations

import sys
import types
from pathlib import Path
import unittest

try:
    import numpy as np
    import pandas as pd
    from pandas.testing import assert_frame_equal
except ImportError as exc:  # pragma: no cover - optional test dependency
    raise unittest.SkipTest(f"Optional dependency missing: {exc.name}") from exc


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_IAML_PATH = _PROJECT_ROOT / "src" / "iaml"


def _import_act_vif_selector() -> tuple[type, type]:
    import importlib

    existing = set(sys.modules)

    def _cleanup() -> None:
        for name in list(sys.modules):
            if name.startswith("iaml") and name not in existing:
                sys.modules.pop(name, None)

    def _ensure_package(name: str, path: Path) -> None:
        if name in sys.modules:
            return
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package

    _ensure_package("iaml", _IAML_PATH)
    _ensure_package("iaml.actionables", _IAML_PATH / "actionables")
    _ensure_package(
        "iaml.actionables.features_selection",
        _IAML_PATH / "actionables" / "features_selection",
    )

    if "iaml.candidate" not in sys.modules:
        candidate_module = types.ModuleType("iaml.candidate")

        class Candidate:  # pragma: no cover - stub only
            pass

        candidate_module.Candidate = Candidate
        sys.modules["iaml.candidate"] = candidate_module

    try:
        dataset_module = importlib.import_module("iaml.dataset")
        Dataset = dataset_module.Dataset
    except Exception:
        sys.modules.pop("iaml.dataset", None)
        dataset_module = types.ModuleType("iaml.dataset")

        class Dataset:  # pragma: no cover - stub only
            def __init__(self, X: pd.DataFrame, y=None) -> None:
                self.X = X
                if y is None:
                    y = [0] * len(X)
                self.y = np.array(y)

            def get_columns_names_by_type(self, data_type) -> list[str]:
                if getattr(data_type, "name", None) not in (None, "NUMERIC"):
                    return []
                return [
                    col for col in self.X.columns
                    if pd.api.types.is_numeric_dtype(self.X[col])
                ]

        dataset_module.Dataset = Dataset
        sys.modules["iaml.dataset"] = dataset_module

    try:
        module = importlib.import_module(
            "iaml.actionables.features_selection.act_vif_selector"
        )
        ActVIFSelector = module.ActVIFSelector
    except Exception as exc:  # pragma: no cover - optional dependencies
        _cleanup()
        raise unittest.SkipTest(f"Unable to import ActVIFSelector: {exc}") from exc

    _cleanup()

    return ActVIFSelector, Dataset


ActVIFSelector, Dataset = _import_act_vif_selector()


class TestActVIFSelector(unittest.TestCase):
    def make_dataset(self, X: pd.DataFrame, y=None) -> Dataset:
        return Dataset(X, y)

    def fit_step(self, step: object, dataset: Dataset) -> object:
        return step.fit(dataset)

    def apply_transform(self, step: object, X: pd.DataFrame, y=None) -> pd.DataFrame:
        dataset = self.make_dataset(X, y)
        self.fit_step(step, dataset)
        if not hasattr(step, "transform"):
            raise AttributeError("Step has no transform method")
        return step.transform(dataset.X.copy())

    def assertFrameEqual(self, left: pd.DataFrame, right: pd.DataFrame, **kwargs) -> None:
        try:
            assert_frame_equal(left, right, **kwargs)
        except AssertionError as exc:
            self.fail(str(exc))

    def test_drops_high_vif_columns(self) -> None:
        df = pd.DataFrame(
            {
                "signal": [1, 2, 3, 4, 5, 6, 7, 8],
                "collinear": [2, 4, 6, 8, 10, 12, 14, 16.1],
                "other": [1, 0, 1, 0, 1, 0, 1, 0],
            }
        )
        dataset = self.make_dataset(df)
        step = ActVIFSelector()

        self.assertTrue(step.suitable(dataset))

        self.fit_step(step, dataset)

        self.assertCountEqual(step.to_drop, ["signal", "collinear"])

        result = step.transform(dataset.X.copy())

        expected = df[["other"]].copy()
        self.assertFrameEqual(result, expected)

    def test_keeps_columns_when_vif_below_threshold(self) -> None:
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": [1, 0, 2, 1, 3, 0],
            }
        )
        dataset = self.make_dataset(df)
        step = ActVIFSelector()

        self.assertFalse(step.suitable(dataset))

        result = self.apply_transform(step, df)

        self.assertFrameEqual(result, df)

    def test_constant_column_does_not_trigger_drop(self) -> None:
        df = pd.DataFrame(
            {
                "constant": [1, 1, 1, 1],
                "signal": [1, 2, 3, 4],
            }
        )
        dataset = self.make_dataset(df)
        step = ActVIFSelector()

        self.assertFalse(step.suitable(dataset))

        self.fit_step(step, dataset)

        self.assertEqual(step.to_drop, [])

        result = step.transform(dataset.X.copy())

        self.assertFrameEqual(result, df)
